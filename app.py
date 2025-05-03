from typing import List
from fastapi import FastAPI, Request , UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse , JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from inference import *
import torch
import uvicorn
import string
from chatbotapi import call_gemini_api
from transformers import pipeline

app = FastAPI()
templates = Jinja2Templates(directory="templates")


# main.py
# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,  # Be cautious with this in production with "*"
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

origins = [
    "http://localhost:3000",  # your React (or whatever) dev server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





# app = FastAPI()
# templates = Jinja2Templates(directory="templates")
model_name = "microsoft/codebert-base"
checkpoints = "checkpoints/best_mcqa_model.pth"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = MCQA(model_name).to(device)
model.load_state_dict(torch.load(checkpoints, map_location=device))
model.eval()

pipe = pipeline("text-classification", model="mrsinghania/asr-question-detection", device=device)


letter_labels = list(string.ascii_uppercase)
UPLOAD_DIR = "uploads"



@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    """
    Renders the index.html form.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/submit_question/")
async def submit_question(request: Request):
    """
    Manually parses the incoming form data.
    Uses request.form() to get a FormData (a MultiDict under the hood),
    then .getlist("options") to collect all inputs named "options".
    """
    form = await request.form()
    question_text = form.get("questionText")
    # form.getlist will collect every form field named "options"
    options = form.get("options").split(",")

    # DEBUG OUTPUT (should now appear in your Uvicorn console)
    inputs = {"question" : [question_text], "options" : [options]}
    with torch.no_grad():
        outputs =  model(inputs)
    print(outputs, flush=True)
    predicted_answer = get_predicted_letter(outputs[0], len(inputs['options']), letter_labels)
    print("Inputs", inputs , flush=True)
    print("Output", outputs, flush=True)
    print("Answer", predicted_answer, flush=True)

    # Here you could save to the database...
    return JSONResponse({
        "predicted_answer": predicted_answer,
        "output": outputs[0].tolist()
    })

@app.post("/submit_copy_text/")
async def submit_copy_text(request: Request):
    """
    Manually parses the incoming form data.
    Uses request.form() to get a FormData (a MultiDict under the hood),
    then .getlist("options") to collect all inputs named "options".
    """
    form = await request.form()
    copied_text = form.get("copiedText")
    # form.getlist will collect every form field named "options"
    request_sentence = """
Could you turn above text into lines response like 

questionText1,optionText1,optionText2,...

questionText2,optionText1,optionText2,...
...
"""
    question_text = []
    options = []
    try:
        call_gemini_api(copied_text +  request_sentence)
        with open('api_response.txt', 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                question_text.append(parts[0])
                options.append(parts[1:])
        print(question_text, options)

            

        # DEBUG OUTPUT (should now appear in your Uvicorn console)
        inputs = {"question" : question_text, "options" : options}
        with torch.no_grad():
            outputs =  model(inputs)
        print(outputs, flush=True)
        predicted_answer_list = [get_predicted_letter(x, len(inputs['options']), letter_labels) for x in outputs]
        print("Inputs", inputs , flush=True)
        print("Output", outputs, flush=True)
        print("Answer", predicted_answer_list, flush=True)

        # Here you could save to the database...
        return JSONResponse({
            "predicted_answer": predicted_answer_list,
            "output": [x.tolist() for x in outputs]
        })
    except Exception as e:
        print(e)
        return JSONResponse({
            "predicted_answer": None,
            "output": None
        })



@app.post("/submit_file/")
async def submit_file(request: Request):
    """
    submit a file with the following format
    Q1. questionText?
    A.option1
    B.option2
    C.option3
    ...
    Q2.questionText?
    A.option1
    B.option2
    C.option3
    ...
    """
    form = await request.form()
    print(form)
    file : UploadFile = form.get("file")
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # Read in chunks of 1MB
                f.write(chunk)
    except Exception as e:
        raise e
    finally:
        await file.close()
    questions = []
    options = []
    if file.filename:
        _ , extension = os.path.splitext(file.filename)
        if extension == ".txt": #for now , only permit txt file
            with open(UPLOAD_DIR + '/' + file.filename, "r") as f:
                questions += [f.readline().rstrip('\n')]
                helper = []
                for line in f:
                    if pipe(line.rstrip())[0]['label'] == 'LABEL_1':
                        questions.append(line.rstrip('\n'))
                        options += [helper]
                        helper = []
                    else:
                        helper.append(line.rstrip('\n'))
                if len(helper) != 0 :
                    options += [helper]
            
            print(questions)
            print(options)
            inputs = {"question" : questions, "options" : options}
            with torch.no_grad():
                outputs =  model(inputs)
            print(outputs, flush=True)
            predicted_answer_list = [get_predicted_letter(x, len(inputs['options']), letter_labels) for x in outputs]
            print("Inputs", inputs , flush=True)
            print("Output", outputs, flush=True)
            print("Answer", predicted_answer_list, flush=True)

            # Here you could save to the database...
            return JSONResponse({
                "predicted_answer": predicted_answer_list,
                "output": [x.tolist() for x in outputs]
            })
        else:
            return JSONResponse({
                "message" : "Unsupported file type"
            })
    else:
        return JSONResponse({
            "message" : "No file uploaded"
        })







if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
