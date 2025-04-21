from typing import List
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse , JSONResponse
from inference import *
import torch
import uvicorn
import string

app = FastAPI()
templates = Jinja2Templates(directory="templates")


# main.py



app = FastAPI()
templates = Jinja2Templates(directory="templates")
model_name = "microsoft/codebert-base"
checkpoints = "checkpoints/best_mcqa_model.pth"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = MCQA(model_name).to(device)
model.load_state_dict(torch.load(checkpoints, map_location=device))
model.eval()


letter_labels = list(string.ascii_uppercase)



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
    options = form.getlist("options")

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
