import { useState } from "react";
import axios from '../services/axiosServices'
import { toast } from "react-toastify";
function CopyPastQuestion() {
    const [question, setQuestion] = useState<string>("");

    const updateQuestion = (question: string) => {
        setQuestion(question);
    }

    const handleSubmit = async () => {
        if (question == "") {
            toast.error("Please enter a question");
            return;
        }
        try {
            const formData = new FormData();
            formData.append("copiedText", question);
            const res = await axios.post('/submit_copy_text/', formData);
            console.log(res.data); // []
        } catch (error) {
            console.log(error);
        }

    }
    return (<>
        <div className="each-question-container w-full h-full flex justify-center items-center">
            <div className="w-4/5 h-full flex flex-col justify-center items-center">
                <textarea
                    autoFocus={true}
                    onChange={(e) => updateQuestion(e.target.value)}
                    value={question}
                    className="px-2 py-1 rounded-lg w-full h-full border-1 border-black resize-none fixed-textarea"
                ></textarea>
                <button className="mt-3 cursor-pointer border-1 border-black fixed-button bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg"
                    onClick={handleSubmit}
                >Submit</button>
            </div>


        </div>

    </>);
}
export default CopyPastQuestion;