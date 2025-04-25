import { useRef, useState } from "react";
import { toast } from "react-toastify";
import axios from '../services/axiosServices'
function EachQuestion() {
    const [question, setQuestion] = useState<string>("");
    const [answerBlocks, setAnswerBlocks] = useState<{ id: number, value: string }[]>([{ id: 1, value: "" }]);
    const [current, setCurrent] = useState<number>(0);
    const [focusOnQuestion, setFocusOnQuestion] = useState<boolean>(false);
    const addAnswerBlock = () => {
        const newBlock = { id: answerBlocks.length + 1, value: "" };
        setCurrent(newBlock.id);
        setAnswerBlocks([...answerBlocks, newBlock]);
    }
    const updateQuestion = (question: string) => {
        setQuestion(question);
    }
    const handleKeyDown = (e: any, id: number) => {
        if (e.ctrlKey && e.key == "Enter" && answerBlocks.filter((block) => block.id == id)[0].value != "") {
            addAnswerBlock();
        }
    }
    const handleKeyDownOnQuestion = (e: any) => {
        // setFocusOnQuestion(false);

        if (e.key == "Enter" && e.ctrlKey) {
            if (answerBlocks.length == 0) {
                addAnswerBlock();
            } else {
                console.log(answerBlocks)
                console.log(answerBlocks[answerBlocks.length - 1].id);
                setCurrent(answerBlocks[answerBlocks.length - 1].id);
            }
            // setFocusOnQuestion(false);
        }
    }
    const updateAnswers = (ans: string, id: number) => {
        const updatedAnswers = answerBlocks.map((block) => {
            if (block.id == id) {
                block.value = ans;
            }
            return block;
        });
        setAnswerBlocks(updatedAnswers);
    }
    const removeAnswerBlock = (id: number) => {
        const updatedAnswers = answerBlocks.filter((block) => block.id != id);
        setAnswerBlocks(updatedAnswers);
    }
    const renderAnswerBlocks = () => {
        return answerBlocks.map((block) => {
            return <div key={"answer-block" + block.id} className="w-full h-full flex flex-row mt-3">
                <input autoFocus={block.id == current} value={block.value} type="text" className="px-2 rounded-lg w-4/5 h-10 border-1 border-black resize-none fixed-textarea" onChange={(e) => updateAnswers(e.target.value, block.id)} onKeyDown={(e) => handleKeyDown(e, block.id)} />
                <button className="ml-3 px-2 cursor-pointer border-1 border-black fixed-button bg-red-500 hover:bg-red-700 text-white font-bold rounded-lg" onClick={() => removeAnswerBlock(block.id)}>Remove</button>
            </div>
        });
    }
    const handleSubmit = async () => {
        if (question == "") {
            toast.error("Please enter a question");
            return;
        } else if (answerBlocks.length == 0) {
            toast.error("Please add at least one answer");
            return;
        } else if (answerBlocks.filter((block) => block.value == "").length > 0) {
            toast.error("Please fill in all the answers");
            return;
        }
        // const res = await axios.get('/api/users?page=2');
        //call API
        // res = 
    }
    return (<>
        <div className="each-question-container w-full h-full">
            <div className="w-4/5 h-1/2 flex flex-col">
                <textarea
                    autoFocus={focusOnQuestion}
                    onKeyDown={(e) => handleKeyDownOnQuestion(e)}
                    onChange={(e) => updateQuestion(e.target.value)}
                    value={question}
                    className="px-2 py-1 rounded-lg w-full h-full border-1 border-black resize-none fixed-textarea"
                ></textarea>
            </div>
            <div className="mt-3 h-3/5 overflow-y-scroll w-full">
                <div>
                    {renderAnswerBlocks()}
                </div>
                <div className="flex justify-between w-4/5">
                    <button className="mt-3 cursor-pointer border-1 border-black fixed-button bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-lg"
                        onClick={() => addAnswerBlock()}>Add +</button>
                    <button className="mt-3 cursor-pointer border-1 border-black fixed-button bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg"
                        onClick={handleSubmit}
                    >Submit</button>
                </div>
            </div>

        </div>

    </>);
}

export default EachQuestion;