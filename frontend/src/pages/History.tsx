import { useState } from "react";
import QuestionBlock from "../components/questionBlock";

function HisToryPage() {
    let data: any = localStorage.getItem("data");
    data = JSON.parse(data);

    const [questions, setQuestions] = useState<{ question: string, answerBlocks: { id: number, value: string, acc: number, correct: boolean }[] }[]>([
        { question: "Question 1", answerBlocks: [{ id: 1, value: "Answer 1", acc: 0.33, correct: false }, { id: 2, value: "Answer 2", acc: 0.33, correct: false }, { id: 3, value: "Answer 3", acc: 0.33, correct: false }] },
        { question: "Question 2", answerBlocks: [{ id: 1, value: "Answer 1", acc: 0.33, correct: false }, { id: 2, value: "Answer 2", acc: 0.33, correct: false }, { id: 3, value: "Answer 3", acc: 0.33, correct: false }] },
        { question: "Question 3", answerBlocks: [{ id: 1, value: "Answer 1", acc: 0.33, correct: false }, { id: 2, value: "Answer 2", acc: 0.33, correct: false }, { id: 3, value: "Answer 3", acc: 0.33, correct: false }] },
        { question: "Question 4", answerBlocks: [{ id: 1, value: "Answer 1", acc: 0.33, correct: false }, { id: 2, value: "Answer 2", acc: 0.33, correct: false }, { id: 3, value: "Answer 3", acc: 0.33, correct: false }] },
        { question: "Question 5", answerBlocks: [{ id: 1, value: "Answer 1", acc: 0.33, correct: false }, { id: 2, value: "Answer 2", acc: 0.33, correct: false }, { id: 3, value: "Answer 3", acc: 0.33, correct: false }] },
        { question: "Question 6", answerBlocks: [{ id: 1, value: "Answer 1", acc: 0.33, correct: false }, { id: 2, value: "Answer 2", acc: 0.33, correct: false }, { id: 3, value: "Answer 3", acc: 0.33, correct: false }] },
        { question: "Question 7", answerBlocks: [{ id: 1, value: "Answer 1", acc: 0.33, correct: false }, { id: 2, value: "Answer 2", acc: 0.33, correct: false }, { id: 3, value: "Answer 3", acc: 0.33, correct: false }] },
        { question: "Question 8", answerBlocks: [{ id: 1, value: "Answer 1", acc: 0.33, correct: false }, { id: 2, value: "Answer 2", acc: 0.33, correct: false }, { id: 3, value: "Answer 3", acc: 0.33, correct: false }] },
        { question: "Question 9", answerBlocks: [{ id: 1, value: "Answer 1", acc: 0.33, correct: false }, { id: 2, value: "Answer 2", acc: 0.33, correct: false }, { id: 3, value: "Answer 3", acc: 0.33, correct: false }] },
        { question: "Question 10", answerBlocks: [{ id: 1, value: "Answer 1", acc: 0.33, correct: false }, { id: 2, value: "Answer 2", acc: 0.33, correct: false }, { id: 3, value: "Answer 3", acc: 0.33, correct: false }] },
    ]);

    if (data.length == 0) {
        setQuestions(data)

    }

    const render = questions.map((question) => {

        return <div className="my-2 w-8/9 mx-auto flex flex-col justify-center items-center">
            <QuestionBlock question={question.question} answerBlocks={question.answerBlocks} />
        </div>

    });
    return (<>
        <div className="my-2 flex flex-col justify-center items-center">
            {render}
        </div>

    </>);
}

export default HisToryPage;