function QuestionBlock({
    question,
    answerBlocks,
}: {
    question: string;
    answerBlocks: { id: number; value: string; acc: number; correct: boolean }[];
}) {
    const renderAnswerBlocks = () => {
        return answerBlocks.map((block) => {
            return (
                <div
                    key={"answer-block" + block.id}
                    className="w-full flex flex-col justify-center items-center mt-3"
                >
                    {/* Container with fixed width */}
                    <div className="relative w-full h-10 flex flex-row justify-center items-center">
                        {/* Green background proportional to acc */}
                        {block.acc > 0 && (
                            <div
                                className="absolute z-10 top-0 left-0 h-full bg-green-300 opacity-30 rounded-lg pointer-events-none"
                                style={{ width: `${block.acc * 100}%` }}
                            ></div>
                        )}

                        <input
                            value={block.value}
                            type="text"
                            className={`relative z-1 px-2 w-full h-full border border-black shadow-md rounded-lg resize-none ${block.correct ? "bg-green-500" : "bg-gray-400"
                                }`}
                        />
                        <text className="w-1/5 right-0 absolute z-1 text-center text-white font-bold">{+block.acc.toFixed(2) * 100}%</text>
                    </div>
                </div>
            );
        });
    };

    return (
        <div className="each-question-container w-full h-full flex flex-col bg-gray-300 justify-center items-center shadow-lg border border-gray-600 rounded-lg p-4">
            <div className="w-4/5 h-1/2 flex flex-col">
                <textarea
                    onKeyDown={() => { }}
                    onChange={() => { }}
                    value={question}
                    className="px-2 py-1 rounded-lg w-full h-full border border-black shadow-md resize-none"
                ></textarea>
            </div>
            <div className="mt-3 w-4/5">
                <div className="flex flex-col justify-center items-center">
                    {renderAnswerBlocks()}
                </div>
            </div>
        </div>
    );
}

export default QuestionBlock;
