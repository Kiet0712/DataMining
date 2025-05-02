import { useState } from "react";
import axios from "../services/axiosServices";
function FileQuestion() {
    const [file, setFile] = useState<File | any | null>(null);
    const [error, setError] = useState("");
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [textContent, setTextContent] = useState<string>("");

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0];
        if (!selectedFile) return;

        const allowedTypes = [
            // "application/pdf",
            // "application/msword",
            // "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ];

        if (!allowedTypes.includes(selectedFile.type)) {
            setError("Only Text files are allowed.");
            setFile(null);
            setPreviewUrl(null);
            setTextContent("");
            return;
        }

        setError("");
        setFile(selectedFile);

        if (selectedFile.type === "text/plain") {
            const reader = new FileReader();
            reader.onload = (e) => setTextContent(e.target?.result as string);
            reader.readAsText(selectedFile);
            setPreviewUrl(null);
        } else if (selectedFile.type === "application/pdf") {
            const url = URL.createObjectURL(selectedFile);
            setPreviewUrl(url);
            setTextContent("");
        } else {
            // Word file – no preview
            setPreviewUrl(null);
            setTextContent("");
        }
    };
    const onSubmit = async () => {
        const formData = new FormData();
        formData.append("file", file);
        const res = await axios.post("/submit_file/", formData);
        console.log(res);
    };


    return (
        <div className="flex flex-col justify-center items-center w-full">
            <label className="font-medium">Upload Word, PDF, or TXT:</label>
            <input
                id="upload"
                type="file"
                accept=".txt"
                onChange={handleFileChange}
                className="hidden"
            />
            <div className="w-full h-30 flex justify-center items-center my-2">
                <label
                    htmlFor="upload"
                    className="font-medium w-4/5 h-full bg-blue-300 cursor-pointer flex justify-center items-center rounded-lg"
                >
                    Upload
                </label>
            </div>

            {error && <p className="text-red-600">{error}</p>}
            {file && <p className="text-green-600">Selected: {file.name}</p>}

            {((textContent || previewUrl) || file?.type.includes("word")) && (
                <div className="w-full max-w-6xl flex flex-col justify-center items-center">
                    <div className="flex flex-row w-4/5 justify-between items-center">
                        {!file?.type.includes("word") ?
                            <p className="font-semibold mb-2">Review file:</p>
                            : <p>Don't support Word file review</p>
                        }
                        <button className="my-3 w-30 cursor-pointer border-1 border-black fixed-button bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-lg"
                            onClick={() => onSubmit()}
                        >
                            Submit
                        </button>
                    </div>
                    {textContent || previewUrl ? (
                        <div className="w-4/5 h-[500px] border border-gray-500 rounded shadow-inner p-3 bg-white overflow-auto">
                            {textContent ? (
                                <pre className="whitespace-pre-wrap text-sm">{textContent}</pre>
                            ) : file?.type === "application/pdf" && previewUrl ? (
                                <iframe
                                    src={previewUrl}
                                    title="PDF Preview"
                                    className="w-full h-full border-none"
                                ></iframe>
                            ) : null}
                        </div>
                    ) : null
                    }
                </div>
            )}
        </div>
    );
}

export default FileQuestion;
