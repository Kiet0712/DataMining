from transformers import pipeline

# Check if a GPU is available
import torch
if torch.cuda.is_available():
    device = 0  # Index of the GPU to use (usually 0 for the first GPU)
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
else:
    device = -1 # Use CPU
    print("No GPU available, using CPU instead.")

pipe = pipeline("text-classification", model="mrsinghania/asr-question-detection", device=device)


texts_to_classify = [
"Q1: what am I?",
"A. Dog",
"B. Cat",
"C. Roster",
"D. Catilyn",
"Q2. who am I?"
"A. Nobody"
"B.Cat"
"C. Dog"
"D.Lmao"
]

results = pipe(texts_to_classify)
print(results)