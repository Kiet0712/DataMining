from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
import pandas as pd
import evaluate
import string
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from functools import reduce
from tqdm import tqdm

model_name = "microsoft/codebert-base"

class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        return x

class MCQA(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder_name = encoder_name
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder.config.output_hidden_states=True

        self.cls = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.encoder.config.hidden_size, 1)
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.encoder.config.hidden_size,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )
        self.ffn = FFN(self.encoder.config.hidden_size, self.encoder.config.hidden_size*4, self.encoder.config.hidden_size, 0.1)

    def forward(self, inputs):
        options_list = inputs['options']
        question = inputs['question']
        batch_size = len(question)

        # Tokenize questions and options
        question_with_options = [reduce(lambda acc, ele: self.tokenizer.cls_token + acc, options, q) for q, options in zip(question, options_list)]
        question_tokenized = self.tokenizer(question_with_options, return_tensors='pt', padding='max_length', truncation=True, add_special_tokens=False)
        question_input_ids = question_tokenized['input_ids'].to(self.encoder.device)
        question_attention_mask = question_tokenized['attention_mask'].to(self.encoder.device)

        # Tokenize concatenated options
        options_concat = [reduce(lambda acc, ele: acc + self.tokenizer.sep_token +  ele, options, '') for options in options_list]
        options_tokenized = self.tokenizer(options_concat, return_tensors='pt', padding='max_length', truncation=True, add_special_tokens=False)
        options_input_ids = options_tokenized['input_ids'].to(self.encoder.device)
        options_attention_mask = options_tokenized['attention_mask'].to(self.encoder.device)

        # Encode questions and options
        question_encoder_outputs = self.encoder(question_input_ids, attention_mask=question_attention_mask)
        question_hidden_states = question_encoder_outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)

        options_encoder_outputs = self.encoder(options_input_ids, attention_mask=options_attention_mask)
        options_hidden_states = options_encoder_outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)

        # Cross-attention
        cross_attn_output, _ = self.cross_attn(
            question_hidden_states,
            options_hidden_states,
            options_hidden_states,
            key_padding_mask=~options_attention_mask.bool()  # Invert mask
        )  # (batch_size, seq_len, hidden_size)

        results = self.ffn(cross_attn_output) # (batch_size, seq_len, hidden_size)

        # Extract the CLS token representations for each sample in the batch
        choose_list = []
        cls_token_id = self.tokenizer.cls_token_id
        for i in range(batch_size):
            # Find CLS token indices for the i-th example
            cls_indices = (question_input_ids[i] == cls_token_id).nonzero(as_tuple=True)[0]
            choose_sample = results[i, cls_indices, :]
            choose_list.append(choose_sample)

        # Process each item in the list
        choose_list = [self.cls(item).squeeze(-1) for item in choose_list]
        choose_list = [torch.nn.functional.softmax(item, dim=-1) for item in choose_list]

        return choose_list

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MCQA(model_name).to(device)

file_path = r"C:\Users\dangv\Desktop\DM\dataset\b6_train_data.csv"
test_file_path = r"C:\Users\dangv\Desktop\DM\dataset\b6_test_data.csv"
df = pd.read_csv(file_path)
df = df[df['choices'] != '[]']
test_df = pd.read_csv(test_file_path)
test_df = test_df[test_df['choices'] != '[]']

def format_data(row):
    task_id = row['task_id']
    question = row['question']
    options_str = row['choices']
    correct_answer_letter = str(row['answer']).upper()
    options = re.findall(r"'(.*?)'", options_str)
    try:
        label_map = {letter: i for i, letter in enumerate(string.ascii_uppercase)} # Use all uppercase ASCII letters
        label = label_map.get(correct_answer_letter[-1])
        if label is None or label >= len(options):
            return None
    except KeyError:
        return None
    assert len(options) != 0, (task_id, options)
    return {
        'id': task_id,
        'question': question,
        'answer': label,
        'num_options': len(options),
        'options': options
    }

def format_for_prediction(row):
    task_id = row['task_id']
    question = row['question']
    options_str = row['choices']
    options = re.findall(r"'(.*?)'", options_str)
    assert len(options) != 0, (task_id, options)
    return {
        'id': task_id,
        'question': question,
        'options': options
    }

formatted_data = [item for item in df.apply(format_data, axis=1).tolist() if item is not None]
formatted_test_data = test_df.apply(format_for_prediction, axis=1).tolist()
train_data, val_data = train_test_split(formatted_data, test_size=0.1, random_state=261207)
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
accuracy_metric = evaluate.load("accuracy")

def collate_fn(batch):
    questions = [item['question'] for item in batch]
    options_list = [item['options'] for item in batch]
    if 'answer' in batch[0]:  # Only training/validation data has 'answer'
        answers = [item['answer'] for item in batch]
        return {'question': questions, 'options': options_list, 'labels': answers}
    else:
        return {'question': questions, 'options': options_list}

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn)

optimizer = optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()
num_epochs = 10
output_dir = "./outputs"
best_accuracy = 0.0
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    # Wrap the train_dataloader with tqdm
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
        optimizer.zero_grad()
        inputs = {'question': batch['question'], 'options': batch['options']}
        outputs = model(inputs)
        labels = [torch.tensor(label,dtype=torch.long).to(device) for label in batch['labels']]
        # Calculate loss for each sample in the batch
        loss_list = [criterion(output, label) for output, label in zip(outputs, labels)]
        loss = sum(loss_list) / len(loss_list) #take the mean
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

    model.eval()
    total_eval_loss = 0
    all_preds = []
    all_labels = []
    # Wrap the val_dataloader with tqdm
    for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
        with torch.no_grad():
            inputs = {'question': batch['question'], 'options': batch['options']}
            outputs = model(inputs)
            labels = [torch.tensor(label,dtype=torch.long).to(device) for label in batch['labels']]
            # Calculate loss for each sample in the batch
            loss_list = [criterion(output, label) for output, label in zip(outputs, labels)]
            loss = sum(loss_list) / len(loss_list)
            total_eval_loss += loss.item()
            predictions = [torch.argmax(output, dim=-1).cpu().numpy() for output in outputs]
            all_preds.extend(predictions)
            all_labels.extend([label.cpu().numpy() for label in labels])

    avg_eval_loss = total_eval_loss / len(val_dataloader)
    accuracy = accuracy_metric.compute(predictions=all_preds, references=all_labels)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_eval_loss:.4f}, Validation Accuracy: {accuracy['accuracy']:.4f}")

    # Save the model if it has the best validation accuracy so far
    if accuracy['accuracy'] > best_accuracy:
        best_accuracy = accuracy['accuracy']
        best_model_state = model.state_dict()
        print(f"Validation accuracy improved. Saving model with accuracy: {best_accuracy:.4f}")

print("Training finished.")
# Save the best model
if best_model_state is not None:
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_mcqa_model.pth")
    torch.save(best_model_state, best_model_path)
    print(f"Best model saved to {best_model_path}")
else:
    print("No model with improved validation accuracy was saved.")

# Prediction on test data using the best model
def predict(model, test_data, batch_size=4):
    model.eval()
    predictions = []
    test_dataset = Dataset.from_list(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = {'question': batch['question'], 'options': batch['options']}
            outputs = model(inputs)
            predictions.extend(outputs)
    return predictions

# Load the best model for prediction
if best_model_state is not None:
    best_model = MCQA(model_name).to(device)
    best_model.load_state_dict(best_model_state)
    print("Loaded the best model for prediction.")
    model_to_predict = best_model
else:
    print("Using the last trained model for prediction.")
    model_to_predict = model

# Make predictions on the formatted test data
test_predictions = predict(model_to_predict, formatted_test_data)

# Map the numerical predictions to letter labels
letter_labels = list(string.ascii_uppercase)  # Use all uppercase ASCII letters

def get_predicted_letter(predictions, num_options):
    """
    Get the predicted letter label for a single sample.

    Args:
        predictions: A list of tensors, where each tensor represents the
                     probabilities for the options of a single sample.
        num_options: The number of options for the current sample.

    Returns:
        str: The predicted letter label (e.g., 'A', 'B', 'C', etc.).
             Returns 'A' if num_options is 0.
    """
    if num_options == 0:
        return 'A'  # Or any default value you prefer

    # Get the probabilities for the current sample
    sample_predictions = predictions.cpu().numpy()
    # Get the index of the maximum probability
    predicted_index = np.argmax(sample_predictions)
    # Return the letter corresponding to the predicted index
    return letter_labels[predicted_index]

# Get the predicted letters for the test data
predicted_letters = [get_predicted_letter(pred, len(item['options'])) for pred, item in zip(test_predictions, formatted_test_data)]

# Create a DataFrame for the results
results_df = pd.DataFrame({'task_id': [item['id'] for item in formatted_test_data], 'answer': predicted_letters})

# Write the results to a CSV file
results_csv_path = "./outputs/predictions.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Predictions saved to {results_csv_path}")