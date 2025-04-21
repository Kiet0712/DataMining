import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import re
import string
import os
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from functools import reduce
import numpy as np

# Define the model and tokenizer
model_name = "microsoft/codebert-base"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    questions = [item['question'] for item in batch]
    options_list = [item['options'] for item in batch]
    return {'question': questions, 'options': options_list}

def get_predicted_letter(predictions, num_options, letter_labels):
    """
    Get the predicted letter label for a single sample.

    Args:
        predictions: A list of tensors, where each tensor represents the
                     probabilities for the options of a single sample.
        num_options: The number of options for the current sample.
        letter_labels: the list of possible letter labels

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

def main(model_path, test_csv_path, output_csv_path):
    """
    Main function to load the model, process the test data,
    make predictions, and save the results to a CSV file.

    Args:
        model_path (str): Path to the saved model file (.pth).
        test_csv_path (str): Path to the test CSV file.
        output_csv_path (str): Path to save the predictions CSV file.
    """
    # Load the model
    model = MCQA(model_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load the test data
    test_df = pd.read_csv(test_csv_path)
    test_df = test_df[test_df['choices'] != '[]']  # Filter out rows with empty choices
    formatted_test_data = test_df.apply(format_for_prediction, axis=1).tolist()

    # Create a DataLoader for the test data
    test_dataset = TestDataset(formatted_test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)

    # Make predictions
    predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            inputs = {'question': batch['question'], 'options': batch['options']}
            outputs = model(inputs)
            predictions.extend(outputs)

    # Map predictions to letter labels
    letter_labels = list(string.ascii_uppercase)
    predicted_letters = [get_predicted_letter(pred, len(item['options']), letter_labels) for pred, item in zip(predictions, formatted_test_data)]

    # Create a DataFrame for the results
    results_df = pd.DataFrame({'task_id': [item['id'] for item in formatted_test_data], 'answer': predicted_letters})

    # Save the results to a CSV file
    #os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

if __name__ == "__main__":
    # Example usage:
    model_path = "checkpoints/best_mcqa_model.pth"  # Replace with your actual model path
    test_csv_path = "dataset/b6_test_data.csv"  # Replace with your test CSV path
    output_csv_path = "lmao.csv" # Replace with your desired output path
    main(model_path, test_csv_path, output_csv_path)
