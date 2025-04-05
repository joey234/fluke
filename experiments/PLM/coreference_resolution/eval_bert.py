import os
import re
import json
import torch
import argparse
from torch import nn
from torch.nn import init
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset


# Dataset Class
class SentencePairDataset(Dataset):
    def __init__(self, sentence_pairs, labels, indices):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        s1, s2 = self.sentence_pairs[idx]
        label = self.labels[idx]
        index = self.indices[idx]
        return s1, s2, label, index


def collate_fn(batch, tokenizer, max_length):
    """
    Custom collate function for dynamic padding.
    """
    s1_list = [item[0] for item in batch]
    s2_list = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    instance_indices = [item[3] for item in batch]

    # Tokenize and dynamically pad the input sequences
    encoded_s1 = tokenizer(s1_list, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    encoded_s2 = tokenizer(s2_list, truncation=True, padding=True, max_length=max_length, return_tensors="pt")

    return encoded_s1, encoded_s2, labels, instance_indices


def load_data_from_json(file_path):
    """
    Load and prepare data from a JSON file. This function is used to read the test file
    in "../datasets/transformed_train_dev_test_data/test.json"
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        sentence_pairs (list): List of (sentence1, sentence2) pairs.
        labels (list): List of labels corresponding to the sentence pairs.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    instance_indices = list(range(len(data)))
    sentence_pairs = [(item["sentence1"], item["sentence2"]) for item in data]
    labels = [item["label"] for item in data]
    return sentence_pairs, labels, instance_indices


def load_ood_data_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    sentence_pairs = []
    labels = [0] * len(data)
    instance_indices = []
    for x in data:
        sentence1 = safe_replace(x["modified_text"], x["modified_pronoun"], x["modified_candidates"][0])
        sentence2 = safe_replace(x["modified_text"], x["modified_pronoun"], x["modified_candidates"][1])
        sentence_pairs.append((sentence1, sentence2))
        labels.append(x["modified_label"])
        instance_indices.append(x["index"])
    return sentence_pairs, labels, instance_indices


# Model Class
class BertBinaryClassifier(nn.Module):
    def __init__(self, pretrained_model_name, embedding_dim=768):
        super(BertBinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.v = nn.Parameter(torch.randn(embedding_dim, 1))  # Learnable parameter vector
        self._init_weights()

    def _init_weights(self):
        # Use Xavier initialization (Glorot uniform)
        init.xavier_uniform_(self.v)

    def forward(self, input_ids_s1, attention_mask_s1, input_ids_s2, attention_mask_s2):
        cls_s1 = self.bert(input_ids=input_ids_s1, attention_mask=attention_mask_s1).last_hidden_state[:, 0, :]
        cls_s2 = self.bert(input_ids=input_ids_s2, attention_mask=attention_mask_s2).last_hidden_state[:, 0, :]
        score_s1 = torch.matmul(cls_s1, self.v).squeeze(-1)
        score_s2 = torch.matmul(cls_s2, self.v).squeeze(-1)
        logits = torch.stack([score_s1, score_s2], dim=-1)  # Shape: [batch_size, 2]
        return logits


def evaluate_and_save_predictions(model, test_file_path, tokenizer, max_length, batch_size, device, output_file, load_data_from_json):
    """
    Evaluates a saved checkpoint on the test set and saves predictions to a JSON file.

    Args:
        checkpoint_path (str): Path to the saved checkpoint.
        test_file_path (str): Path to the test JSON file.
        pretrained_model_name (str): Pretrained model name or path.
        max_length (int): Maximum sequence length.
        batch_size (int): Evaluation batch size.
        device (torch.device): The device to run the evaluation on.
        output_file (str): Path to save the predictions JSON file.
    """
    # Load test data
    test_sentence_pairs, test_labels, test_indices = load_data_from_json(test_file_path)

    # Initialize tokenizer and dataset
    test_dataset = SentencePairDataset(test_sentence_pairs, test_labels, test_indices)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length)
    )

    # Store predictions in a dictionary
    predictions = {}
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_loader)):
            encoded_s1, encoded_s2, _, indices = batch  # Ignore labels since we are predicting
            input_ids_s1 = encoded_s1["input_ids"].squeeze(1).to(device)
            attention_mask_s1 = encoded_s1["attention_mask"].squeeze(1).to(device)
            input_ids_s2 = encoded_s2["input_ids"].squeeze(1).to(device)
            attention_mask_s2 = encoded_s2["attention_mask"].squeeze(1).to(device)

            # Model inference
            logits = model(input_ids_s1, attention_mask_s1, input_ids_s2, attention_mask_s2)
            batch_predictions = torch.argmax(logits, dim=1).cpu().tolist()

            # Save predictions with instance indices
            for index, prediction in zip(indices, batch_predictions):
                predictions[str(index)] = prediction

    # Save predictions to a JSON file
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved to {output_file}")


# Replace the pronoun in the text with the candidate
def safe_replace(text, pronoun, candidate):
    """
    Replace the pronoun in the text with the candidate:
    - Capitalize the candidate if the replacement is at the start of the sentence.
    - Use lowercase for replacements in the middle of the sentence.
    """
    # Use regex to match the pronoun as a whole word
    # Check if the pronoun appears at the start of the text
    match = re.match(rf'\b{re.escape(pronoun)}\b', text, re.IGNORECASE)
    if match:
        # Replacement at the start of the sentence; capitalize the candidate
        replaced_text = re.sub(rf'\b{re.escape(pronoun)}\b', candidate, text, count=1)
    else:
        # Replacement in the middle of the sentence; use lowercase
        replaced_text = re.sub(rf'\b{re.escape(pronoun)}\b', candidate, text, count=1)

    return replaced_text


def process_ood_test_files(model, tokenizer, test_dir, output_dir, device, batch_size=16, max_length=128):
    """
    Processes all OOD test files in a directory, performs inference, and saves predictions for each file.

    Args:
        model (BertBinaryClassifier): Pretrained BERT model.
        tokenizer (BertTokenizer): Tokenizer corresponding to the BERT model.
        test_dir (str): Directory containing OOD test files in JSON format.
        output_dir (str): Directory to save the output predictions in JSON format.
        batch_size (int): Number of examples to process in each batch.
        max_length (int): Maximum sequence length for tokenization.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for test_file in os.listdir(test_dir):
        if test_file.endswith("_100.json"):
            test_file_path = os.path.join(test_dir, test_file)
            output_file_path = os.path.join(output_dir, test_file)
            evaluate_and_save_predictions(
                model, test_file_path, tokenizer, max_length,
                batch_size, device, output_file_path, load_ood_data_from_json
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a BERT checkpoint on a test set.")

    # Add arguments
    parser.add_argument("--checkpoint_path", type=str,
                        default="./tmp/bert-base-cased/checkpoint_step_30000.pt",
                        help="Path to the saved checkpoint.")
    parser.add_argument("--test_file_path", type=str,
                        default="../datasets/transformed_train_dev_test_data/thinh/test.json",
                        help="Path to the test JSON file.")
    parser.add_argument("--pretrained_model_name", type=str, default="bert-base-cased", help="Pretrained model name or path.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for evaluation.")
    parser.add_argument("--output_file", type=str, default="./tmp/bert-base-cased_results/bert_predictions.json")

    args = parser.parse_args()

    # Load the model and checkpoint
    model = BertBinaryClassifier(args.pretrained_model_name).to(args.device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(args.checkpoint_path))
    else:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name)

    # Run evaluation on the original test set
    evaluate_and_save_predictions(
        model=model,
        test_file_path=args.test_file_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        output_file=args.output_file,
        load_data_from_json=load_data_from_json
    )

    # Run evaluation on ood test set
    # test_dir = "../datasets/test_data_after_modifications/thinh"
    # output_dir = "./tmp/bert-base-cased_results/bert_ood_test_preds"
    # process_ood_test_files(model, tokenizer, test_dir, output_dir, args.device, batch_size=16, max_length=128)
