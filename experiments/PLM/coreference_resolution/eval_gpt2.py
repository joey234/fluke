import os
import json
import torch
import argparse
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel

########################################
# Dataset Classes
########################################

class PronounResolutionGPT2Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        """
        Args:
            data (list): List of examples from the test data.
                         Each example has:
                         - "original_sentence"
                         - "pronoun"
                         - "candidates"
                         - "label"
            tokenizer (GPT2Tokenizer): Tokenizer for GPT2.
            max_length (int): Maximum sequence length.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        text = example["original_sentence"]
        pronoun = example["pronoun"]
        candidates = example["candidates"]
        label = example["label"]

        input_text = (
            f"resolve: In the sentence '{text}', does the pronoun '{pronoun}' refer to the "
            f"'first' candidate ('{candidates[0]}') or the 'second' candidate ('{candidates[1]}')?"
        )
        target_text = "first" if label == 0 else "second"

        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        target_ids = self.tokenizer.encode(" " + target_text, add_special_tokens=False)
        combined_ids = input_ids + target_ids

        # Truncate if needed
        combined_ids = combined_ids[:self.max_length]

        # Labels: input prompt = -100, target tokens = actual tokens
        labels = [-100]*len(input_ids) + target_ids
        labels = labels[:self.max_length]

        attention_mask = [1]*len(combined_ids)

        # Left-padding if needed
        pad_length = self.max_length - len(combined_ids)
        if pad_length > 0:
            combined_ids = [self.tokenizer.pad_token_id]*pad_length + combined_ids
            attention_mask = [0]*pad_length + attention_mask
            labels = [-100]*pad_length + labels

        return {
            "input_ids": torch.tensor(combined_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "index": idx
        }

class PronounResolutionGPT2OODDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        """
        Args:
            data (list): OOD test examples.
                         Each example has:
                         - "modified_text"
                         - "modified_pronoun"
                         - "modified_candidates"
                         - "modified_label"
                         - "index"
            tokenizer (GPT2Tokenizer): Tokenizer for GPT2.
            max_length (int): Maximum sequence length.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        text = example["modified_text"]
        pronoun = example["modified_pronoun"]
        candidates = example["modified_candidates"]
        label = example["modified_label"]
        instance_index = example["index"]

        input_text = (
            f"resolve: In the sentence '{text}', does the pronoun '{pronoun}' refer to the "
            f"'first' candidate ('{candidates[0]}') or the 'second' candidate ('{candidates[1]}')?"
        )
        target_text = "first" if label == 0 else "second"

        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        target_ids = self.tokenizer.encode(" " + target_text, add_special_tokens=False)
        combined_ids = input_ids + target_ids
        combined_ids = combined_ids[:self.max_length]

        labels = [-100]*len(input_ids) + target_ids
        labels = labels[:self.max_length]

        attention_mask = [1]*len(combined_ids)

        pad_length = self.max_length - len(combined_ids)
        if pad_length > 0:
            combined_ids = [self.tokenizer.pad_token_id]*pad_length + combined_ids
            attention_mask = [0]*pad_length + attention_mask
            labels = [-100]*pad_length + labels

        return {
            "input_ids": torch.tensor(combined_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "index": instance_index
        }

########################################
# Data Loading
########################################

def load_data_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

########################################
# Evaluation Function
########################################

def evaluate_and_save_predictions(model, data_loader, tokenizer, device):
    """
    Evaluate the model on a dataset and save predictions in a dictionary:
    key = instance index, value = predicted label ("first" or "second").
    """
    model.eval()
    predictions = {}

    with torch.no_grad():
        batch_idx = 0
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            indices = batch["index"]  # The index of each instance

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                example_label_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
                if len(example_label_positions) == 0:
                    # No label tokens? Skip
                    continue
                start_label = example_label_positions[0]
                end_label = example_label_positions[-1] + 1
                example_pred_tokens = torch.argmax(logits[i, start_label:end_label, :], dim=-1)

                pred_str = tokenizer.decode(example_pred_tokens).strip()
                # Store prediction: key = instance index, value = pred_str
                instance_idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
                predictions[str(instance_idx)] = pred_str
            batch_idx += 1

    return predictions

########################################
# OOD Evaluation
########################################

def evaluate_ood_and_save_predictions(model, tokenizer, device, max_length, batch_size, test_dir, output_dir):
    """
    Evaluate the model on all OOD test files and save predictions as JSON files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for test_file in os.listdir(test_dir):
        if test_file.endswith("_100.json"):
            test_file_path = os.path.join(test_dir, test_file)
            data = load_data_from_json(test_file_path)
            dataset = PronounResolutionGPT2OODDataset(data, tokenizer, max_length)
            data_loader = DataLoader(dataset, batch_size=batch_size)

            predictions = evaluate_and_save_predictions(model, data_loader, tokenizer, device)
            output_file_path = os.path.join(output_dir, test_file)
            dirname = os.path.dirname(output_file_path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)

            with open(output_file_path, "w") as f:
                json.dump(predictions, f, indent=4)
            print(f"Predictions on OOD file {test_file} saved to {output_file_path}")

########################################
# Main Execution
########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a GPT model on original and OOD test sets and save predictions.")
    parser.add_argument("--checkpoint_path", type=str, default="./tmp/gpt2/checkpoint_step_30000.pt", help="Path to the GPT model checkpoint directory.")
    parser.add_argument("--test_file_path", type=str, default="../datasets/transformed_train_dev_test_data/thinh/test.json", help="Path to the original test JSON file.")
    parser.add_argument("--ood_test_dir", type=str, default="../datasets/test_data_after_modifications/thinh", help="Directory containing OOD test JSON files.")
    parser.add_argument("--pretrained_model_name", type=str, default="gpt2", help="Pretrained GPT model name or path.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for evaluation.")
    parser.add_argument("--output_file_original", type=str, default="./tmp/gpt2_results/gpt2_predictions.json", help="Output file for original test predictions.")
    parser.add_argument("--output_ood_dir", type=str, default="./tmp/gpt2_results/gpt2_ood_test_preds", help="Directory to save OOD test predictions.")

    args = parser.parse_args()

    # Initialize tokenizer with left-padding for GPT2
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model_name, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(args.checkpoint_path).to(args.device)

    # Evaluate on original test set
    test_data = load_data_from_json(args.test_file_path)
    test_dataset = PronounResolutionGPT2Dataset(test_data, tokenizer, args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    original_predictions = evaluate_and_save_predictions(model, test_loader, tokenizer, torch.device(args.device))
    # Save original test predictions
    dirname = os.path.dirname(args.output_file_original)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(args.output_file_original, "w") as f:
        json.dump(original_predictions, f, indent=4)
    print(f"Predictions on original test saved to {args.output_file_original}")

    # Evaluate on OOD test sets
    evaluate_ood_and_save_predictions(
        model=model,
        tokenizer=tokenizer,
        device=torch.device(args.device),
        max_length=args.max_length,
        batch_size=args.batch_size,
        test_dir=args.ood_test_dir,
        output_dir=args.output_ood_dir
    )
