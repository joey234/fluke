import os
import json
import torch
import argparse
import shutil
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

########################################
# Dataset Classes
########################################

class PronounResolutionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        """
        Args:
            data (list): List of examples from the test data (original).
            tokenizer (T5Tokenizer): Tokenizer for T5.
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
        # The label is available, but for prediction we just need the prompt.
        # label = example["label"]

        input_text = (
            f"resolve: In the sentence '{text}', does the pronoun '{pronoun}' refer to the "
            f"'first' candidate ('{candidates[0]}') or the 'second' candidate ('{candidates[1]}')?"
        )

        inputs = self.tokenizer(
            input_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "index": idx
        }

class PronounResolutionOODDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        """
        Args:
            data (list): OOD test examples.
            tokenizer (T5Tokenizer): Tokenizer for T5.
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

        input_text = (
            f"resolve: In the sentence '{text}', does the pronoun '{pronoun}' refer to the "
            f"'first' candidate ('{candidates[0]}') or the 'second' candidate ('{candidates[1]}')?"
        )

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "index": example["index"]
        }

########################################
# Data Loading Functions
########################################

def load_data_from_json(file_path):
    """
    Load and prepare data from a JSON file for the original test set.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

########################################
# Evaluation Functions
########################################

def evaluate_and_save_predictions(model, test_file_path, tokenizer, max_length, batch_size, device, output_file):
    """
    Evaluates the model on the original test set and saves predictions to a JSON file.
    """
    test_data = load_data_from_json(test_file_path)
    test_dataset = PronounResolutionDataset(test_data, tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model.eval()
    predictions = {}
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating Original Test")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            indices = batch["index"]

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            decoded_preds = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            # Assign predictions by index
            for index, pred in zip(indices, decoded_preds):
                predictions[str(index.item())] = pred


    # Save predictions
    dirname = os.path.dirname(output_file)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions on original test saved to {output_file}")


def evaluate_and_save_predictions_ood(model, test_file_path, tokenizer, max_length, batch_size, device, output_file):
    """
    Evaluates the model on an OOD test file and saves predictions.
    """
    test_data = load_data_from_json(test_file_path)
    test_dataset = PronounResolutionOODDataset(test_data, tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model.eval()
    predictions = {}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating OOD {os.path.basename(test_file_path)}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            indices = batch["index"]

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            decoded_preds = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            for idx_val, pred in zip(indices, decoded_preds):
                predictions[str(idx_val.item())] = pred

    # Save predictions
    dirname = os.path.dirname(output_file)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions on OOD file {os.path.basename(test_file_path)} saved to {output_file}")


def process_ood_test_files(model, tokenizer, test_dir, output_dir, device, batch_size=16, max_length=128):
    """
    Processes all OOD test files in a directory, performs inference, and saves predictions.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for test_file in os.listdir(test_dir):
        if test_file.endswith("_100.json"):
            test_file_path = os.path.join(test_dir, test_file)
            output_file_path = os.path.join(output_dir, f"{test_file}")
            evaluate_and_save_predictions_ood(
                model, test_file_path, tokenizer, max_length, batch_size, device, output_file_path
            )

########################################
# Main Execution
########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a T5 model on original and OOD test sets.")

    # Add arguments
    parser.add_argument("--checkpoint_path", type=str, default="./tmp/t5-base/checkpoint_step_22000.pt", help="Path to the saved T5 checkpoint directory.")
    parser.add_argument("--test_file_path", type=str, default="../datasets/transformed_train_dev_test_data/thinh/test.json", help="Path to the original test JSON file.")
    parser.add_argument("--ood_test_dir", type=str, default="../datasets/test_data_after_modifications/thinh", help="Directory containing OOD test JSON files.")
    parser.add_argument("--pretrained_model_name", type=str, default="t5-base", help="Pretrained T5 model name or path.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for evaluation.")
    parser.add_argument("--output_file_original", type=str, default="./tmp/t5-base_results/t5_predictions.json", help="Output file for original test predictions.")
    parser.add_argument("--output_ood_dir", type=str, default="./tmp/t5-base_results/t5_ood_test_preds", help="Directory to save OOD test predictions.")

    args = parser.parse_args()

    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint_path).to(args.device)

    # Evaluate on the original test set
    evaluate_and_save_predictions(
        model=model,
        test_file_path=args.test_file_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        output_file=args.output_file_original
    )

    # Evaluate on all OOD test files in ood_test_dir
    process_ood_test_files(
        model=model,
        tokenizer=tokenizer,
        test_dir=args.ood_test_dir,
        output_dir=args.output_ood_dir,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        max_length=args.max_length
    )
