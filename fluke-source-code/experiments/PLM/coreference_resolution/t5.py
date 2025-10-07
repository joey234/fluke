import argparse
from argparse import ArgumentParser
import json
import os
import torch
import shutil
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from tqdm import tqdm
from datetime import datetime
import wandb


# Dataset Class
class PronounResolutionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        """
        Args:
            data (list): List of examples from the training data.
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
        label = example["label"]

        # Construct input and target strings
        input_text = (
            f"resolve: In the sentence '{text}', does the pronoun '{pronoun}' refer to the "
            f"'first' candidate ('{candidates[0]}') or the 'second' candidate ('{candidates[1]}')?"
        )
        target_text = "first" if label == 0 else "second"

        # Tokenize inputs and targets
        inputs = self.tokenizer(
            input_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        targets = self.tokenizer(
            target_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": targets["input_ids"].squeeze(0),
        }


def load_data_from_json(file_path):
    """
    Load and prepare data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        list: List of data examples.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def evaluate(model, data_loader, tokenizer, device):
    """
    Evaluate the model on a dataset.

    Args:
        model: Trained T5 model.
        data_loader: DataLoader for evaluation data.
        tokenizer: Tokenizer for decoding outputs.
        device: Torch device (CPU/GPU).

    Returns:
        float: Accuracy of the model on the dataset.
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.generate(input_ids=inputs, attention_mask=attention_mask)
            predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            targets = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

            total_correct += sum(p == t for p, t in zip(predictions, targets))
            total_samples += len(targets)

    return total_correct / total_samples


def train_model(model, train_loader, dev_loader, test_loader, tokenizer, optimizer, device, save_dir, log_interval, max_steps, epochs, use_wandb):
    """
    Train the T5 model.

    Args:
        model: T5 model to train.
        train_loader: DataLoader for training data.
        dev_loader: DataLoader for validation data.
        tokenizer: Tokenizer for decoding outputs.
        optimizer: Optimizer for training.
        device: Torch device (CPU/GPU).
        save_dir: Directory to save model checkpoints.
        log_interval: Interval for logging metrics.
        max_steps: Maximum training steps.
        use_wandb: Whether to use wandb for logging.
    """
    model.train()
    total_loss = 0
    global_step = 0
    checkpoint_paths = []

    with tqdm(total=max_steps, desc="Training") as pbar:
        for epoch in range(epochs):
            for batch in train_loader:
                if global_step >= max_steps:
                    break

                inputs = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                global_step += 1
                pbar.update(1)

                # Log metrics
                if global_step % log_interval == 0:
                    dev_accuracy = evaluate(model, dev_loader, tokenizer, device)
                    test_accuracy = evaluate(model, test_loader, tokenizer, device)
                    if use_wandb:
                        wandb.log({"Global Step": global_step, "Loss": total_loss / log_interval,
                                   "Dev Accuracy": dev_accuracy, "Test Accuracy": test_accuracy})
                    total_loss = 0

                    # Save model checkpoint
                    checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{global_step}.pt")
                    model.save_pretrained(checkpoint_path)
                    checkpoint_paths.append(checkpoint_path)

                    # Keep only the most recent 5 checkpoints
                    if len(checkpoint_paths) > 5:
                        oldest_checkpoint = checkpoint_paths.pop(0)
                        shutil.rmtree(oldest_checkpoint)
                        print(f"Deleted old checkpoint: {oldest_checkpoint}")


# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train a T5 model for pronoun resolution.")

    # Add arguments
    parser.add_argument("--pretrained_model_name", type=str, default="t5-base", help="Pretrained T5 model name or path.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval (steps).")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum number of training steps.")
    parser.add_argument("--use_wandb", type=bool, default=True, help="Whether to use wandb for logging.")
    parser.add_argument("--save_dir", type=str, default="./t5_checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data JSON file.")
    parser.add_argument("--dev_file", type=str, required=True, help="Path to the validation data JSON file.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the validation data JSON file.")

    return parser.parse_args()


# Main Execution
if __name__ == "__main__":
    args = parse_args()

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize wandb
    # Get the current date and time
    now = datetime.now()
    # Format the datetime string
    datetime_string = now.strftime("%Y-%m-%d-%H%M")
    if args.use_wandb:
        wandb.init(
            project="ood-plms-coref",
            name=f"t5-{datetime_string}",
            config=vars(args)
        )

    # Load data
    train_data = load_data_from_json(args.train_file)
    dev_data = load_data_from_json(args.dev_file)
    test_data = load_data_from_json(args.test_file)

    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Create data loaders
    train_dataset = PronounResolutionDataset(train_data, tokenizer, args.max_length)
    dev_dataset = PronounResolutionDataset(dev_data, tokenizer, args.max_length)
    test_dataset = PronounResolutionDataset(test_data, tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        tokenizer=tokenizer,
        optimizer=optimizer,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        max_steps=args.max_steps,
        epochs=args.epochs,
        use_wandb=args.use_wandb
    )
