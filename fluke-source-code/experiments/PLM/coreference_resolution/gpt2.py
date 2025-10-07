import argparse
import json
import os
import torch
import shutil
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from tqdm import tqdm
from datetime import datetime
import wandb

# Dataset Class for GPT2
class PronounResolutionGPT2Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        """
        Args:
            data (list): List of examples from the training data.
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

        # Construct input and target strings
        # Same prompt as T5, but now we will concatenate the target label at the end.
        input_text = (
            f"resolve: In the sentence '{text}', does the pronoun '{pronoun}' refer to the "
            f"'first' candidate ('{candidates[0]}') or the 'second' candidate ('{candidates[1]}')?"
        )
        target_text = "first" if label == 0 else "second"

        # Tokenize the input and target
        # We'll create a single sequence: [input_prompt tokens ... target_tokens]
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        target_ids = self.tokenizer.encode(" " + target_text, add_special_tokens=False)
        # We add a preceding space before the target to ensure correct tokenization.

        # Combine input and target
        combined_ids = input_ids + target_ids

        # Truncate if longer than max_length
        combined_ids = combined_ids[:self.max_length]

        # Create labels: For GPT-2, we want to predict the target tokens.
        # We'll set labels = combined_ids, but all input prompt tokens replaced with -100
        labels = [-100] * len(input_ids) + target_ids
        labels = labels[:self.max_length]

        # Create attention mask
        attention_mask = [1] * len(combined_ids)

        # Pad if necessary
        pad_length = self.max_length - len(combined_ids)
        if pad_length > 0:
            combined_ids += [self.tokenizer.eos_token_id] * pad_length
            attention_mask += [0] * pad_length
            labels += [-100] * pad_length

        return {
            "input_ids": torch.tensor(combined_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_data_from_json(file_path):
    """
    Load and prepare data from a JSON file.

    Returns a list of data examples.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def evaluate(model, data_loader, tokenizer, device):
    """
    Evaluate the model on a dataset.

    We will generate predictions by prompting the model and checking if the next tokens match "first" or "second".
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    # Since GPT-2 is a causal model, we can either compare directly by greedy generation or by decoding the target tokens.
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Generate predictions:
            # We'll just run the model and get the predicted tokens at the label positions.
            # Since we know the label is always at the end, we can decode predictions directly.
            # Another approach is to do greedy generation from the prompt, but here we already have full input including the label tokens.
            # We'll compare the model's most likely tokens on the target segment with the actual labels.

            # Get logits
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Find where labels != -100 to identify label positions
            label_positions = (labels != -100)
            # For simplicity, decode predicted tokens at label positions and compare to targets
            pred_tokens = torch.argmax(logits[label_positions], dim=-1)
            target_tokens = labels[label_positions]

            # Decode predictions and targets
            # Since the label is a single word "first" or "second", we can reconstruct them and check equality.
            # The label might be multiple tokens. Let's decode and compare full strings.
            # We'll do this per example: we know the label is at the end, so let's isolate per example.
            # A simpler approach: Just decode from the first label token to the end.
            # We'll do a per-example approach:

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                # Extract this example's label portion
                example_label_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
                if len(example_label_positions) == 0:
                    # No label tokens? skip
                    continue
                start_label = example_label_positions[0]
                end_label = example_label_positions[-1] + 1
                example_pred_tokens = torch.argmax(logits[i, start_label:end_label, :], dim=-1)
                example_target_tokens = labels[i, start_label:end_label]

                # Decode
                pred_str = tokenizer.decode(example_pred_tokens)
                target_str = tokenizer.decode(example_target_tokens)

                # Compare the normalized strings (strip spaces)
                # The target_str should be either "first" or "second"
                if pred_str.strip() == target_str.strip():
                    total_correct += 1
                total_samples += 1

    return total_correct / total_samples if total_samples > 0 else 0.0

def train_model(model, train_loader, dev_loader, test_loader, tokenizer, optimizer, device, save_dir, log_interval, max_steps, epochs, use_wandb):
    """
    Train the GPT-2 model.

    Similar to T5, we do gradient steps, log metrics, and save checkpoints.
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

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT-2 model for pronoun resolution.")

    # Add arguments
    parser.add_argument("--pretrained_model_name", type=str, default="gpt2", help="Pretrained GPT-2 model name or path.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval (steps).")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum number of training steps.")
    parser.add_argument("--use_wandb", type=bool, default=True, help="Whether to use wandb for logging.")
    parser.add_argument("--save_dir", type=str, default="./gpt2_checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data JSON file.")
    parser.add_argument("--dev_file", type=str, required=True, help="Path to the validation data JSON file.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the validation data JSON file.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize wandb
    now = datetime.now()
    datetime_string = now.strftime("%Y-%m-%d-%H%M")
    if args.use_wandb:
        wandb.init(
            project="ood-plms-coref",
            name=f"gpt2-{datetime_string}",
            config=vars(args)
        )

    # Load data
    train_data = load_data_from_json(args.train_file)
    dev_data = load_data_from_json(args.dev_file)
    test_data = load_data_from_json(args.test_file)

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model_name)
    # GPT-2 does not have a pad token by default, we set it if necessary
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.pretrained_model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Create data loaders
    train_dataset = PronounResolutionGPT2Dataset(train_data, tokenizer, args.max_length)
    dev_dataset = PronounResolutionGPT2Dataset(dev_data, tokenizer, args.max_length)
    test_dataset = PronounResolutionGPT2Dataset(test_data, tokenizer, args.max_length)
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
