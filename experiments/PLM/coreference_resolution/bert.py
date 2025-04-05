import argparse
from argparse import ArgumentParser

import torch
import json
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
from tqdm import tqdm
from datetime import datetime
import wandb
import os


# Dataset Class
class SentencePairDataset(Dataset):
    def __init__(self, sentence_pairs, labels):
        self.sentence_pairs = sentence_pairs
        self.labels = labels

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        s1, s2 = self.sentence_pairs[idx]
        label = self.labels[idx]
        return s1, s2, label


def collate_fn(batch, tokenizer, max_length):
    """
    Custom collate function for dynamic padding.
    """
    s1_list = [item[0] for item in batch]
    s2_list = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)

    # Tokenize and dynamically pad the input sequences
    encoded_s1 = tokenizer(s1_list, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    encoded_s2 = tokenizer(s2_list, truncation=True, padding=True, max_length=max_length, return_tensors="pt")

    return encoded_s1, encoded_s2, labels


def load_data_from_json(file_path):
    """
    Load and prepare data from a JSON file.
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        sentence_pairs (list): List of (sentence1, sentence2) pairs.
        labels (list): List of labels corresponding to the sentence pairs.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    sentence_pairs = [(item["sentence1"], item["sentence2"]) for item in data]
    labels = [item["label"] for item in data]
    return sentence_pairs, labels


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


# Evaluation Function
def evaluate(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            encoded_s1, encoded_s2, labels = batch
            input_ids_s1 = encoded_s1["input_ids"].squeeze(1).to(device)
            attention_mask_s1 = encoded_s1["attention_mask"].squeeze(1).to(device)
            input_ids_s2 = encoded_s2["input_ids"].squeeze(1).to(device)
            attention_mask_s2 = encoded_s2["attention_mask"].squeeze(1).to(device)
            labels = labels.to(device).float()

            logits = model(input_ids_s1, attention_mask_s1, input_ids_s2, attention_mask_s2)
            # Predictions
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    return accuracy


# Training Function
def train_model(model, train_loader, dev_loader, test_loader, optimizer, criterion, device, save_dir, log_interval, accumulation_steps, max_steps, use_wandb):
    print(f"accumulation steps: {accumulation_steps}")
    print(f"log interval: {log_interval}")

    model.train()
    total_loss = 0

    global_step = 0  # Track the total number of steps across all epochs
    checkpoint_paths = []  # Track the saved checkpoint paths

    with tqdm(total=max_steps) as pbar:
        step = 0
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}")
            if global_step >= max_steps:  # Stop training if maximum steps are reached
                break

            for batch in train_loader:
                step += 1
                if global_step >= max_steps:  # Stop training if maximum steps are reached
                    break

                encoded_s1, encoded_s2, labels = batch
                input_ids_s1 = encoded_s1["input_ids"].squeeze(1).to(device)
                attention_mask_s1 = encoded_s1["attention_mask"].squeeze(1).to(device)
                input_ids_s2 = encoded_s2["input_ids"].squeeze(1).to(device)
                attention_mask_s2 = encoded_s2["attention_mask"].squeeze(1).to(device)
                labels = labels.to(device).long()

                logits = model(input_ids_s1, attention_mask_s1, input_ids_s2, attention_mask_s2)

                loss = criterion(logits, labels)
                loss = loss / accumulation_steps  # Scale the loss for gradient accumulation
                loss.backward()
                total_loss += loss.item()

                # Gradient accumulation
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1  # Increment the global step
                    total_loss = 0.0
                    pbar.update(1)

                # Log accuracy and save checkpoint every log_interval steps
                if global_step > 0 and step % log_interval == 0 and global_step % log_interval == 0:
                    print("=== Evaluation Loop ===")
                    dev_accuracy = evaluate(model, dev_loader, device)
                    test_accuracy = evaluate(model, test_loader, device)
                    if use_wandb:
                        wandb.log({
                            "Global Step": global_step,
                            "Loss": total_loss / log_interval,
                            "Dev Accuracy": dev_accuracy,
                            "Test Accuracy": test_accuracy
                        })
                    total_loss = 0

                    # Save model checkpoint
                    checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{global_step}.pt")
                    torch.save(model.state_dict(), checkpoint_path)
                    checkpoint_paths.append(checkpoint_path)

                    # Remove older checkpoints to keep only the most recent 5
                    if len(checkpoint_paths) > 5:
                        oldest_checkpoint = checkpoint_paths.pop(0)
                        os.remove(oldest_checkpoint)
                        print(f"Deleted old checkpoint: {oldest_checkpoint}")

                    print(f"Model checkpoint saved at step {global_step}")

                # Stop training if max_steps is reached
                if global_step >= max_steps:
                    break


# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train a BERT binary classifier with customizable parameters.")

    # Add arguments
    parser.add_argument("--pretrained_model_name", type=str, default="bert-base-cased", help="Pretrained model name or path.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval (steps).")
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum number of training steps.")
    parser.add_argument("--use_wandb", type=bool, default=True, help="Whether to use wandb for logging.")
    parser.add_argument("--save_dir", type=str, default="./bert_checkpoints", help="Directory to save checkpoints.")

    return parser.parse_args()


# Main Execution
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Use parsed arguments in your script
    PRETRAINED_MODEL_NAME = args.pretrained_model_name
    MAX_LENGTH = args.max_length
    BATCH_SIZE = args.batch_size
    EVAL_BATCH_SIZE = args.eval_batch_size
    ACCUMULATION_STEPS = args.accumulation_steps
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    LOG_INTERVAL = args.log_interval
    MAX_STEPS = args.max_steps
    USE_WANDB = args.use_wandb
    SAVE_DIR = args.save_dir


    os.makedirs(SAVE_DIR, exist_ok=True)

    # Get the current date and time
    now = datetime.now()
    # Format the datetime string
    datetime_string = now.strftime("%Y-%m-%d-%H%M")

    # Initialize wandb
    if USE_WANDB:
        wandb.init(
            project="ood-plms-coref",
            name=f"bert-{datetime_string}",
            config={"max_steps": MAX_STEPS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE}
        )

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training and dev data from JSON files
    train_file_path = "../datasets/transformed_train_dev_test_data/thinh/train.json"
    dev_file_path = "../datasets/transformed_train_dev_test_data/thinh/dev.json"

    train_sentence_pairs, train_labels = load_data_from_json(train_file_path)
    dev_sentence_pairs, dev_labels = load_data_from_json(dev_file_path)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    # Dataset and DataLoader
    train_dataset = SentencePairDataset(train_sentence_pairs, train_labels)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, MAX_LENGTH)
    )
    dev_dataset = SentencePairDataset(dev_sentence_pairs, dev_labels)
    dev_loader = DataLoader(
        dev_dataset, batch_size=EVAL_BATCH_SIZE,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, MAX_LENGTH)
    )

    # Model, Optimizer, Criterion
    model = BertBinaryClassifier(PRETRAINED_MODEL_NAME).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train_model(model, train_loader, dev_loader, optimizer, criterion,
                device, SAVE_DIR, LOG_INTERVAL, ACCUMULATION_STEPS, MAX_STEPS, use_wandb=USE_WANDB)