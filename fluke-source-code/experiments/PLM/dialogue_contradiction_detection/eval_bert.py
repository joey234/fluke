import os
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def preprocess_function(contexts, utterances, tokenizer, max_context_length=400, max_utterance_length=64):
    """
    Preprocesses dialogue contexts and last utterances by tokenizing and truncating them separately.

    Args:
        contexts (List[str]): Dialogue contexts.
        utterances (List[str]): Last utterances.
        tokenizer (AutoTokenizer): Tokenizer for tokenization.
        max_context_length (int): Maximum length for dialogue contexts.
        max_utterance_length (int): Maximum length for last utterances.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing input_ids and attention_mask.
    """
    # Tokenize dialogue contexts
    tokenized_contexts = tokenizer(
        contexts,
        padding=False,
        truncation=True,
        max_length=max_context_length,
    )

    # Tokenize last utterances
    tokenized_utterances = tokenizer(
        utterances,
        padding=False,
        truncation=True,
        max_length=max_utterance_length,
    )

    # Combine tokenized context and last utterance
    input_ids = [
        context + utterance[1:]  # Skip [CLS] from last utterance
        for context, utterance in zip(tokenized_contexts["input_ids"], tokenized_utterances["input_ids"])
    ]
    attention_masks = [
        context_mask + utterance_mask[1:]  # Combine attention masks
        for context_mask, utterance_mask in zip(tokenized_contexts["attention_mask"], tokenized_utterances["attention_mask"])
    ]

    # Pad combined input_ids and attention_masks
    batch = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_masks},
        padding=True,
        max_length=None,  # Dynamically pad to the longest sequence in the batch
        return_tensors="pt",
    )

    return batch


def evaluate_checkpoint(model, tokenizer, test_file, output_file, batch_size=16, max_context_length=400, max_utterance_length=64):
    """
    Evaluates a saved BERT checkpoint on a test dataset and saves predictions to a JSON file.

    Args:
        model (AutoModelForSequenceClassification): Pretrained BERT model.
        tokenizer (AutoTokenizer): Tokenizer corresponding to the BERT model.
        test_file (str): Path to the test dataset file (JSON).
        output_file (str): Path to the output JSON file to save predictions.
        batch_size (int): Number of examples to process in each batch.
        max_context_length (int): Maximum input length for dialogue contexts.
        max_utterance_length (int): Maximum input length for last utterances.
    """
    # Load the test dataset
    with open(test_file, "r") as f:
        test_data = json.load(f)

    # Prepare inputs
    contexts = [example["dialogue_context"] for example in test_data]
    utterances = [example["last_utterance"] for example in test_data]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Batch processing
    predictions = {}
    for i in tqdm(range(0, len(contexts), batch_size), desc="Processing test dataset"):
        batch_contexts = contexts[i: i + batch_size]
        batch_utterances = utterances[i: i + batch_size]
        batch_encodings = preprocess_function(
            batch_contexts, batch_utterances, tokenizer, max_context_length, max_utterance_length
        )

        batch_encodings = {key: torch.tensor(val).to(device) for key, val in batch_encodings.items()}

        with torch.no_grad():
            logits = model(**batch_encodings).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        for idx, pred in enumerate(preds):
            predictions[i + idx] = "contradictory" if pred == 1 else "not contradictory"

    dirname = os.path.dirname(output_file)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

    # Save predictions to JSON
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"Predictions saved to {output_file}")


def process_ood_test_files(model, tokenizer, test_dir, output_dir, batch_size=16, max_context_length=400, max_utterance_length=64):
    """
    Processes all OOD test files in a directory, performs inference, and saves predictions for each file.

    Args:
        model (AutoModelForSequenceClassification): Pretrained BERT model.
        tokenizer (AutoTokenizer): Tokenizer corresponding to the BERT model.
        test_dir (str): Directory containing OOD test files in JSON format.
        output_dir (str): Directory to save the output predictions in JSON format.
        batch_size (int): Number of examples to process in each batch.
        max_context_length (int): Maximum input length for dialogue contexts.
        max_utterance_length (int): Maximum input length for last utterances.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for test_file in os.listdir(test_dir):
        if not test_file.endswith("_100.json"):
            continue
        test_file_path = os.path.join(test_dir, test_file)
        output_file_path = os.path.join(output_dir, test_file)

        with open(test_file_path, "r") as f:
            test_data = json.load(f)

        with open("../datasets/train_dev_test_data/rongxin/train.json", "r") as file:
            full_test_data = json.load(file)

        contexts = []
        utterances = []
        record_ids = []

        for record in test_data:
            record_id, record_content = record
            record_ids.append(record_id)

            if "turns" in record_content:
                dialogue_context = " ".join(
                    [f"[AGENT{turn['agent_id']}] {turn['text']}" for turn in record_content["turns"][:-1]]
                )
                last_utterance = f"[AGENT{record_content['turns'][-1]['agent_id']}] {record_content['modified_text']}"
            else:
                x = full_test_data[record_id]
                dialogue_context = " ".join(
                    [f"[AGENT{turn['agent_id']}] {turn['text']}" for turn in x["turns"][:-1]]
                )
                last_utterance = f"[AGENT{x['turns'][-1]['agent_id']}] {record_content['modified_text']}"

            # dialogue_context = " ".join([f"[AGENT{turn['agent_id']}] {turn['text']}" for turn in content["turns"][:-1]])
            # last_utterance = f"[AGENT{content['turns'][-1]['agent_id']}] {content['modified_text']}"
            contexts.append(dialogue_context)
            utterances.append(last_utterance)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        predictions = {}
        for i in tqdm(range(0, len(contexts), batch_size), desc=f"Processing {test_file}"):
            batch_contexts = contexts[i: i + batch_size]
            batch_utterances = utterances[i: i + batch_size]
            batch_ids = record_ids[i: i + batch_size]
            batch_encodings = preprocess_function(
                batch_contexts, batch_utterances, tokenizer, max_context_length, max_utterance_length
            )

            batch_encodings = {key: torch.tensor(val).to(device) for key, val in batch_encodings.items()}

            with torch.no_grad():
                logits = model(**batch_encodings).logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

            for idx, pred in enumerate(preds):
                predictions[batch_ids[idx]] = "contradictory" if pred == 1 else "not contradictory"

        dirname = os.path.dirname(output_file_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        # Save predictions to JSON
        with open(output_file_path, "w") as f:
            json.dump(predictions, f, indent=4)

        print(f"Predictions saved to {output_file_path}")


# Example usage
if __name__ == "__main__":
    saved_model_path = "./tmp/bert-base-cased/checkpoint-9800"  # Replace with the path to your saved BERT checkpoint
    tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(saved_model_path)

    # Evaluate on original test file
    test_file_path = "../datasets/transformed_train_dev_test_data/rongxin/test.json"
    output_file_path = "./tmp/bert-base-cased_results/bert_predictions.json"
    evaluate_checkpoint(model, tokenizer, test_file_path, output_file_path, batch_size=8)

    # Evaluate on OOD test files
    test_dir = "../datasets/test_data_after_modifications/rongxin"
    output_dir = "./tmp/bert-base-cased_results/bert_ood_test_preds"
    process_ood_test_files(model, tokenizer, test_dir, output_dir, batch_size=8)
