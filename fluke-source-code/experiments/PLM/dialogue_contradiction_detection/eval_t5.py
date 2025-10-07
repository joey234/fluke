import json
import os

import torch

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset

from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq


def evaluate_checkpoint(model, tokenizer, test_file, output_file):
    """
    Evaluates a saved T5 checkpoint on a test dataset and saves the decoded predictions to a text file.

    Args:
        saved_checkpoint (str): Path to the saved T5 checkpoint.
        test_file (str): Path to the test dataset file (CSV or JSON).
        output_file (str): Path to the output text file to save predictions.
    """
    # Load the test dataset
    test_dataset = load_dataset(
        "csv" if test_file.endswith(".csv") else "json",
        data_files={"test": test_file},
    )["test"]

    # Preprocess the test dataset
    def preprocess_function(examples):
        inputs = [
            f"Is the following dialogue contradictory? Context: {context} Utterance: {utterance}"
            for context, utterance in zip(examples["dialogue_context"], examples["last_utterance"])
        ]
        return tokenizer(inputs, max_length=512, padding="max_length", truncation=True)

    test_dataset = test_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.remove_columns("label")

    # Set up the data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model)

    # Set up Seq2SeqTrainer
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        logging_dir="./logs",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Predict on the test set
    predictions = trainer.predict(test_dataset).predictions
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    dirname = os.path.dirname(output_file)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

    # Save decoded predictions to a text file
    with open(output_file, "w") as f:
        output_dict = dict()
        for i, prediction in enumerate(decoded_predictions):
            output_dict[i] = prediction
        json.dump(output_dict, f)

    print(f"Predictions saved to {output_file}")


def process_test_file(model, tokenizer, test_file_path, output_file_path, batch_size=16, max_input_length=512,
                      max_output_length=10):
    """
    Processes a single test file, performs inference in batches, and saves predictions to an output file.

    Args:
        model (T5ForConditionalGeneration): Pretrained T5 model.
        tokenizer (T5Tokenizer): Tokenizer corresponding to the T5 model.
        test_file_path (str): Path to the test file in JSON format.
        output_file_path (str): Path to save the output predictions in JSON format.
        batch_size (int): Number of examples to process in each batch.
        max_input_length (int): Maximum input sequence length for tokenization.
        max_output_length (int): Maximum output sequence length for generation.
    """
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load and preprocess test data
    with open(test_file_path, "r") as f:
        test_data = json.load(f)

    with open("../datasets/train_dev_test_data/rongxin/train.json", "r") as file:
        full_test_data = json.load(file)

    # Prepare data for inference
    test_ids = []
    test_inputs = []
    for record in test_data:
        record_id = record[0]
        record_content = record[1]
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

        test_ids.append(record_id)
        test_inputs.append(
            f"Is the following dialogue contradictory? Context: {dialogue_context} Utterance: {last_utterance}")

    # Function to process a single batch
    def process_batch(batch_inputs, batch_ids):
        encodings = tokenizer(batch_inputs, padding=True, truncation=True, max_length=max_input_length,
                              return_tensors="pt")
        encodings = {key: tensor.to(device) for key, tensor in encodings.items()}
        with torch.no_grad():
            generated_ids = model.generate(encodings["input_ids"], max_length=max_output_length)
        decoded_predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return {record_id: prediction for record_id, prediction in zip(batch_ids, decoded_predictions)}

    # Batch processing
    output_predictions = {}
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(test_inputs), batch_size), desc=f"Processing {test_file_path}"):
            batch_inputs = test_inputs[i:i + batch_size]
            batch_ids = test_ids[i:i + batch_size]
            batch_predictions = process_batch(batch_inputs, batch_ids)
            output_predictions.update(batch_predictions)

    # Save the predictions to a JSON file
    dirname = os.path.dirname(output_file_path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(output_file_path, "w") as f:
        json.dump(output_predictions, f, indent=4)

    print(f"Predictions saved to {output_file_path}")


# Example Usage
if __name__ == "__main__":
    saved_checkpoint_path = "./tmp/t5-base/checkpoint-10000"  # Path to your saved T5 checkpoint
    test_file_path = "../datasets/transformed_train_dev_test_data/rongxin/test.json"  # Path to your test dataset

    # Load tokenizer and model from the saved checkpoint
    tokenizer = T5Tokenizer.from_pretrained(saved_checkpoint_path)
    model = T5ForConditionalGeneration.from_pretrained(saved_checkpoint_path)

    output_file_path = "./tmp/t5-base_results/t5_predictions.json"  # Path to save decoded predictions

    # get predictions on the original test file
    evaluate_checkpoint(model, tokenizer, test_file_path, output_file_path)

    # get predictions on ood instances
    output_dir = "./tmp/t5-base_results/t5_ood_test_preds"
    for test_file in os.listdir("../datasets/test_data_after_modifications/rongxin"):
        if not test_file.endswith("_100.json"):
            continue
        if os.path.exists(os.path.join(output_dir, test_file)):
            continue
        print(f"Predict on {test_file}")
        output_file = f'{output_dir}/{test_file}'
        process_test_file(
            model, tokenizer,
            f"../datasets/test_data_after_modifications/rongxin/{test_file}",
            output_file,
            batch_size=8
        )