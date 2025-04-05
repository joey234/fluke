import os
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd


def load_data_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


# Paths
test_file_path = "../datasets/transformed_train_dev_test_data/rongxin/test.json"
ood_file_dir = "../datasets/test_data_after_modifications/rongxin/"
gpt2_predictions_path = "./tmp/gpt2_results/gpt2_predictions.json"
ood_preds_dir = "./tmp/gpt2_results/gpt2_ood_test_preds"

# Load ground truth test data
with open(test_file_path, "r") as f:
    ground_truth_data = json.load(f)

# Load GPT-2 predictions for the entire test set
with open(gpt2_predictions_path, "r") as f:
    gpt2_predictions = json.load(f)

# Create a mapping of indices to ground truth labels
ground_truth_labels = {
    str(i): int(item["label"]) for i, item in enumerate(ground_truth_data)
}

# Function to compute accuracy
def compute_accuracy(predictions, indices, ground_truth_labels, label_mapping):
    y_true = [ground_truth_labels[idx] for idx in indices]
    y_pred = [label_mapping[predictions[idx]] for idx in indices]
    return accuracy_score(y_true, y_pred)

# Map string labels to numeric labels
label_mapping = {"contradictory": 1, "not contradictory": 0}

# Initialize the results table
results = []

# Loop through each file in the ood_preds_dir
for ood_file in tqdm(os.listdir(ood_preds_dir), desc="Processing OOD prediction files"):
    if not ood_file.endswith(".json"):
        continue

    ood_ori_file_path = os.path.join(ood_file_dir, ood_file)
    idx2label = dict()
    ood_data = load_data_from_json(ood_ori_file_path)
    for index, d in ood_data:
        idx2label[str(index)] = d["label"]

    ood_file_path = os.path.join(ood_preds_dir, ood_file)
    with open(ood_file_path, "r") as f:
        ood_predictions = json.load(f)

    # Get the indices in the current OOD file
    ood_indices = list(ood_predictions.keys())

    # Calculate accuracy for OOD predictions
    ood_labels = []
    ood_preds = []
    for idx, pred in ood_predictions.items():
        ood_labels.append(idx2label[idx])
        ood_preds.append(label_mapping[pred])
    ood_accuracy = accuracy_score(ood_labels, ood_preds)
    # Calculate accuracy for the same indices from gpt2_predictions
    gpt2_accuracy = compute_accuracy(gpt2_predictions, ood_indices, ground_truth_labels, label_mapping)

    # Add the results to the table
    results.append({
        "file": ood_file.replace(".json", ""),
        "accuracy (ori)": gpt2_accuracy,
        "accuracy (ood)": ood_accuracy,
    })

# Save the results as a table
results_df = pd.DataFrame(results)
results_df.to_csv("./gpt2_accuracy_comparison.csv", index=False)

print("Accuracy comparison saved to gpt2_accuracy_comparison.csv")