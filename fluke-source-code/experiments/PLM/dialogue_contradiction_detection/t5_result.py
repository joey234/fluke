import os
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def load_data_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

# Paths
test_file_path = "../datasets/transformed_train_dev_test_data/rongxin/test.json"
ood_file_dir = "../datasets/test_data_after_modifications/rongxin/"
t5_predictions_path = "tmp/t5-base_results/t5_predictions.json"
ood_preds_dir = "tmp/t5-base_results/t5_ood_test_preds"

# Load ground truth test data
with open(test_file_path, "r") as f:
    ground_truth_data = json.load(f)

# Load T5 predictions for the entire test set
with open(t5_predictions_path, "r") as f:
    t5_predictions = json.load(f)

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

# Loop through each file in the ood_preds_dir
rows = ["temporal_bias", "geographical_bias", "length_bias", "typo_bias",
        "capitalization", "punctuation", "derivation", "compound_word",
        "active_to_passive", "grammatical_role", "coordinating_conjunction",
        "concept_replacement", "negation", "discourse", "sentiment",
        "casual", "dialectal"]
# Initialize the results table
results = []
results_dict = dict()
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

    # Calculate accuracy for the same indices from t5_predictions
    t5_accuracy = compute_accuracy(t5_predictions, ood_indices, ground_truth_labels, label_mapping)

    # Add the results to the table
    results_dict[ood_file.replace("_100.json", "")] = (t5_accuracy, ood_accuracy)

for capability_test_name in rows:
    results.append((
        capability_test_name,
        results_dict[capability_test_name][0],
        results_dict[capability_test_name][1]
    ))

# Save the results as a table
import pandas as pd
results_df = pd.DataFrame(results)
results_df.to_csv("./t5_accuracy_comparison.csv", index=False)

print("Accuracy comparison saved to accuracy_comparison.csv")
