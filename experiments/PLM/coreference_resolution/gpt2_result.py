import os
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd

# Paths (adjust these paths according to your setup)
test_file_path = "../datasets/transformed_train_dev_test_data/thinh/test.json"
ood_test_file_dir = "../datasets/test_data_after_modifications/thinh"

# GPT-2 predictions for the original test set
gpt2_predictions_path = "./tmp/gpt2_results/gpt2_predictions.json"
# GPT-2 predictions for OOD test sets
gpt2_ood_preds_dir = "./tmp/gpt2_results/gpt2_ood_test_preds"

# Load ground truth test data
with open(test_file_path, "r") as f:
    ground_truth_data = json.load(f)

# Create a mapping of indices to ground truth labels
ground_truth_labels = {
    str(i): int(item["label"]) for i, item in enumerate(ground_truth_data)
}

# Load GPT-2 predictions for the original test set
with open(gpt2_predictions_path, "r") as f:
    ori_predictions = json.load(f)

# Helper function to convert "first"/"second" to integers 0/1
def convert_pred_to_int(pred_str):
    pred_str = pred_str.strip().lower()
    if pred_str == "first":
        return 0
    elif pred_str == "second":
        return 1
    else:
        # If for some reason it's not "first" or "second", return -1 or handle differently
        return -1

results = []

# Loop through each OOD prediction file
for ood_file in tqdm(os.listdir(gpt2_ood_preds_dir), desc="Processing OOD prediction files"):
    if not ood_file.endswith(".json"):
        continue

    ood_pred_file_path = os.path.join(gpt2_ood_preds_dir, ood_file)
    with open(ood_pred_file_path, "r") as f:
        ood_predictions = json.load(f)

    # Load OOD data to get modified_label
    ood_data_file_path = os.path.join(ood_test_file_dir, ood_file)
    with open(ood_data_file_path, "r") as f:
        ood_data = json.load(f)

    # Map indices to OOD labels
    id2label = {str(x["index"]): int(x["modified_label"]) for x in ood_data}

    # Lists to store predictions and labels for OOD and original test
    ood_labels = []
    ood_preds = []
    ori_labels = []
    ori_preds = []

    # For each instance in OOD predictions
    for index, pred_str in ood_predictions.items():
        # Convert GPT-2 prediction to int
        ood_pred_int = convert_pred_to_int(pred_str)
        ood_preds.append(ood_pred_int)
        ood_labels.append(id2label[index])

        # Corresponding original prediction
        ori_pred_str = ori_predictions[index]
        ori_pred_int = convert_pred_to_int(ori_pred_str)
        ori_preds.append(ori_pred_int)
        ori_labels.append(ground_truth_labels[index])

    # Compute accuracy
    ood_accuracy = accuracy_score(ood_labels, ood_preds)
    ori_accuracy = accuracy_score(ori_labels, ori_preds)

    results.append({
        "file": ood_file.replace(".json", ""),
        "accuracy (ori)": f"{ori_accuracy:.3f}",
        "accuracy (ood)": f"{ood_accuracy:.3f}",
    })

# Save the results as a table
results_df = pd.DataFrame(results)
results_df.to_csv("./gpt2_accuracy_comparison.csv", index=False)

print("Accuracy comparison saved to gpt2_accuracy_comparison.csv")
