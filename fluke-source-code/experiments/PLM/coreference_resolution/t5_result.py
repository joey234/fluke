import os
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd

# Paths
dev_file_path = "../datasets/transformed_train_dev_test_data/thinh/dev.json"
test_file_path = "../datasets/transformed_train_dev_test_data/thinh/test.json"
ood_test_file_dir = "../datasets/test_data_after_modifications/thinh"
t5_dev_predictions_path = "./tmp/t5-base_results/t5_predictions_dev_original.json"
t5_test_predictions_path = "./tmp/t5-base_results/t5_predictions.json"
t5_ood_preds_dir = "./tmp/t5-base_results/t5_ood_test_preds"

# Load dev test data
with open(dev_file_path, "r") as f:
    dev_data = json.load(f)

# Load dev test data
with open(test_file_path, "r") as f:
    test_data = json.load(f)

# Create a mapping of indices to ground truth labels
dev_labels = {str(i): item["label"] for i, item in enumerate(dev_data)}
test_labels = {str(i): item["label"] for i, item in enumerate(test_data)}

# Load T5 predictions for the entire test set
with open(t5_dev_predictions_path, "r") as f:
    dev_predictions = json.load(f)

# Load T5 predictions for the entire dev set
with open(t5_test_predictions_path, "r") as f:
    test_predictions = json.load(f)


# Function to convert "first"/"second" to integer labels
def convert_pred_to_int(pred_str):
    pred_str = pred_str.strip().lower()
    if pred_str == "first":
        return 0
    elif pred_str == "second":
        return 1
    else:
        # If prediction doesn't match expected strings, handle gracefully or return a default
        # But ideally, it should always be "first" or "second"
        return -1


# New function to compute accuracy on the original test set
def compute_original_accuracy(ground_truth_labels, ori_predictions):
    ori_labels_all = []
    ori_preds_all = []
    for idx, true_label in ground_truth_labels.items():
        if idx in ori_predictions:
            ori_pred_str = ori_predictions[idx]
            ori_pred_int = convert_pred_to_int(ori_pred_str)
            ori_labels_all.append(true_label)
            ori_preds_all.append(ori_pred_int)
        else:
            print(f"can not find idx in predictions: {idx}")
    return accuracy_score(ori_labels_all, ori_preds_all)


# Compute and print accuracy on the original test set
dev_accuracy = compute_original_accuracy(dev_labels, dev_predictions)
print(f"Accuracy (dev set): {dev_accuracy:.4f}")

test_accuracy = compute_original_accuracy(test_labels, test_predictions)
print(f"Accuracy (test set): {test_accuracy:.4f}")

results = []
# Loop through each OOD prediction file
for ood_file in tqdm(os.listdir(t5_ood_preds_dir), desc="Processing OOD prediction files"):
    if not ood_file.endswith(".json"):
        continue

    ood_pred_file_path = os.path.join(t5_ood_preds_dir, ood_file)
    with open(ood_pred_file_path, "r") as f:
        ood_predictions = json.load(f)

    ood_data_file_path = os.path.join(ood_test_file_dir, ood_file)
    with open(ood_data_file_path, "r") as f:
        data = json.load(f)

    # Build index to label mapping for OOD data
    id2label = {str(x["index"]): int(x["modified_label"]) for x in data}

    # Lists to store predictions and labels
    ood_labels = []
    ood_preds = []
    ori_labels = []
    ori_preds = []

    # For each instance in OOD predictions
    for index, pred_str in ood_predictions.items():
        # Convert T5 prediction ("first"/"second") to int
        pred_int = convert_pred_to_int(pred_str)
        ood_preds.append(pred_int)
        ood_labels.append(id2label[index])

        # Get corresponding original prediction
        ori_pred_str = test_predictions[index]
        ori_pred_int = convert_pred_to_int(ori_pred_str)
        ori_preds.append(ori_pred_int)
        ori_labels.append(test_labels[index])

    # Compute accuracy
    ood_accuracy = accuracy_score(ood_labels, ood_preds)
    ori_accuracy = accuracy_score(ori_labels, ori_preds)

    results.append({
        "file": ood_file.replace(".json", ""),
        "accuracy (ori)": f"{ori_accuracy:.3f}",
        "accuracy (ood)": f"{ood_accuracy:.3f}",
    })

# Save results as a CSV
results_df = pd.DataFrame(results)
results_df.to_csv("./t5_accuracy_comparison.csv", index=False)

print("Accuracy comparison saved to t5_accuracy_comparison.csv")