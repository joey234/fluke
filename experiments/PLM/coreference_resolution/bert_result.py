import os
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd

# Paths
test_file_path = "../datasets/transformed_train_dev_test_data/thinh/test.json"
ood_test_file_dir = "../datasets/test_data_after_modifications/thinh"
bert_predictions_path = "./tmp/bert-base-cased_results/bert_predictions.json"
ood_preds_dir = "./tmp/bert-base-cased_results/bert_ood_test_preds"

# Load ground truth test data
with open(test_file_path, "r") as f:
    ground_truth_data = json.load(f)

# Load BERT predictions for the entire test set
with open(bert_predictions_path, "r") as f:
    ori_predictions = json.load(f)

# Create a mapping of indices to ground truth labels
ground_truth_labels = {
    str(i): int(item["label"]) for i, item in enumerate(ground_truth_data)
}

labels = [int(item["label"]) for i, item in enumerate(ground_truth_data)]


# Initialize the results table
results = []

# Loop through each file in the ood_preds_dir
for ood_file in tqdm(os.listdir(ood_preds_dir), desc="Processing OOD prediction files"):
    if not ood_file.endswith("_100.json"):
        continue

    ood_pred_file_path = os.path.join(ood_preds_dir, ood_file)
    with open(ood_pred_file_path, "r") as f:
        ood_predictions = json.load(f)

    ood_data_file_path = os.path.join(ood_test_file_dir, ood_file)
    id2label = dict()
    with open(ood_data_file_path, "r") as f:
        data = json.load(f)
        for x in data:
            id2label[str(x["index"])] = x["modified_label"]

    # for accuracy computation on ood test set
    ood_labels = []
    ood_preds = []
    # for accuracy computation on original test set
    ori_labels = []
    ori_preds = []

    for index, pred in ood_predictions.items():
        ood_preds.append(pred)
        ood_labels.append(id2label[index])

        ori_preds.append(ori_predictions[index])
        ori_labels.append(ground_truth_labels[index])


    ood_accuracy = accuracy_score(ood_labels, ood_preds)
    ori_accuracy = accuracy_score(ori_labels, ori_preds)

    # Add the results to the table
    results.append({
        "file": ood_file.replace(".json", ""),
        "accuracy (ori)": ori_accuracy,
        "accuracy (ood)": ood_accuracy,
    })

# Save the results as a table
results_df = pd.DataFrame(results)
results_df.to_csv("./bert_accuracy_comparison.csv", index=False)

print("Accuracy comparison saved to bert_accuracy_comparison.csv")