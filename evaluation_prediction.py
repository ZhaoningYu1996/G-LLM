import json
from datasets import load_dataset
import pandas as pd

pred_path = "predictions/BACE/sft_12_16B_v_2_train.csv"
data = pd.read_csv(pred_path)

pred = data['predictions']
labels = data['labels']
pred_res = []
for i in range(len(pred)):
    if "0" in pred[i]:
        # pred_res.append(0)
        pred[i] = 0
    elif "1" in pred[i]:    
        # pred_res.append(1)
        pred[i] = 1
    else:
        print(f"Error: {pred[i]}")
    if "0" in labels[i]:
        labels[i] = 0
    elif "1" in labels[i]:
        labels[i] = 1


# if "train" in pred_path:
#     print("================== Train Set =================")
#     pred = [pred[i] for i in range(len(pred)) if i % 2 == 1]
#     labels = [labels[i] for i in range(len(labels)) if i % 2 == 1]

# pred = [int(pred[i]) for i in range(len(pred))]
# pred = pred_res
# labels = [int(labels[i]) for i in range(len(labels))]
print(pred)
print(labels)

# Calculate accuracy
correct = 0
for i in range(len(pred)):
    if pred[i] == labels[i]:
        correct += 1
accuracy = correct / len(pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate ROC AUC
# from sklearn.metrics import roc_auc_score
# roc_auc = roc_auc_score(labels, pred)
# print(f"ROC AUC: {roc_auc:.2f}")