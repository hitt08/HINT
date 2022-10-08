import os
import torch
from sklearn.metrics import balanced_accuracy_score,precision_score,recall_score,fbeta_score

def load_model(model_id,prefix="model_",dir="checkpoints"):
    model_file=f"{dir}/{prefix}{model_id}"
    if os.path.exists(model_file):
        model = torch.load(model_file)
    else:
        print("Model not Found")
        model=None
    return model

def evaluate_model(pred_labels, true_labels):
    precision = precision_score(true_labels, pred_labels, average='binary')
    recall = recall_score(true_labels, pred_labels, average='binary')
    accuracy = balanced_accuracy_score(true_labels, pred_labels)
    f1 = fbeta_score(pred_labels, true_labels, 1, average='binary')  # 1 means f_1 measure
    pred_classes = len(set(pred_labels))
    true_classes = len(set(true_labels))

    return "BAC=%0.3f P=%0.3f R=%0.3f F1=%0.3f Predicted_Classes=%d True_Classes=%d" % (
    accuracy, precision, recall, f1, pred_classes, true_classes)