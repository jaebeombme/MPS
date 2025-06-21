import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    confusion_matrix,
    ConfusionMatrixDisplay,
)

LABEL_MAP_INV= {
    0: "t1",
    1: "t2",
    2: "t1ce",
    3: "flair"
}

def compute_per_class_accuracy(y_true, y_pred, num_classes=4):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    for i in range(len(y_true)):
        gt = y_true[i]
        pred = y_pred[i]
        total_per_class[gt] += 1
        if pred == gt:
            correct_per_class[gt] += 1

    accuracy_dict = {}
    for c in range(num_classes):
        total = total_per_class[c]
        correct = correct_per_class[c]
        class_name = LABEL_MAP_INV.get(c, f"class_{c}")
        accuracy = correct / total if total > 0 else 0.0
        accuracy_dict[class_name] = accuracy

    return accuracy_dict

def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def compute_precision(y_true, y_pred, average="macro"):
    return precision_score(y_true, y_pred, average=average, zero_division=0)

def compute_recall(y_true, y_pred, average="macro"):
    return recall_score(y_true, y_pred, average=average, zero_division=0)

def compute_f1_score(y_true, y_pred, average="macro"):
    return f1_score(y_true, y_pred, average=average, zero_division=0)

def plot_confusion(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def show_gt_pred_table(y_true, y_pred):
    df = pd.DataFrame({"GT": y_true, "Pred": y_pred})
    print("\nGT vs Prediction Table:")
    print(df)