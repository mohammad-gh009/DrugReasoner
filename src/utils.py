import argparse
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score
)



def parse_args():
    parser = argparse.ArgumentParser(description="Model Evaluation or Dataset Loading")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--evaluate",
        type=str,
        help="Path to CSV file for evaluation (with y_true and y_pred columns)",
    )

    return parser.parse_args()


def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    TN, FP = cm[0][0], cm[0][1]
    FN, TP = cm[1][0], cm[1][1]

    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    NPV = TN / (TN + FN) if (TN + FN) != 0 else 0

    auc = roc_auc_score(y_true, y_pred)

    results = {
        "F1 score": f1,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "AUC": auc,
        "Specificity": specificity,
        "NPV": NPV,
    }
    return results


def main():
    args = parse_args()

    if args.evaluate:
        df = pd.read_csv(args.evaluate)
        df["y_pred"].replace({"approved": 1.0, "unapproved": 0.0}, inplace=True)
        df["y_true"].replace({"approved": 1.0, "unapproved": 0.0}, inplace=True)
        y_pred = [1 if i == 1 else 0 for i in df["y_pred"].tolist()]
        evaluate(df["y_true"].tolist(), y_pred)

if __name__ == "__main__":
    main()
