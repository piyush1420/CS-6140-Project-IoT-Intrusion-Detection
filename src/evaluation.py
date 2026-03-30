"""
Shared evaluation utilities.
Both Piyush and Mahip import from here so results are directly comparable.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(y_true, y_pred, target_names=None, model_name="Model"):
    """
    Compute all evaluation metrics.
    Returns a dict with macro_f1, accuracy, and per-class report.
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    print(f"\n{'='*50}")
    print(f" {model_name}")
    print(f"{'='*50}")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"{'='*50}")
    print(classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0
    ))

    return {
        "model": model_name,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "report": report,
    }


def plot_confusion_matrix(y_true, y_pred, target_names, model_name="Model", save_path=None):
    """Plot a 13x13 confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def results_to_dataframe(results_list):
    """Convert a list of evaluate_model() outputs to a DataFrame."""
    rows = []
    for r in results_list:
        rows.append({
            "model": r["model"],
            "macro_f1": r["macro_f1"],
            "accuracy": r["accuracy"],
        })
    return pd.DataFrame(rows)


def save_results(results_list, filepath):
    """Save results to CSV."""
    df = results_to_dataframe(results_list)
    df.to_csv(filepath, index=False)
    print(f"  Results saved to {filepath}")
    return df
