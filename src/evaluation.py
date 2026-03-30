
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred, target_encoder=None):
    """
    Compute all key metrics for the project.
    
    Parameters:
        y_true: True labels (encoded)
        y_pred: Predicted labels (encoded)
        target_encoder: LabelEncoder for target (to get class names)
    
    Returns:
        dict with macro_f1, accuracy, per-class report DataFrame
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Get class names if encoder provided
    if target_encoder is not None:
        target_names = target_encoder.classes_
    else:
        target_names = [str(i) for i in sorted(np.unique(np.concatenate([y_true, y_pred])))]

    report_dict = classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).T

    print(f"\n{'='*50}")
    print(f"  Macro F1 (PRIMARY):  {macro_f1:.4f}")
    print(f"  Accuracy:            {accuracy:.4f}")
    print(f"  Macro Precision:     {macro_precision:.4f}")
    print(f"  Macro Recall:        {macro_recall:.4f}")
    print(f"{'='*50}\n")

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "report_df": report_df,
    }


def print_per_class_report(report_df):
    """
    Print a clean per-class performance table.
    
    Parameters:
        report_df: DataFrame from compute_metrics()
    """
    # Filter to only class rows (exclude accuracy, macro avg, weighted avg)
    class_rows = report_df.iloc[:-3]  # last 3 rows are summary rows

    print(f"{'Class':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 72)
    for cls, row in class_rows.iterrows():
        print(f"{cls:<30} {row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1-score']:>10.4f} {row['support']:>10.0f}")
    print("-" * 72)


def plot_confusion_matrix(y_true, y_pred, target_encoder=None, title="Confusion Matrix",
                          figsize=(12, 10), save_path=None):
    """
    Plot a confusion matrix heatmap.
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        target_encoder: LabelEncoder for class names
        title: Plot title
        figsize: Figure size
        save_path: If provided, saves the figure to this path
    """
    if target_encoder is not None:
        labels = target_encoder.classes_
    else:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved confusion matrix to {save_path}")

    plt.show()


def compare_experiments(results_dict):
    """
    Compare multiple experiments side by side.
    
    Parameters:
        results_dict: dict of {experiment_name: metrics_dict}
                      where metrics_dict comes from compute_metrics()
    
    Returns:
        pd.DataFrame with one row per experiment
    """
    rows = []
    for name, metrics in results_dict.items():
        rows.append({
            "Experiment": name,
            "Macro F1": metrics["macro_f1"],
            "Accuracy": metrics["accuracy"],
            "Macro Precision": metrics["macro_precision"],
            "Macro Recall": metrics["macro_recall"],
        })

    df = pd.DataFrame(rows).set_index("Experiment")
    df = df.sort_values("Macro F1", ascending=False)

    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPARISON (sorted by Macro F1)")
    print("=" * 70)
    print(df.to_string(float_format="{:.4f}".format))
    print("=" * 70 + "\n")

    return df


def save_results(metrics, experiment_name, save_dir="results"):
    """
    Save experiment results to CSV.
    
    Parameters:
        metrics: dict from compute_metrics()
        experiment_name: name for the file
        save_dir: directory to save in
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Save per-class report
    report_path = os.path.join(save_dir, f"{experiment_name}_report.csv")
    metrics["report_df"].to_csv(report_path)
    print(f"  Saved report to {report_path}")

    # Save summary metrics
    summary_path = os.path.join(save_dir, f"{experiment_name}_summary.csv")
    summary = pd.DataFrame([{
        "experiment": experiment_name,
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
    }])
    summary.to_csv(summary_path, index=False)
    print(f"  Saved summary to {summary_path}")


# Quick test
if __name__ == "__main__":
    # Simulate a small test with dummy data
    np.random.seed(42)
    y_true = np.random.randint(0, 12, size=500)
    y_pred = y_true.copy()
    # Introduce some errors
    noise_idx = np.random.choice(500, size=50, replace=False)
    y_pred[noise_idx] = np.random.randint(0, 12, size=50)

    print("Testing evaluation utilities with dummy data...")
    metrics = compute_metrics(y_true, y_pred)
    print_per_class_report(metrics["report_df"])
    print("\nAll evaluation functions working!")