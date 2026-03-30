# CS-6140-Project-IoT-Intrusion-Detection
import pandas as pd
import os


def load_dataset(data_path=None):
    """
    Load the RT-IoT2022 dataset.
    
    Parameters:
        data_path (str): Path to the dataset file. 
                         Defaults to 'data/RT_IOT2022' relative to project root.
    
    Returns:
        pd.DataFrame: The full dataset with the unnamed index column dropped.
    """
    if data_path is None:
        # Default path relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_root, "data", "RT_IOT2022")
    
    df = pd.read_csv(data_path, index_col=0)  # index_col=0 handles the unnamed index
    
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"Features: {df.shape[1] - 1}  |  Target: 'Attack_type'")
    
    return df


def separate_features_target(df, target_col="Attack_type"):
    """
    Separate the DataFrame into features (X) and target (y).
    
    Parameters:
        df (pd.DataFrame): Full dataset.
        target_col (str): Name of the target column.
    
    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y


def print_class_distribution(y, title="Class Distribution"):
    """
    Print class distribution with counts, percentages, and a visual bar.
    
    Parameters:
        y (pd.Series): Target labels.
        title (str): Title for the printout.
    """
    counts = y.value_counts()
    percentages = y.value_counts(normalize=True) * 100
    total = len(y)
    
    print(f"\n{'='*65}")
    print(f" {title}")
    print(f" Total samples: {total:,}")
    print(f"{'='*65}")
    print(f"{'Class':<30} {'Count':>8} {'%':>8}  Bar")
    print(f"{'-'*65}")
    
    for cls in counts.index:
        count = counts[cls]
        pct = percentages[cls]
        bar = "█" * int(pct / 2)  # Scale bar to fit
        print(f"{cls:<30} {count:>8,} {pct:>7.2f}%  {bar}")
    
    print(f"{'-'*65}")
    print(f"{'Unique classes:':<30} {y.nunique()}")
    print(f"{'='*65}\n")


def get_feature_info(X):
    """
    Print summary of feature types (numerical vs categorical).
    
    Parameters:
        X (pd.DataFrame): Feature matrix.
    """
    numerical = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    
    print(f"Total features: {len(X.columns)}")
    print(f"  Numerical:   {len(numerical)}")
    print(f"  Categorical: {len(categorical)} → {categorical}")
    print(f"  Missing values: {X.isnull().sum().sum()}")
    
    return numerical, categorical


# Quick test — run this file directly to verify everything works
if __name__ == "__main__":
    df = load_dataset()
    X, y = separate_features_target(df)
    print_class_distribution(y)
    numerical, categorical = get_feature_info(X)