"""
Shared data loading utilities.
Both Piyush and Mahip import from here to ensure consistency.
"""
import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_dataset():
    """Load RT-IoT2022 dataset from UCI repo."""
    rt_iot2022 = fetch_ucirepo(id=942)
    X = rt_iot2022.data.features
    y = rt_iot2022.data.targets
    return X, y


def load_from_csv(filepath):
    """Load dataset from local CSV file."""
    df = pd.read_csv(filepath)
    X = df.drop(columns=["Attack_type"])
    y = df["Attack_type"]
    return X, y


def print_class_distribution(y, title="Class Distribution"):
    """Print class distribution with counts and percentages."""
    dist = y.value_counts()
    total = len(y)
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f" Total samples: {total:,}")
    print(f"{'='*60}")
    for cls, count in dist.items():
        pct = count / total * 100
        print(f"  {cls:<35} {count:>7,}  ({pct:5.2f}%)")
    print(f"{'='*60}\n")


def dataset_summary(X, y):
    """Print a quick summary of the dataset."""
    print(f"Features shape: {X.shape}")
    print(f"Target shape:   {y.shape}")
    print(f"Missing values: {X.isnull().sum().sum()}")
    print(f"\nData types:")
    print(X.dtypes.value_counts())
    print(f"\nCategorical columns: {list(X.select_dtypes(include='object').columns)}")
    print_class_distribution(y)
