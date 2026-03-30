
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

RANDOM_STATE = 42


def encode_categoricals(X):
    """Label encode categorical columns (proto, service)."""
    X = X.copy()
    le_dict = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
        print(f"  Encoded '{col}': {len(le.classes_)} unique values")
    return X, le_dict


def encode_target(y):
    """Label encode the target variable."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"  Encoded target: {len(le.classes_)} classes")
    print(f"  Mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
    return y_encoded, le


def split_data(X, y, test_size=0.2):
    """Stratified train-test split. Same seed always."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE
    )
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Test:  {X_test.shape[0]:,} samples")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Fit StandardScaler on train, transform both. Prevents data leakage."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    print(f"  Scaled features: mean ≈ 0, std ≈ 1")
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train):
    """Apply SMOTE to training data ONLY. Never apply to test data."""
    print(f"  Before SMOTE: {len(X_train):,} samples")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE:  {len(X_resampled):,} samples")
    return X_resampled, y_resampled


def apply_smote_tomek(X_train, y_train):
    """Apply SMOTE + Tomek Links to training data ONLY (hybrid strategy)."""
    print(f"  Before SMOTE+Tomek: {len(X_train):,} samples")
    smt = SMOTETomek(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
    print(f"  After SMOTE+Tomek:  {len(X_resampled):,} samples")
    return X_resampled, y_resampled


def get_class_weights(y_train):
    """Compute balanced class weights for cost-sensitive learning."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = dict(zip(classes, weights))
    print(f"  Computed class weights for {len(classes)} classes")
    return weight_dict


def full_preprocessing_pipeline(X, y):
    """
    Run the complete preprocessing pipeline.
    Returns all preprocessed data needed for experiments.
    
    NOTE: SMOTE is NOT applied here — it's applied per-experiment
    so we can compare baseline (no resampling) vs SMOTE vs cost-sensitive.
    """
    print("Step 1: Encoding categorical features...")
    X_encoded, le_dict = encode_categoricals(X)

    print("\nStep 2: Encoding target variable...")
    y_encoded, target_le = encode_target(y)

    print("\nStep 3: Stratified train-test split (80/20)...")
    X_train, X_test, y_train, y_test = split_data(X_encoded, y_encoded)

    print("\nStep 4: Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("\nPreprocessing complete!")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoders": le_dict,
        "target_encoder": target_le,
    }


# Quick test — run this file directly to verify
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_dataset, separate_features_target

    df = load_dataset()
    X, y = separate_features_target(df)
    result = full_preprocessing_pipeline(X, y)
    print(f"\nX_train shape: {result['X_train'].shape}")
    print(f"X_test shape:  {result['X_test'].shape}")