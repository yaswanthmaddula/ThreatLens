"""
train_model.py
URL-only training pipeline for URLShield.

IMPORTANT: This script trains exclusively on features that are available
at inference time (URL text only). This eliminates the 54-feature vs
11-feature mismatch from the original pipeline.

Run from the backend/ directory:
    python train_model.py
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from feature_extraction import extract_features

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_PATH = os.environ.get("DATASET_PATH", "../dataset/phishing_urls.csv")
MODEL_OUTPUT_PATH = os.environ.get("MODEL_PATH", "url_model.pkl")
METRICS_OUTPUT_PATH = "model_metrics.json"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 300
N_JOBS = -1  # use all CPU cores

# ---------------------------------------------------------------------------
# These are the EXACT feature keys returned by extract_features()
# that will be used as model inputs. Keys starting with _ are excluded
# (they are explanation-only). This list must stay in sync with
# feature_extraction.py's return dict.
# ---------------------------------------------------------------------------
MODEL_FEATURE_COLUMNS = [
    "URLLength",
    "DomainLength",
    "TLDLength",
    "NoOfSubDomain",
    "IsHTTPS",
    "NoOfLettersInURL",
    "NoOfDegitsInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "URLEntropy",
    "DomainEntropy",
    "PathEntropy",
    "DigitRatioInURL",
    "LetterRatioInURL",
    "NoOfHyphensInURL",
    "NoOfDotsInURL",
    "NoOfAtInURL",
    "NoOfPercentInURL",
    # URLDepth excluded: dataset artifact — safe URLs in dataset are all root
    # domains (depth=0), which would cause false positives on legit /path URLs.
    "IsIPAddress",
    "HasPunycode",
    "HasAtSign",
    "HasDoubleSlashRedirect",
    "HasHexEncoding",
    "IsSuspiciousTLD",
    "SuspiciousKeywordCount",
    "BrandInSubdomain",
]


def build_features_from_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-extract URL-only features from the URL column of the dataset.
    This guarantees training and inference use identical feature engineering.
    """
    print(f"Extracting features from {len(df):,} URLs — this may take a few minutes...")
    records = []
    for i, url in enumerate(df["URL"]):
        if i % 20000 == 0:
            print(f"  [{i:,} / {len(df):,}]")
        feats = extract_features(str(url))
        # Keep only model input features (no _ prefixed explanation fields)
        records.append({k: feats[k] for k in MODEL_FEATURE_COLUMNS})
    return pd.DataFrame(records, columns=MODEL_FEATURE_COLUMNS)


def main():
    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    print(f"\nLoading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    print(f"  (0 = malicious, 1 = safe)\n")

    # -----------------------------------------------------------------------
    # Build feature matrix from URL strings
    # -----------------------------------------------------------------------
    X = build_features_from_dataset(df)
    y = df["label"].values

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features used ({len(MODEL_FEATURE_COLUMNS)}): {MODEL_FEATURE_COLUMNS}\n")

    # -----------------------------------------------------------------------
    # Train / test split (stratified to preserve class ratio)
    # -----------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    # -----------------------------------------------------------------------
    # Train model
    # -----------------------------------------------------------------------
    print(f"\nTraining RandomForestClassifier (n_estimators={N_ESTIMATORS})...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",  # handles slight class imbalance
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    model.fit(X_train, y_train)
    print("Training complete.\n")

    # -----------------------------------------------------------------------
    # Evaluate on held-out test set
    # -----------------------------------------------------------------------
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Malicious class index (label=0)
    class_list = list(model.classes_)
    malicious_idx = class_list.index(0) if 0 in class_list else 0
    y_proba_malicious = y_proba[:, malicious_idx]

    # Binary metrics: positive class = malicious (0)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    # roc_auc_score expects proba of the positive class.
    # Since sklearn treats label=1 as positive by default,
    # we pass the probability of the safe class (index 1) for standard AUC,
    # then invert so higher = more malicious.
    safe_idx = class_list.index(1) if 1 in class_list else 1
    # Passing malicious proba with pos_label=0 gives correct AUC orientation.
    roc_auc = roc_auc_score((y_test == 0).astype(int), y_proba_malicious)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("=" * 55)
    print("  EVALUATION METRICS (held-out test set)")
    print("=" * 55)
    print(f"  Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision : {precision:.4f}  (malicious class)")
    print(f"  Recall    : {recall:.4f}  (malicious class)")
    print(f"  F1 Score  : {f1:.4f}  (malicious class)")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {cm}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Malicious(0)", "Safe(1)"]))
    print("=" * 55)

    # -----------------------------------------------------------------------
    # 5-fold cross-validation (F1 on malicious class)
    # -----------------------------------------------------------------------
    print("\nRunning 5-fold stratified cross-validation (F1, malicious class)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=N_JOBS)
    print(f"  CV F1 scores : {[round(s, 4) for s in cv_scores]}")
    print(f"  CV F1 mean   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # -----------------------------------------------------------------------
    # Feature importance
    # -----------------------------------------------------------------------
    importances = model.feature_importances_
    feature_importance = sorted(
        zip(MODEL_FEATURE_COLUMNS, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\n  Feature Importance (top 15):")
    for feat, imp in feature_importance[:15]:
        bar = "█" * int(imp * 200)
        print(f"  {feat:<35} {imp:.4f}  {bar}")

    # -----------------------------------------------------------------------
    # Save metrics to JSON
    # -----------------------------------------------------------------------
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "cv_f1_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_std": round(float(cv_scores.std()), 4),
        "confusion_matrix": cm,
        "feature_count": len(MODEL_FEATURE_COLUMNS),
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "feature_importance": [
            {"feature": f, "importance": round(float(i), 6)}
            for f, i in feature_importance
        ],
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "n_estimators": N_ESTIMATORS,
    }

    with open(METRICS_OUTPUT_PATH, "w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"\nMetrics saved to: {METRICS_OUTPUT_PATH}")

    # -----------------------------------------------------------------------
    # Save model + feature column list
    # -----------------------------------------------------------------------
    joblib.dump((model, MODEL_FEATURE_COLUMNS), MODEL_OUTPUT_PATH)
    print(f"Model saved to: {MODEL_OUTPUT_PATH}\n")

    return metrics


if __name__ == "__main__":
    main()
