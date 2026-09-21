"""
SignBridge - train the RandomForest sign classifier.

Trains on the ~482 `train`-split rows in data/features.csv, checks itself
against the ~124 `val`-split rows (different signers than train, used
for iterating - tuning parameters, seeing which words the model
confuses), and saves the trained model to models/random_forest.joblib.

IMPORTANT: the ~416 `test`-split rows are NOT touched by this script on
purpose. That set exists to report ONE honest, final accuracy number
once you're done tuning - see evaluate_on_test.py, kept as a separate
script so you don't run it by accident while iterating and end up
tuning against it (which would make the final number a lie).

Run:
    python train_classifier.py
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Optional first argument picks which feature file to train on, so v1 and
# v2 feature sets can be compared without editing this file:
#     python train_classifier.py                       -> data/features.csv    (v1)
#     python train_classifier.py data/features_v2.csv  -> v2 keyframe features
FEATURES_PATH = os.path.join(BASE_DIR, sys.argv[1]) if len(sys.argv) > 1 \
    else os.path.join(BASE_DIR, "data", "features.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")


def model_path_for(features_path):
    """
    One model file per feature file, named after it.

    The previous rule was `"_v2" if "_v2" in filename else ""`, which sent
    features_v2.csv and features_v2_baseline.csv to the SAME
    random_forest_v2.joblib - and that actually bit us once: an ensemble
    run silently compared a WLASL-trained RF against a clean-data GRU
    because the file underneath had been overwritten. With three v3
    feature sets (full / hand-local / body-only) being compared against
    each other, that collision would quietly invalidate the ablation.

        data/features.csv              -> models/random_forest.joblib
        data/features_v2.csv           -> models/random_forest_v2.joblib
        data/features_v3.csv           -> models/random_forest_v3.joblib
        data/features_v3_bodyonly.csv  -> models/random_forest_v3_bodyonly.joblib
    """
    stem = os.path.splitext(os.path.basename(features_path))[0]
    suffix = stem[len("features"):] if stem.startswith("features") else "_" + stem
    return os.path.join(MODEL_DIR, f"random_forest{suffix}.joblib")


MODEL_PATH = model_path_for(FEATURES_PATH)

NON_FEATURE_COLS = ("label", "split", "participant")


def load_split(df, feature_cols, split_name):
    subset = df[df["split"] == split_name]
    X = subset[feature_cols].values
    y = subset["label"].values
    return X, y


def main():
    if not os.path.exists(FEATURES_PATH):
        print(f"ERROR: {FEATURES_PATH} not found. Run build_feature_vectors.py first.")
        return

    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"Loaded {len(df)} rows, {len(feature_cols)} features, {df['label'].nunique()} classes")

    X_train, y_train = load_split(df, feature_cols, "train")
    X_val, y_val = load_split(df, feature_cols, "val")
    print(f"train: {len(X_train)} rows, val: {len(X_val)} rows")

    # class_weight='balanced' matters here: merged-variant words (EAT,
    # DRINK, HOW, WHAT, DEAF) have ~2x the samples of everything else,
    # since two ASL Citizen glosses got merged into one label for each.
    # Without this, the model would skew toward predicting those 5
    # words more often just because it saw more examples of them.
    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,     # a little regularization - 548 features vs ~480 train rows
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining...")
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"\nTrain accuracy: {train_acc:.3f}  (expect near-1.0 - this is memorization")
    print("                 capacity, not a meaningful number by itself)")
    print(f"Val accuracy:   {val_acc:.3f}  (THIS is the number that matters - signers")
    print("                 the model never saw during training)")

    print("\nPer-class report on val (precision/recall per word):")
    print(classification_report(y_val, val_pred, zero_division=0))

    print("Top 20 most important features (what the model actually relies on):")
    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(importances.head(20).to_string())

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": clf, "feature_columns": feature_cols}, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    print("\nNOTE: the test split was NOT touched by this script. Once you're happy with")
    print("val performance and done tuning, run evaluate_on_test.py ONCE for your final number.")


if __name__ == "__main__":
    main()
