"""
SignBridge - final, one-time evaluation on the held-out test split.

Run this ONLY once you're satisfied with val performance in
train_classifier.py and are done tuning - that's the whole point of
keeping test separate from val. Reports accuracy on the ~416 test-split
rows (11 signers who never appear in train or val).

Takes the same optional feature-file argument as train_classifier.py and
derives the model path with train_classifier.model_path_for(), so the
model it loads always matches the features it evaluates. Before, this
script was hardwired to v1's features.csv + random_forest.joblib, which
would have quietly reported a v1 number for a v3 experiment.

Run:
    python evaluate_on_test.py                       -> v1
    python evaluate_on_test.py data/features_v3.csv  -> v3
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from train_classifier import model_path_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(BASE_DIR, sys.argv[1]) if len(sys.argv) > 1 \
    else os.path.join(BASE_DIR, "data", "features.csv")
MODEL_PATH = model_path_for(FEATURES_PATH)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found. Run train_classifier.py first.")
        return

    bundle = joblib.load(MODEL_PATH)
    clf, feature_cols = bundle["model"], bundle["feature_columns"]

    df = pd.read_csv(FEATURES_PATH)
    test_df = df[df["split"] == "test"]
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    print(f"Evaluating on {len(X_test)} held-out test clips (signers never seen in train/val)...")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nFINAL test accuracy: {acc:.3f}")

    print("\nPer-class report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(df["label"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("Confusion matrix (rows=true label, cols=predicted label):")
    print(cm_df.to_string())


if __name__ == "__main__":
    main()
