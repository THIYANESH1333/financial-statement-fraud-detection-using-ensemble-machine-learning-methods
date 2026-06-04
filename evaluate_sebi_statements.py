#!/usr/bin/env python3
"""
Evaluate SEBI Fraud Detection on a Labeled Dataset

Input: CSV/Excel with columns:
- text: the financial statement text (SEBI format or similar)
- label: 1 for fraud/violation, 0 for non-fraud

Usage examples:
- python evaluate_sebi_statements.py --data statements.csv
- python evaluate_sebi_statements.py --data statements.xlsx --sheet Sheet1
"""

import argparse
import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Local import
from sebi_fraud_detector import SEBIFraudDetector


def load_dataset(path: str, sheet: str | None = None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv"]:
        df = pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=sheet) if sheet else pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .csv or .xlsx")
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")
    return df[["text", "label"]].dropna()


def evaluate(df: pd.DataFrame) -> dict:
    detector = SEBIFraudDetector()
    y_true = []
    y_pred = []
    scores = []

    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        label = int(row["label"])  # 0 or 1
        pred, conf, _ = detector.predict_fraud(text)
        y_true.append(label)
        y_pred.append(pred)
        scores.append(conf)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SEBI fraud detection on labeled statements")
    parser.add_argument("--data", required=True, help="Path to CSV/XLSX with columns: text,label")
    parser.add_argument("--sheet", default=None, help="Excel sheet name (if using xlsx)")
    args = parser.parse_args()

    try:
        df = load_dataset(args.data, args.sheet)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        sys.exit(1)

    print(f"📦 Loaded {len(df)} labeled statements from {args.data}")

    results = evaluate(df)

    print("\n=== SEBI FRAUD DETECTION EVALUATION ===")
    print(f"Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"Precision: {results['precision']*100:.2f}%")
    print(f"Recall:    {results['recall']*100:.2f}%")
    print(f"F1-Score:  {results['f1']*100:.2f}%")
    print("\nConfusion Matrix [TN FP; FN TP]:")
    print(results["confusion_matrix"])  # 2x2 matrix
    print("\nClassification Report:")
    print(results["report"])  # precision/recall/f1 per class


if __name__ == "__main__":
    main()
