import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from textblob import TextBlob
from xgboost import XGBClassifier

# ----------------------------
# Minimal financial preprocessing
# ----------------------------
def preprocess_text_basic(text: str) -> str:
	if not isinstance(text, str):
		text = str(text)
	text = text.lower()
	text = re.sub(r"₹(\d+(?:,\d+)*(?:\.\d+)?)", "INR_AMOUNT", text)
	text = re.sub(r"\$(\d+(?:,\d+)*(?:\.\d+)?)", "USD_AMOUNT", text)
	text = re.sub(r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:crore|lakh|million|billion)", "AMOUNT_WORD", text)
	text = re.sub(r"(\d+(?:\.\d+)?)\s*%", "PERCENTAGE", text)
	text = re.sub(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", "RATIO", text)
	text = re.sub(r"fy\s*(\d{4})", "FINANCIAL_YEAR", text)
	text = re.sub(r"q(\d)\s*fy\s*(\d{4})", "QUARTER_YEAR", text)
	text = re.sub(r"[^\w\s₹$%:,-]", " ", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text

# ----------------------------
# Sentiment + simple red flags
# ----------------------------
FRAUD_PHRASES = {
	'cash_flow_gap': ['operating cash flow', 'cash flow gap', 'profit without cash'],
	'related_party': ['related party', 'rpt', 'related parties'],
	'auditor_concerns': ['qualified opinion', 'emphasis of matter', 'auditor concern', 'revenue recognition'],
	'statutory_issues': ['statutory dues', 'non-compliance', 'gst dues', 'income tax dues'],
	'receivables_issues': ['overdue receivables', 'bad debts', 'provision for doubtful debts']
}

def make_aux_features(texts):
	# sentiment
	pol = np.array([TextBlob(t).sentiment.polarity for t in texts], dtype=float).reshape(-1, 1)
	subj = np.array([TextBlob(t).sentiment.subjectivity for t in texts], dtype=float).reshape(-1, 1)
	# red flags counts
	flags = []
	for t in texts:
		lc = t.lower()
		row = [sum(1 for p in FRAUD_PHRASES[k] if p in lc) for k in FRAUD_PHRASES]
		flags.append(row)
	flags = np.array(flags, dtype=float)
	return np.hstack([pol, subj, flags])

# ----------------------------
# Load data
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, 'Final_Dataset.csv')

df = pd.read_csv(DATA_PATH)
# Expect columns: Fillings (text), Fraud (yes/no or 1/0)
X_text = df['Fillings'].astype(str)
y_raw = df['Fraud']
if y_raw.dtype == object:
	y = y_raw.map({'no': 0, 'yes': 1}).fillna(y_raw).astype(int)
else:
	y = y_raw.astype(int)

# Optional: filter extremely long docs to avoid memory spikes
mask = X_text.str.len() < 1_000_000
X_text = X_text[mask]
y = y[mask]

# ----------------------------
# Prepare features (TF-IDF + sentiment + red flags)
# ----------------------------
X_text_clean = X_text.apply(preprocess_text_basic)

# TF-IDF using more vocabulary for "more documents"
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
X_tfidf = tfidf.fit_transform(X_text_clean).toarray()

X_aux = make_aux_features(X_text_clean)
X_all = np.hstack([X_tfidf, X_aux])

scaler = StandardScaler(with_mean=False)  # sparse-friendly pattern (we passed dense but keep safety)
X_all_scaled = scaler.fit_transform(X_all)

# ----------------------------
# Train/test split and model
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
	X_all_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
	n_estimators=400,
	max_depth=6,
	learning_rate=0.08,
	subsample=0.9,
	colsample_bytree=0.9,
	reg_lambda=1.0,
	random_state=42,
	n_jobs=-1
)

print("Training XGBoost on expanded TF-IDF features...")
model.fit(X_train, y_train)

# ----------------------------
# Evaluation: Confusion Matrix
# ----------------------------
print("Evaluating...")
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nConfusion Matrix (XGBoost):")
print(cm)
print("\nMetrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Save to text file
out_path = os.path.join(BASE_DIR, 'xgb_confusion_matrix.txt')
with open(out_path, 'w', encoding='utf-8') as f:
	f.write("Confusion Matrix (XGBoost)\n")
	f.write(str(cm) + "\n\n")
	f.write("Metrics\n")
	f.write(f"Accuracy:  {acc:.4f}\n")
	f.write(f"Precision: {prec:.4f}\n")
	f.write(f"Recall:    {rec:.4f}\n")
	f.write(f"F1-Score:  {f1:.4f}\n\n")
	f.write("Classification Report\n")
	f.write(classification_report(y_test, y_pred, digits=4))

print(f"\nSaved confusion matrix and metrics to: {out_path}")
