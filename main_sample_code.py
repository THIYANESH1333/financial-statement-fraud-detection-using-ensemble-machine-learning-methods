"""
Financial Statement Fraud Detection System
Main Sample Code for Project Report
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler

# =============================================================================
# FINANCIAL TEXT PREPROCESSING
# =============================================================================
def advanced_preprocess_text(text):
    """
    Advanced preprocessing with financial-specific enhancements
    """
    # Step 1: Basic cleaning
    text = text.lower()
    
    # Step 2: Preserve financial numbers and amounts
    text = re.sub(r'₹(\d+(?:,\d+)*(?:\.\d+)?)', 'INR_AMOUNT', text)
    text = re.sub(r'\$(\d+(?:,\d+)*(?:\.\d+)?)', 'USD_AMOUNT', text)
    text = re.sub(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:crore|lakh|million|billion)', 'AMOUNT_WORD', text)
    
    # Step 3: Preserve financial ratios and percentages
    text = re.sub(r'(\d+(?:\.\d+)?)\s*%', 'PERCENTAGE', text)
    text = re.sub(r'(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)', 'RATIO', text)
    
    # Step 4: Preserve financial years and quarters
    text = re.sub(r'FY\s*(\d{4})', 'FINANCIAL_YEAR', text)
    text = re.sub(r'Q(\d)\s*FY\s*(\d{4})', 'QUARTER_YEAR', text)
    
    # Step 5: Preserve company names and tickers
    text = re.sub(r'([A-Z]{2,5})\s*Ltd', 'COMPANY_TICKER', text)
    text = re.sub(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*(?:Ltd|Inc|Corp)', 'COMPANY_NAME', text)
    
    # Step 6: Remove general punctuation but preserve financial symbols
    text = re.sub(r'[^\w\s₹$%:,-]', ' ', text)
    
    # Step 7: Tokenize and clean
    tokens = nltk.word_tokenize(text)
    
    # Step 8: Advanced filtering
    financial_terms = {
        'revenue', 'profit', 'loss', 'ebitda', 'cash', 'flow', 'receivables', 'payables',
        'inventory', 'assets', 'liabilities', 'equity', 'auditor', 'audit', 'caro',
        'statutory', 'compliance', 'related', 'party', 'transaction', 'fraud', 'irregularity'
    }
    
    stop_words = set(stopwords.words('english'))
    enhanced_stop_words = stop_words - financial_terms
    lemmatizer = WordNetLemmatizer()
    
    filtered_tokens = []
    for token in tokens:
        if (token in financial_terms or 
            token.startswith(('INR_', 'USD_', 'AMOUNT_', 'PERCENTAGE_', 'RATIO_', 'FINANCIAL_', 'QUARTER_', 'COMPANY_')) or
            (len(token) > 2 and token not in enhanced_stop_words and not token.isdigit())):
            lemmatized = lemmatizer.lemmatize(token)
            filtered_tokens.append(lemmatized)
    
    return ' '.join(filtered_tokens)

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def create_sentiment_features(text):
    """Create sentiment-based features"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    return polarity, subjectivity

def create_fraud_red_flags(text):
    """Create fraud red flag indicators"""
    fraud_phrases = {
        'cash_flow_gap': ['operating cash flow', 'cash flow', 'gap versus net profit'],
        'related_party': ['related party', 'related parties', 'related party transaction'],
        'auditor_concerns': ['auditor', 'revenue recognition', 'emphasis of matter'],
        'statutory_issues': ['caro', 'statutory dues', 'non-compliance'],
        'receivables_issues': ['overdue receivables', 'bad debts', 'provision for doubtful debts']
    }
    
    flags = {}
    for category, phrases in fraud_phrases.items():
        flags[category] = sum(1 for phrase in phrases if phrase in text.lower())
    
    return flags

# =============================================================================
# MAIN FRAUD DETECTION SYSTEM
# =============================================================================
class FinancialFraudDetector:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=3000)
        self.scaler = StandardScaler()
        self.models = {}
        self.voting_classifier = None
        
    def prepare_features(self, texts):
        """Prepare features for training/prediction"""
        # Preprocess texts
        processed_texts = [advanced_preprocess_text(text) for text in texts]
        
        # TF-IDF features
        tfidf_features = self.tfidf_vectorizer.fit_transform(processed_texts).toarray()
        
        # Sentiment features
        sentiment_features = []
        for text in processed_texts:
            polarity, subjectivity = create_sentiment_features(text)
            sentiment_features.append([polarity, subjectivity])
        
        # Fraud red flags
        red_flag_features = []
        for text in processed_texts:
            flags = create_fraud_red_flags(text)
            red_flag_features.append(list(flags.values()))
        
        # Combine all features
        sentiment_array = np.array(sentiment_features)
        red_flag_array = np.array(red_flag_features)
        
        combined_features = np.hstack([
            tfidf_features,
            sentiment_array,
            red_flag_array
        ])
        
        return combined_features
    
    def train_models(self, X, y):
        """Train individual models and ensemble"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train individual models
        self.models['svm'] = SVC(probability=True, random_state=42)
        self.models['xgb'] = XGBClassifier(random_state=42)
        self.models['logistic'] = LogisticRegression(random_state=42)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_scaled, y)
        
        # Create voting classifier
        self.voting_classifier = VotingClassifier([
            ('svm', self.models['svm']),
            ('xgb', self.models['xgb']),
            ('logistic', self.models['logistic'])
        ], voting='soft')
        
        self.voting_classifier.fit(X_scaled, y)
        print("Training completed!")
    
    def predict(self, texts):
        """Make predictions on new texts"""
        features = self.prepare_features(texts)
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from voting classifier
        predictions = self.voting_classifier.predict(features_scaled)
        probabilities = self.voting_classifier.predict_proba(features_scaled)
        
        return predictions, probabilities
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        predictions, probabilities = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, probabilities[:, 1])
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_test, predictions)
        }

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main execution function"""
    print("Financial Statement Fraud Detection System")
    print("=" * 50)
    
    # Load data (replace with your actual data loading)
    # df = pd.read_csv('Final_Dataset.csv')
    # X_text = df['Fillings'].astype(str)
    # y = df['Fraud'].map({'no': 0, 'yes': 1})
    
    # For demonstration, create sample data
    sample_texts = [
        "The company reported strong revenue growth of 25% with operating cash flow of ₹100 million.",
        "There are significant related party transactions that require auditor attention.",
        "The company maintains healthy cash flow and profit margins.",
        "Operating cash flow was ₹20 million while net profit was ₹100 million, creating a gap.",
        "The financial statements show consistent performance across all metrics."
    ]
    
    sample_labels = [0, 1, 0, 1, 0]  # 0 = Normal, 1 = Fraud
    
    # Initialize detector
    detector = FinancialFraudDetector()
    
    # Prepare features
    print("Preparing features...")
    X = detector.prepare_features(sample_texts)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, sample_labels, test_size=0.2, random_state=42, stratify=sample_labels
    )
    
    # Train models
    print("Training models...")
    detector.train_models(X_train, y_train)
    
    # Evaluate
    print("\nModel Performance:")
    results = detector.evaluate_model(X_test, y_test)
    
    # Make predictions on new data
    new_texts = [
        "The company shows unusual cash flow patterns with significant gaps.",
        "Financial performance is consistent with industry standards."
    ]
    
    print("\nPredictions on new data:")
    predictions, probabilities = detector.predict(new_texts)
    
    for i, (text, pred, prob) in enumerate(zip(new_texts, predictions, probabilities)):
        fraud_prob = prob[1] * 100
        status = "FRAUD" if pred == 1 else "NORMAL"
        print(f"Text {i+1}: {status} (Confidence: {fraud_prob:.1f}%)")
        print(f"Content: {text[:50]}...")
        print()

if __name__ == "__main__":
    main()
