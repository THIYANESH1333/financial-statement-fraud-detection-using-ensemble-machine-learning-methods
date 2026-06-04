#!/usr/bin/env python3
"""
Advanced Financial Statement Fraud Detection System
Considers financial red flags, cash flow anomalies, and auditor concerns
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

def safe_nltk_download(resource):
    """Safely download NLTK resources"""
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1])

# Download required NLTK resources
safe_nltk_download('corpora/stopwords')
safe_nltk_download('tokenizers/punkt')
safe_nltk_download('corpora/wordnet')
safe_nltk_download('vader_lexicon')

class AdvancedFraudDetector:
    def __init__(self):
        """Initialize the advanced fraud detector"""
        # Load sentiment dictionaries
        self.pos_words = pd.read_csv('positive.csv').iloc[:,0].str.lower().tolist()
        self.neg_words = pd.read_csv('negative.csv').iloc[:,0].str.lower().tolist()
        self.unc_words = pd.read_csv('uncertainty.csv').iloc[:,0].str.lower().tolist()
        self.lit_words = pd.read_csv('litigious.csv').iloc[:,0].str.lower().tolist()
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Financial red flags and fraud indicators
        self.financial_red_flags = {
            'cash_flow_anomalies': [
                'negative cash flow', 'negative operating cash flow', 'cash flow negative',
                'operating cash flow negative', 'cash outflow', 'cash burn',
                'cash flow from operations negative', 'negative cash from operations'
            ],
            'inventory_issues': [
                'inventory doubled', 'inventory increased significantly', 'inventory buildup',
                'excess inventory', 'inventory write down', 'inventory impairment',
                'inventory valuation', 'emphasis of matter inventory', 'inventory levels'
            ],
            'revenue_concentration': [
                'single customer', 'one customer', 'major customer', 'key customer',
                'customer concentration', 'revenue concentration', 'sales concentration',
                'distributor concentration', 'export concentration', 'geographic concentration'
            ],
            'auditor_concerns': [
                'emphasis of matter', 'qualified opinion', 'adverse opinion',
                'disclaimer of opinion', 'going concern', 'material uncertainty',
                'auditor concern', 'audit qualification', 'audit emphasis',
                'auditor report emphasis', 'audit opinion emphasis'
            ],
            'accounting_irregularities': [
                'restatement', 'accounting error', 'material misstatement',
                'accounting irregularity', 'financial restatement', 'prior period adjustment',
                'accounting change', 'policy change', 'estimate change'
            ],
            'related_party_issues': [
                'related party', 'related party transaction', 'related party disclosure',
                'related party relationship', 'related party agreement'
            ],
            'unusual_patterns': [
                'unusual', 'unprecedented', 'significant increase', 'dramatic increase',
                'sudden spike', 'unexplained', 'anomaly', 'irregular', 'abnormal'
            ],
            'debt_issues': [
                'debt covenant', 'debt default', 'debt restructuring', 'debt refinancing',
                'high debt', 'excessive debt', 'debt burden', 'leverage'
            ]
        }
        
        # SEBI and regulatory specific indicators
        self.regulatory_red_flags = [
            'sebi investigation', 'regulatory investigation', 'regulatory action',
            'regulatory penalty', 'regulatory fine', 'regulatory violation',
            'compliance issue', 'regulatory concern', 'regulatory notice'
        ]
    
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        # Lowercase
        text = text.lower()
        # Remove numbers but keep financial amounts
        text = re.sub(r'\b\d+(?:\.\d+)?\s*(?:crore|crs|lakh|lac|million|billion|thousand|k|m|b)\b', 'AMOUNT', text)
        # Remove other numbers
        text = re.sub(r'\b\d+(?:\.\d+)?\b', 'NUMBER', text)
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize using NLTK
        tokens = nltk.word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)
    
    def count_words_in_text(self, text, word_list):
        """Count occurrences of words from a list in the given text"""
        text_lower = text.lower()
        count = 0
        for word in word_list:
            count += text_lower.count(word)
        return count
    
    def extract_financial_features(self, text):
        """Extract financial red flags and fraud indicators"""
        features = {}
        
        for category, terms in self.financial_red_flags.items():
            features[f'{category}_count'] = self.count_words_in_text(text, terms)
        
        # Regulatory red flags
        features['regulatory_red_flags'] = self.count_words_in_text(text, self.regulatory_red_flags)
        
        # Extract specific financial patterns
        features['cash_flow_negative'] = 1 if any(phrase in text.lower() for phrase in ['negative cash flow', 'negative operating cash flow']) else 0
        features['inventory_issues'] = 1 if any(phrase in text.lower() for phrase in ['inventory doubled', 'inventory increased', 'inventory buildup']) else 0
        features['revenue_concentration'] = 1 if any(phrase in text.lower() for phrase in ['single customer', 'one customer', 'major customer', 'distributor']) else 0
        features['auditor_emphasis'] = 1 if any(phrase in text.lower() for phrase in ['emphasis of matter', 'qualified opinion', 'auditor concern']) else 0
        
        return features
    
    def predict_fraud(self, text):
        """Predict fraud using advanced financial analysis"""
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Extract basic sentiment features
            pos_count = self.count_words_in_text(processed_text, self.pos_words)
            neg_count = self.count_words_in_text(processed_text, self.neg_words)
            unc_count = self.count_words_in_text(processed_text, self.unc_words)
            lit_count = self.count_words_in_text(processed_text, self.lit_words)
            
            # Sentiment analysis
            vader_scores = self.analyzer.polarity_scores(processed_text)
            polarity_score = TextBlob(processed_text).sentiment.polarity
            subjectivity_score = TextBlob(processed_text).sentiment.subjectivity
            
            # Extract financial features
            financial_features = self.extract_financial_features(text)
            
            # Calculate comprehensive fraud score
            fraud_score = 0
            
            # Basic sentiment scoring (lower weight)
            if neg_count > pos_count:
                fraud_score += 0.10
            if unc_count > 5:
                fraud_score += 0.15
            if lit_count > 3:
                fraud_score += 0.15
            if vader_scores['compound'] < -0.2:
                fraud_score += 0.10
            if abs(polarity_score) > 0.3:
                fraud_score += 0.05
            
            # Financial red flags (higher weight)
            if financial_features['cash_flow_negative'] > 0:
                fraud_score += 0.25  # High weight for cash flow issues
            if financial_features['inventory_issues'] > 0:
                fraud_score += 0.20  # High weight for inventory problems
            if financial_features['revenue_concentration'] > 0:
                fraud_score += 0.20  # High weight for revenue concentration
            if financial_features['auditor_emphasis'] > 0:
                fraud_score += 0.30  # Very high weight for auditor concerns
            
            # Financial pattern scoring
            for category in ['cash_flow_anomalies', 'inventory_issues', 'revenue_concentration', 
                           'auditor_concerns', 'accounting_irregularities']:
                if financial_features[f'{category}_count'] > 0:
                    fraud_score += 0.15 * financial_features[f'{category}_count']
            
            # Regulatory issues
            if financial_features['regulatory_red_flags'] > 0:
                fraud_score += 0.25
            
            # Normalize fraud score
            fraud_score = min(fraud_score, 1.0)
            
            prediction = 1 if fraud_score > 0.4 else 0  # Lower threshold for financial red flags
            confidence = min(fraud_score + 0.2, 0.95) if prediction == 1 else max(0.6 - fraud_score, 0.2)
            
            return prediction, confidence, {
                'positive_words': pos_count,
                'negative_words': neg_count,
                'uncertainty_words': unc_count,
                'litigious_words': lit_count,
                'sentiment_polarity': polarity_score,
                'sentiment_subjectivity': subjectivity_score,
                'vader_compound': vader_scores['compound'],
                'fraud_score': fraud_score,
                'cash_flow_negative': financial_features['cash_flow_negative'],
                'inventory_issues': financial_features['inventory_issues'],
                'revenue_concentration': financial_features['revenue_concentration'],
                'auditor_emphasis': financial_features['auditor_emphasis'],
                'financial_red_flags': sum(financial_features[f'{category}_count'] for category in self.financial_red_flags.keys()),
                'regulatory_red_flags': financial_features['regulatory_red_flags']
            }
            
        except Exception as e:
            raise Exception(f"Error in prediction: {str(e)}")

def test_advanced_fraud_detection():
    """Test the advanced fraud detection system"""
    
    detector = AdvancedFraudDetector()
    
    # Test cases including the problematic statement
    test_statements = {
        "High Risk - Cash Flow & Inventory Issues": """
        Revenue from Operations stood at ₹2,850 crore, up 65% YoY. EBITDA margin expanded to 29% versus 18% YoY, reporting EBITDA of ₹820 crore. Net Profit was ₹610 crore, compared to just ₹120 crore in Q4 FY23. However, Inventory levels doubled to ₹1,500 crore, while sales growth was driven largely by a sudden spike in exports to one distributor accounting for 42% of total sales. Operating Cash Flow for the year was negative ₹300 crore despite reported profits. Auditor's Report included an Emphasis of Matter on valuation of inventory and export receivables.
        """,
        
        "High Risk - Auditor Concerns": """
        The company reported strong revenue growth of 25% and improved profitability. However, the statutory auditors have issued a qualified opinion citing material uncertainties regarding the valuation of certain assets and the company's ability to continue as a going concern. The audit report includes an emphasis of matter regarding related party transactions and potential conflicts of interest.
        """,
        
        "Medium Risk - Revenue Concentration": """
        The company achieved revenue growth of 15% with improved margins. However, a significant portion of our revenue (35%) comes from a single major customer, which poses concentration risk. While we have diversified our customer base, this concentration remains a concern for future growth sustainability.
        """,
        
        "Low Risk - Normal Operations": """
        The company delivered solid performance with revenue growth of 12% and improved operational efficiency. Operating cash flow was positive and in line with net profits. Inventory levels remained stable and well-managed. The auditors issued an unqualified opinion with no emphasis of matter.
        """
    }
    
    print("🔍 ADVANCED FINANCIAL FRAUD DETECTION")
    print("=" * 70)
    print()
    
    for title, statement in test_statements.items():
        print(f"📄 Testing: {title}")
        print("-" * 50)
        print(f"Statement: {statement[:120]}...")
        print()
        
        try:
            prediction, confidence, features = detector.predict_fraud(statement)
            
            print("🔍 ANALYSIS RESULTS:")
            if prediction == 1:
                print("🚨 FRAUD DETECTED!")
                print(f"   Confidence: {confidence*100:.2f}%")
                print("   ⚠️  This financial statement shows signs of potential fraud or financial irregularities.")
            else:
                print("✅ NO FRAUD DETECTED")
                print(f"   Confidence: {confidence*100:.2f}%")
                print("   ✅ This financial statement appears to be legitimate.")
            
            print("\n📊 FEATURE ANALYSIS:")
            print(f"   • Positive words: {features['positive_words']}")
            print(f"   • Negative words: {features['negative_words']}")
            print(f"   • Uncertainty words: {features['uncertainty_words']}")
            print(f"   • Litigious words: {features['litigious_words']}")
            print(f"   • Cash flow negative: {features['cash_flow_negative']}")
            print(f"   • Inventory issues: {features['inventory_issues']}")
            print(f"   • Revenue concentration: {features['revenue_concentration']}")
            print(f"   • Auditor emphasis: {features['auditor_emphasis']}")
            print(f"   • Financial red flags: {features['financial_red_flags']}")
            print(f"   • Regulatory red flags: {features['regulatory_red_flags']}")
            print(f"   • Sentiment polarity: {features['sentiment_polarity']:.3f}")
            print(f"   • VADER compound score: {features['vader_compound']:.3f}")
            print(f"   • Overall fraud score: {features['fraud_score']:.3f}")
            
            print("\n💡 INTERPRETATION:")
            if features['cash_flow_negative'] > 0:
                print("   • Negative cash flow detected - major red flag")
            if features['inventory_issues'] > 0:
                print("   • Inventory problems detected - potential fraud indicator")
            if features['revenue_concentration'] > 0:
                print("   • Revenue concentration risk detected")
            if features['auditor_emphasis'] > 0:
                print("   • Auditor concerns detected - critical red flag")
            if features['financial_red_flags'] > 2:
                print("   • Multiple financial red flags present")
            if features['fraud_score'] > 0.4:
                print("   • High fraud risk score")
            
        except Exception as e:
            print(f"❌ Error analyzing statement: {str(e)}")
        
        print("\n" + "=" * 70)
        print()

if __name__ == "__main__":
    test_advanced_fraud_detection()
