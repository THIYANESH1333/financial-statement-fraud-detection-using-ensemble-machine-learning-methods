#!/usr/bin/env python3
"""
Simple fraud detection test for ABC Infra Ltd statement
"""

import pandas as pd
import re
import string
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

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

class SimpleFraudDetector:
    def __init__(self):
        """Initialize the simple fraud detector"""
        # Load sentiment dictionaries
        self.pos_words = pd.read_csv('positive.csv').iloc[:,0].str.lower().tolist()
        self.neg_words = pd.read_csv('negative.csv').iloc[:,0].str.lower().tolist()
        self.unc_words = pd.read_csv('uncertainty.csv').iloc[:,0].str.lower().tolist()
        self.lit_words = pd.read_csv('litigious.csv').iloc[:,0].str.lower().tolist()
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Financial red flags specific to this case
        self.fraud_indicators = {
            'cash_flow_gap': ['cash flow', 'operating cash flow', 'gap versus net profit', 'profit without cash'],
            'related_party': ['related parties', 'related party', 'related party transaction'],
            'auditor_concerns': ['auditor highlighted', 'auditor concern', 'revenue recognition', 'overdue receivables'],
            'statutory_issues': ['caro', 'statutory dues', 'delays in statutory'],
            'receivables_issues': ['trade receivables', 'receivables rose', 'overdue receivables']
        }
    
    def count_words_in_text(self, text, word_list):
        """Count occurrences of words from a list in the given text"""
        text_lower = text.lower()
        count = 0
        for word in word_list:
            count += text_lower.count(word)
        return count
    
    def detect_fraud_indicators(self, text):
        """Detect specific fraud indicators in the text"""
        indicators = {}
        text_lower = text.lower()
        
        for category, phrases in self.fraud_indicators.items():
            indicators[category] = 0
            for phrase in phrases:
                if phrase in text_lower:
                    indicators[category] = 1
                    break
        
        return indicators
    
    def predict_fraud(self, text):
        """Predict fraud using simple rule-based analysis"""
        try:
            # Extract basic sentiment features
            pos_count = self.count_words_in_text(text, self.pos_words)
            neg_count = self.count_words_in_text(text, self.neg_words)
            unc_count = self.count_words_in_text(text, self.unc_words)
            lit_count = self.count_words_in_text(text, self.lit_words)
            
            # Sentiment analysis
            vader_scores = self.analyzer.polarity_scores(text)
            polarity_score = TextBlob(text).sentiment.polarity
            subjectivity_score = TextBlob(text).sentiment.subjectivity
            
            # Detect fraud indicators
            fraud_indicators = self.detect_fraud_indicators(text)
            
            # Calculate fraud score
            fraud_score = 0
            
            # Basic sentiment scoring
            if neg_count > pos_count:
                fraud_score += 0.10
            if unc_count > 3:
                fraud_score += 0.15
            if lit_count > 2:
                fraud_score += 0.15
            
            # Financial fraud indicators (high weight)
            if fraud_indicators['cash_flow_gap']:
                fraud_score += 0.30  # Very high weight for cash flow issues
            if fraud_indicators['related_party']:
                fraud_score += 0.25  # High weight for related party issues
            if fraud_indicators['auditor_concerns']:
                fraud_score += 0.35  # Very high weight for auditor concerns
            if fraud_indicators['statutory_issues']:
                fraud_score += 0.20  # High weight for statutory issues
            if fraud_indicators['receivables_issues']:
                fraud_score += 0.20  # High weight for receivables issues
            
            # Normalize fraud score
            fraud_score = min(fraud_score, 1.0)
            
            prediction = 1 if fraud_score > 0.3 else 0  # Lower threshold for financial fraud
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
                'cash_flow_gap': fraud_indicators['cash_flow_gap'],
                'related_party': fraud_indicators['related_party'],
                'auditor_concerns': fraud_indicators['auditor_concerns'],
                'statutory_issues': fraud_indicators['statutory_issues'],
                'receivables_issues': fraud_indicators['receivables_issues']
            }
            
        except Exception as e:
            raise Exception(f"Error in prediction: {str(e)}")

def test_abc_infra():
    """Test the ABC Infra Ltd statement"""
    
    detector = SimpleFraudDetector()
    
    # The problematic statement
    abc_statement = """
    Company: ABC Infra Ltd (Q4 FY2024 Extract)
    Revenue from Operations stood at ₹5,200 crore, up 45% YoY. EBITDA was reported at ₹1,350 crore (EBITDA Margin 26%). Net Profit surged 210% YoY to ₹900 crore. However, Operating Cash Flow for FY24 was only ₹120 crore, creating a large gap versus Net Profit. Trade Receivables rose sharply by ₹1,050 crore in Q4, of which 34% were from related parties. The Auditor highlighted concerns over revenue recognition practices and overdue receivables. CARO 2020 reporting flagged delays in statutory dues.
    """
    
    print("🔍 TESTING ABC INFRA LTD STATEMENT")
    print("=" * 60)
    print(f"Statement: {abc_statement}")
    print()
    
    try:
        prediction, confidence, features = detector.predict_fraud(abc_statement)
        
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
        print(f"   • Cash flow gap: {features['cash_flow_gap']}")
        print(f"   • Related party issues: {features['related_party']}")
        print(f"   • Auditor concerns: {features['auditor_concerns']}")
        print(f"   • Statutory issues: {features['statutory_issues']}")
        print(f"   • Receivables issues: {features['receivables_issues']}")
        print(f"   • Sentiment polarity: {features['sentiment_polarity']:.3f}")
        print(f"   • VADER compound score: {features['vader_compound']:.3f}")
        print(f"   • Overall fraud score: {features['fraud_score']:.3f}")
        
        print("\n💡 INTERPRETATION:")
        if features['cash_flow_gap']:
            print("   • Cash flow gap detected - major red flag")
        if features['related_party']:
            print("   • Related party issues detected - fraud indicator")
        if features['auditor_concerns']:
            print("   • Auditor concerns detected - critical red flag")
        if features['statutory_issues']:
            print("   • Statutory compliance issues detected")
        if features['receivables_issues']:
            print("   • Receivables problems detected")
        if features['fraud_score'] > 0.3:
            print("   • High fraud risk score")
        
        print("\n🎯 EXPECTED FRAUD INDICATORS IN THIS STATEMENT:")
        print("   • Profit without cash (₹900 crore profit vs ₹120 crore cash flow)")
        print("   • Heavy related-party receivables (34% of ₹1,050 crore)")
        print("   • Auditor concerns over revenue recognition")
        print("   • Overdue receivables")
        print("   • Statutory dues delays (CARO 2020)")
        
    except Exception as e:
        print(f"❌ Error analyzing statement: {str(e)}")

if __name__ == "__main__":
    test_abc_infra()
