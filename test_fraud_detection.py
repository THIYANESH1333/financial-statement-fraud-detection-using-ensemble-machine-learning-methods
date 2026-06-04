#!/usr/bin/env python3
"""
Test Script for Financial Statement Fraud Detection
This script demonstrates how to use the fraud detection system with sample financial statements.
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

class FraudDetector:
    def __init__(self):
        """Initialize the fraud detector with sentiment dictionaries"""
        # Load sentiment dictionaries
        self.pos_words = pd.read_csv('positive.csv').iloc[:,0].str.lower().tolist()
        self.neg_words = pd.read_csv('negative.csv').iloc[:,0].str.lower().tolist()
        self.unc_words = pd.read_csv('uncertainty.csv').iloc[:,0].str.lower().tolist()
        self.lit_words = pd.read_csv('litigious.csv').iloc[:,0].str.lower().tolist()
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.analyzer = SentimentIntensityAnalyzer()
    
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        # Lowercase
        text = text.lower()
        # Remove numbers
        text = re.sub(r'\d+', '', text)
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
    
    def predict_fraud(self, text):
        """Predict fraud for given text"""
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Extract features
            pos_count = self.count_words_in_text(processed_text, self.pos_words)
            neg_count = self.count_words_in_text(processed_text, self.neg_words)
            unc_count = self.count_words_in_text(processed_text, self.unc_words)
            lit_count = self.count_words_in_text(processed_text, self.lit_words)
            
            # Sentiment analysis
            vader_scores = self.analyzer.polarity_scores(processed_text)
            polarity_score = TextBlob(processed_text).sentiment.polarity
            subjectivity_score = TextBlob(processed_text).sentiment.subjectivity
            
            # Calculate fraud score
            fraud_score = 0
            
            # Add scores based on features
            if neg_count > pos_count:
                fraud_score += 0.2
            if unc_count > 5:
                fraud_score += 0.3
            if lit_count > 3:
                fraud_score += 0.3
            if vader_scores['compound'] < -0.2:
                fraud_score += 0.2
            if abs(polarity_score) > 0.3:
                fraud_score += 0.1
            
            prediction = 1 if fraud_score > 0.5 else 0
            confidence = min(fraud_score + 0.3, 0.95) if prediction == 1 else max(0.7 - fraud_score, 0.3)
            
            return prediction, confidence, {
                'positive_words': pos_count,
                'negative_words': neg_count,
                'uncertainty_words': unc_count,
                'litigious_words': lit_count,
                'sentiment_polarity': polarity_score,
                'sentiment_subjectivity': subjectivity_score,
                'vader_compound': vader_scores['compound'],
                'fraud_score': fraud_score
            }
            
        except Exception as e:
            raise Exception(f"Error in prediction: {str(e)}")

def test_fraud_detection():
    """Test the fraud detection system with sample financial statements"""
    
    # Initialize the fraud detector
    detector = FraudDetector()
    
    # Sample financial statements for testing
    test_statements = {
        "High Risk - Excessive Negative Language": """
        Our company has experienced unprecedented losses this quarter, with revenue declining by 40% and net income turning negative. The future remains highly uncertain, and we cannot guarantee that our current difficulties will not continue indefinitely. We face potential legal action from multiple parties and regulatory investigations that could result in substantial penalties. The board of directors is considering all options including potential bankruptcy proceedings.
        """,
        
        "High Risk - Litigation and Uncertainty": """
        TechCorp Inc. faces various risks that could materially affect our business, financial condition, and results of operations. We are currently involved in several legal proceedings that could result in significant financial liabilities. Additionally, changes in regulations could require costly compliance measures or restrict our operations. Our supply chain is vulnerable to disruptions that could impact our ability to deliver products on time and at competitive prices. We also face risks related to cybersecurity threats and data breaches that may result in litigation.
        """,
        
        "Medium Risk - Balanced but Concerning": """
        While we faced some challenges this quarter, including supply chain disruptions and increased competition, we remain confident in our long-term prospects. Revenue declined 10% but we implemented cost-saving measures that partially offset the impact. We expect conditions to improve in the coming quarters, though the timing remains uncertain. Our restructuring plan is progressing but results may take longer than initially anticipated.
        """,
        
        "Low Risk - Positive Performance": """
        Our quarterly results reflect solid performance with revenue growth of 8% and improved profit margins. We successfully launched new products and expanded into new markets. While we face some competitive pressures, our strong market position and innovative product pipeline give us confidence in continued growth. Our cost-saving initiatives are delivering results ahead of schedule.
        """,
        
        "Low Risk - Transparent Disclosure": """
        Our fourth quarter results reflect both challenges and opportunities. While revenue declined 5% due to temporary supply chain issues, we successfully implemented cost-saving measures that improved our profit margins by 2%. We have resolved the supply chain challenges and expect to return to growth in the next quarter. Our new product launches are performing well, and we remain confident in our long-term strategy.
        """
    }
    
    print("🔍 FINANCIAL STATEMENT FRAUD DETECTION TEST")
    print("=" * 60)
    print()
    
    for title, statement in test_statements.items():
        print(f"📄 Testing: {title}")
        print("-" * 40)
        print(f"Statement: {statement[:100]}...")
        print()
        
        try:
            prediction, confidence, features = detector.predict_fraud(statement)
            
            print("🔍 ANALYSIS RESULTS:")
            if prediction == 1:
                print("🚨 FRAUD DETECTED!")
                print(f"   Confidence: {confidence*100:.2f}%")
                print("   ⚠️  This financial statement shows signs of potential fraud.")
            else:
                print("✅ NO FRAUD DETECTED")
                print(f"   Confidence: {confidence*100:.2f}%")
                print("   ✅ This financial statement appears to be legitimate.")
            
            print("\n📊 FEATURE ANALYSIS:")
            print(f"   • Positive words: {features['positive_words']}")
            print(f"   • Negative words: {features['negative_words']}")
            print(f"   • Uncertainty words: {features['uncertainty_words']}")
            print(f"   • Litigious words: {features['litigious_words']}")
            print(f"   • Sentiment polarity: {features['sentiment_polarity']:.3f}")
            print(f"   • Sentiment subjectivity: {features['sentiment_subjectivity']:.3f}")
            print(f"   • VADER compound score: {features['vader_compound']:.3f}")
            print(f"   • Fraud score: {features['fraud_score']:.3f}")
            
            print("\n💡 INTERPRETATION:")
            if features['negative_words'] > features['positive_words']:
                print("   • Higher negative sentiment detected")
            if features['uncertainty_words'] > 5:
                print("   • High uncertainty language detected")
            if features['litigious_words'] > 3:
                print("   • Litigious language detected")
            if abs(features['sentiment_polarity']) > 0.3:
                print("   • Strong sentiment detected")
            if features['fraud_score'] > 0.5:
                print("   • Multiple fraud indicators present")
            
        except Exception as e:
            print(f"❌ Error analyzing statement: {str(e)}")
        
        print("\n" + "=" * 60)
        print()

def interactive_test():
    """Interactive testing mode"""
    detector = FraudDetector()
    
    print("🔍 INTERACTIVE FRAUD DETECTION TESTING")
    print("=" * 50)
    print("Enter financial statements to analyze (type 'quit' to exit)")
    print()
    
    while True:
        print("-" * 50)
        statement = input("Enter financial statement text: ").strip()
        
        if statement.lower() in ['quit', 'exit', 'q']:
            print("👋 Thank you for testing the fraud detection system!")
            break
        
        if not statement:
            print("❌ Please enter some text to analyze.")
            continue
        
        try:
            prediction, confidence, features = detector.predict_fraud(statement)
            
            print("\n🔍 ANALYSIS RESULTS:")
            if prediction == 1:
                print("🚨 FRAUD DETECTED!")
                print(f"   Confidence: {confidence*100:.2f}%")
            else:
                print("✅ NO FRAUD DETECTED")
                print(f"   Confidence: {confidence*100:.2f}%")
            
            print(f"\n📊 Quick Analysis:")
            print(f"   • Negative words: {features['negative_words']}")
            print(f"   • Uncertainty words: {features['uncertainty_words']}")
            print(f"   • Litigious words: {features['litigious_words']}")
            print(f"   • Fraud score: {features['fraud_score']:.3f}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print()

if __name__ == "__main__":
    print("Financial Statement Fraud Detection Test Suite")
    print("Choose testing mode:")
    print("1. Run predefined test cases")
    print("2. Interactive testing mode")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        test_fraud_detection()
    elif choice == "2":
        interactive_test()
    else:
        print("Invalid choice. Running predefined tests...")
        test_fraud_detection()
