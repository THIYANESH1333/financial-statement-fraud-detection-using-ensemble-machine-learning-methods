#!/usr/bin/env python3
"""
Indian SEBI Financial Statement Fraud Detection System
Specialized for Indian stock market companies and SEBI regulatory requirements
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

class SEBIFraudDetector:
    def __init__(self):
        """Initialize the SEBI fraud detector with Indian financial context"""
        # Load sentiment dictionaries
        self.pos_words = pd.read_csv('positive.csv').iloc[:,0].str.lower().tolist()
        self.neg_words = pd.read_csv('negative.csv').iloc[:,0].str.lower().tolist()
        self.unc_words = pd.read_csv('uncertainty.csv').iloc[:,0].str.lower().tolist()
        self.lit_words = pd.read_csv('litigious.csv').iloc[:,0].str.lower().tolist()
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Indian financial and regulatory specific keywords
        self.indian_financial_terms = {
            'regulatory': [
                'sebi', 'rbi', 'mca', 'nse', 'bse', 'depositories', 'clearing corporation',
                'stock exchange', 'regulatory compliance', 'listing agreement', 'takeover code',
                'insider trading', 'prohibition of fraudulent', 'unfair trade practices',
                'disclosure requirements', 'corporate governance', 'board of directors',
                'independent directors', 'audit committee', 'nomination committee',
                'remuneration committee', 'risk management committee'
            ],
            'financial_metrics': [
                'revenue', 'profit after tax', 'pat', 'ebitda', 'operating profit',
                'net profit', 'total income', 'total expenses', 'operating expenses',
                'finance costs', 'depreciation', 'amortization', 'impairment',
                'provisions', 'contingent liabilities', 'deferred tax', 'current tax',
                'earnings per share', 'eps', 'book value', 'return on equity', 'roe',
                'return on capital employed', 'roce', 'debt to equity', 'current ratio',
                'quick ratio', 'inventory turnover', 'receivables turnover'
            ],
            'indian_specific': [
                'gst', 'goods and services tax', 'tds', 'tax deducted at source',
                'advance tax', 'income tax', 'corporate tax', 'minimum alternate tax',
                'dividend distribution tax', 'fpi', 'foreign portfolio investors',
                'fdi', 'foreign direct investment', 'promoter holding', 'public shareholding',
                'qualified institutional buyers', 'qib', 'retail investors',
                'mutual funds', 'insurance companies', 'pension funds', 'provident funds'
            ],
            'fraud_indicators': [
                'satyam', 'enron', 'worldcom', 'ponzi scheme', 'pyramid scheme',
                'round tripping', 'window dressing', 'creative accounting',
                'off balance sheet', 'special purpose vehicle', 'spv', 'related party',
                'benami', 'shell company', 'bogus transactions', 'fake invoices',
                'over invoicing', 'under invoicing', 'money laundering', 'hawala',
                'black money', 'unaccounted income', 'benami properties',
                'regulatory investigation', 'sfio', 'serious fraud investigation office',
                'enforcement directorate', 'ed', 'income tax raid', 'search and seizure'
            ]
        }
        
        # SEBI specific red flags
        self.sebi_red_flags = [
            'non-compliance', 'violation', 'breach', 'penalty', 'fine', 'suspension',
            'delisting', 'trading suspension', 'investigation', 'inquiry', 'show cause',
            'adjudication', 'settlement', 'consent order', 'disgorgement', 'cease and desist',
            'restraining order', 'interim order', 'final order', 'appeal', 'review',
            'writ petition', 'high court', 'supreme court', 'securities appellate tribunal',
            'sat', 'sebi board', 'whole time member', 'executive director'
        ]
    
    def preprocess_text(self, text):
        """Preprocess text for analysis with Indian financial context"""
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
    
    def extract_indian_financial_features(self, text):
        """Extract Indian financial and regulatory specific features"""
        features = {}
        
        for category, terms in self.indian_financial_terms.items():
            features[f'{category}_count'] = self.count_words_in_text(text, terms)
        
        # SEBI red flags
        features['sebi_red_flags'] = self.count_words_in_text(text, self.sebi_red_flags)
        
        # Indian regulatory compliance indicators
        compliance_indicators = [
            'compliance', 'regulatory', 'sebi', 'rbi', 'mca', 'listing agreement',
            'corporate governance', 'disclosure', 'transparency', 'audit', 'internal audit',
            'statutory audit', 'secretarial audit', 'cost audit', 'concurrent audit'
        ]
        features['compliance_indicators'] = self.count_words_in_text(text, compliance_indicators)
        
        # Financial performance indicators
        performance_indicators = [
            'growth', 'increase', 'improvement', 'positive', 'strong', 'robust',
            'decline', 'decrease', 'negative', 'weak', 'poor', 'loss', 'profit'
        ]
        features['performance_indicators'] = self.count_words_in_text(text, performance_indicators)
        
        return features
    
    def predict_fraud(self, text):
        """Predict fraud for Indian SEBI format financial statements"""
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
            
            # Extract Indian financial features
            indian_features = self.extract_indian_financial_features(text)
            
            # Calculate comprehensive fraud score
            fraud_score = 0
            
            # Basic sentiment scoring
            if neg_count > pos_count:
                fraud_score += 0.15
            if unc_count > 5:
                fraud_score += 0.20
            if lit_count > 3:
                fraud_score += 0.25
            if vader_scores['compound'] < -0.2:
                fraud_score += 0.15
            if abs(polarity_score) > 0.3:
                fraud_score += 0.10
            
            # Indian regulatory specific scoring
            if indian_features['sebi_red_flags'] > 2:
                fraud_score += 0.30  # High weight for SEBI violations
            if indian_features['fraud_indicators_count'] > 1:
                fraud_score += 0.25  # High weight for fraud indicators
            if indian_features['regulatory_count'] > 5:
                fraud_score += 0.10  # Moderate weight for regulatory mentions
            if indian_features['compliance_indicators'] < 2:
                fraud_score += 0.15  # Penalty for lack of compliance mentions
            
            # Financial performance scoring
            if indian_features['performance_indicators'] > 0:
                # Check if negative performance indicators outweigh positive ones
                negative_performance = self.count_words_in_text(text, ['decline', 'decrease', 'negative', 'weak', 'poor', 'loss'])
                positive_performance = self.count_words_in_text(text, ['growth', 'increase', 'improvement', 'positive', 'strong', 'robust', 'profit'])
                if negative_performance > positive_performance:
                    fraud_score += 0.20
            
            # Normalize fraud score
            fraud_score = min(fraud_score, 1.0)
            
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
                'fraud_score': fraud_score,
                'sebi_red_flags': indian_features['sebi_red_flags'],
                'fraud_indicators': indian_features['fraud_indicators_count'],
                'regulatory_mentions': indian_features['regulatory_count'],
                'compliance_indicators': indian_features['compliance_indicators'],
                'indian_financial_terms': indian_features['indian_specific_count']
            }
            
        except Exception as e:
            raise Exception(f"Error in prediction: {str(e)}")

def test_sebi_fraud_detection():
    """Test the SEBI fraud detection system with Indian financial statements"""
    
    detector = SEBIFraudDetector()
    
    # Sample Indian SEBI format financial statements
    test_statements = {
        "High Risk - SEBI Violations": """
        The company has received a show cause notice from SEBI alleging violations of the SEBI (Prohibition of Fraudulent and Unfair Trade Practices) Regulations, 2003. The regulator has initiated investigation into alleged insider trading activities and manipulation of share prices. The company faces potential penalties including disgorgement of profits and trading suspension. The board of directors is considering legal options including filing an appeal with the Securities Appellate Tribunal.
        """,
        
        "High Risk - Financial Irregularities": """
        During the audit period, several irregularities were detected including round tripping transactions, related party transactions not disclosed as per SEBI regulations, and off-balance sheet arrangements through special purpose vehicles. The statutory auditors have qualified their opinion citing material misstatements. The company has received notices from the Serious Fraud Investigation Office (SFIO) and Enforcement Directorate for investigation of alleged money laundering activities.
        """,
        
        "Medium Risk - Regulatory Concerns": """
        The company has been non-compliant with certain SEBI listing agreement requirements including delayed disclosure of material events and inadequate corporate governance practices. While we have taken corrective measures and appointed additional independent directors, there remains uncertainty regarding potential regulatory action. The audit committee has identified several areas for improvement in internal controls and risk management.
        """,
        
        "Low Risk - Compliant Performance": """
        The company has maintained strong compliance with all SEBI regulations and corporate governance requirements. Our quarterly results show robust performance with revenue growth of 15% and improved profit margins. The board of directors, including independent directors, have provided strong oversight. All material events have been promptly disclosed to stock exchanges as per SEBI requirements. The statutory auditors have issued an unqualified opinion.
        """,
        
        "Low Risk - Transparent Disclosure": """
        The company's financial performance reflects both challenges and opportunities in the Indian market. While we faced some regulatory changes including new GST requirements, we successfully implemented compliance measures. Revenue grew 8% with improved operational efficiency. The board has strengthened corporate governance practices and enhanced disclosure transparency as per SEBI guidelines. We remain committed to maintaining high standards of regulatory compliance.
        """
    }
    
    print("🔍 INDIAN SEBI FINANCIAL STATEMENT FRAUD DETECTION")
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
                print("   ⚠️  This financial statement shows signs of potential fraud or regulatory violations.")
            else:
                print("✅ NO FRAUD DETECTED")
                print(f"   Confidence: {confidence*100:.2f}%")
                print("   ✅ This financial statement appears to be compliant and legitimate.")
            
            print("\n📊 FEATURE ANALYSIS:")
            print(f"   • Positive words: {features['positive_words']}")
            print(f"   • Negative words: {features['negative_words']}")
            print(f"   • Uncertainty words: {features['uncertainty_words']}")
            print(f"   • Litigious words: {features['litigious_words']}")
            print(f"   • SEBI red flags: {features['sebi_red_flags']}")
            print(f"   • Fraud indicators: {features['fraud_indicators']}")
            print(f"   • Regulatory mentions: {features['regulatory_mentions']}")
            print(f"   • Compliance indicators: {features['compliance_indicators']}")
            print(f"   • Indian financial terms: {features['indian_financial_terms']}")
            print(f"   • Sentiment polarity: {features['sentiment_polarity']:.3f}")
            print(f"   • VADER compound score: {features['vader_compound']:.3f}")
            print(f"   • Overall fraud score: {features['fraud_score']:.3f}")
            
            print("\n💡 INTERPRETATION:")
            if features['sebi_red_flags'] > 2:
                print("   • Multiple SEBI regulatory violations detected")
            if features['fraud_indicators'] > 1:
                print("   • Fraud-related keywords identified")
            if features['negative_words'] > features['positive_words']:
                print("   • Higher negative sentiment detected")
            if features['uncertainty_words'] > 5:
                print("   • High uncertainty language detected")
            if features['compliance_indicators'] < 2:
                print("   • Limited compliance and governance mentions")
            if features['fraud_score'] > 0.5:
                print("   • Multiple fraud indicators present")
            
        except Exception as e:
            print(f"❌ Error analyzing statement: {str(e)}")
        
        print("\n" + "=" * 70)
        print()

def interactive_sebi_test():
    """Interactive testing mode for SEBI statements"""
    detector = SEBIFraudDetector()
    
    print("🔍 INTERACTIVE SEBI FRAUD DETECTION TESTING")
    print("=" * 60)
    print("Enter Indian financial statements to analyze (type 'quit' to exit)")
    print("Supported formats: Annual Reports, Quarterly Results, Board Reports,")
    print("Regulatory Disclosures, Corporate Governance Reports")
    print()
    
    while True:
        print("-" * 60)
        statement = input("Enter financial statement text: ").strip()
        
        if statement.lower() in ['quit', 'exit', 'q']:
            print("👋 Thank you for testing the SEBI fraud detection system!")
            break
        
        if not statement:
            print("❌ Please enter some text to analyze.")
            continue
        
        try:
            prediction, confidence, features = detector.predict_fraud(statement)
            
            print("\n🔍 ANALYSIS RESULTS:")
            if prediction == 1:
                print("🚨 FRAUD/VIOLATION DETECTED!")
                print(f"   Confidence: {confidence*100:.2f}%")
                print("   ⚠️  Potential SEBI violations or fraud indicators found")
            else:
                print("✅ NO FRAUD DETECTED")
                print(f"   Confidence: {confidence*100:.2f}%")
                print("   ✅ Appears compliant with SEBI regulations")
            
            print(f"\n📊 Quick Analysis:")
            print(f"   • SEBI red flags: {features['sebi_red_flags']}")
            print(f"   • Fraud indicators: {features['fraud_indicators']}")
            print(f"   • Regulatory mentions: {features['regulatory_mentions']}")
            print(f"   • Compliance indicators: {features['compliance_indicators']}")
            print(f"   • Overall fraud score: {features['fraud_score']:.3f}")
            
            # Provide specific recommendations
            print(f"\n💡 RECOMMENDATIONS:")
            if features['sebi_red_flags'] > 0:
                print("   • Review SEBI compliance and regulatory requirements")
            if features['fraud_indicators'] > 0:
                print("   • Conduct detailed forensic audit and investigation")
            if features['compliance_indicators'] < 3:
                print("   • Strengthen corporate governance and compliance framework")
            if features['fraud_score'] > 0.3:
                print("   • Consider external audit and legal consultation")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print()

if __name__ == "__main__":
    print("Indian SEBI Financial Statement Fraud Detection System")
    print("=" * 60)
    print("Choose testing mode:")
    print("1. Run predefined SEBI test cases")
    print("2. Interactive testing mode")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        test_sebi_fraud_detection()
    elif choice == "2":
        interactive_sebi_test()
    else:
        print("Invalid choice. Running predefined SEBI tests...")
        test_sebi_fraud_detection()
