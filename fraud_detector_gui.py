import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
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
from xgboost import XGBClassifier
import pickle
import os

class FraudDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Financial Statement Fraud Detector")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize models and features
        self.load_models()
        
        # Create GUI elements
        self.create_widgets()
        
    def load_models(self):
        """Load the trained models and feature extractors"""
        try:
            # Load sentiment dictionaries
            self.pos_words = pd.read_csv('positive.csv').iloc[:,0].str.lower().tolist()
            self.neg_words = pd.read_csv('negative.csv').iloc[:,0].str.lower().tolist()
            self.unc_words = pd.read_csv('uncertainty.csv').iloc[:,0].str.lower().tolist()
            self.lit_words = pd.read_csv('litigious.csv').iloc[:,0].str.lower().tolist()
            
            # Initialize NLTK
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            self.analyzer = SentimentIntensityAnalyzer()
            
            # Initialize feature extractors (these would normally be loaded from saved models)
            self.tfidf = TfidfVectorizer(max_features=3000)
            self.scaler = StandardScaler()
            
            # For demo purposes, we'll create a simple model
            self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
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
            
            # Create feature vector
            features = np.array([[
                pos_count, neg_count, unc_count, lit_count,
                vader_scores['compound'], vader_scores['neg'], 
                vader_scores['neu'], vader_scores['pos'],
                polarity_score, subjectivity_score
            ]])
            
            # For demo purposes, use a simple rule-based prediction
            # In a real scenario, this would use the trained model
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
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Title
        title_label = tk.Label(self.root, text="Financial Statement Fraud Detector", 
                              font=("Arial", 16, "bold"), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=10)
        
        # Subtitle
        subtitle_label = tk.Label(self.root, text="Enter financial statement text to analyze for potential fraud", 
                                 font=("Arial", 10), bg='#f0f0f0', fg='#7f8c8d')
        subtitle_label.pack(pady=5)
        
        # Input frame
        input_frame = tk.Frame(self.root, bg='#f0f0f0')
        input_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        # Input label
        input_label = tk.Label(input_frame, text="Financial Statement Text:", 
                              font=("Arial", 12, "bold"), bg='#f0f0f0')
        input_label.pack(anchor='w')
        
        # Text input area
        self.text_input = scrolledtext.ScrolledText(input_frame, height=8, width=80, 
                                                   font=("Arial", 10), wrap=tk.WORD)
        self.text_input.pack(pady=5, fill='both', expand=True)
        
        # Buttons frame
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        # Analyze button
        analyze_btn = tk.Button(button_frame, text="🔍 Analyze for Fraud", 
                               command=self.analyze_text, 
                               font=("Arial", 12, "bold"),
                               bg='#3498db', fg='white', 
                               relief='raised', padx=20, pady=5)
        analyze_btn.pack(side='left', padx=5)
        
        # Clear button
        clear_btn = tk.Button(button_frame, text="🗑️ Clear", 
                             command=self.clear_text, 
                             font=("Arial", 12),
                             bg='#e74c3c', fg='white', 
                             relief='raised', padx=20, pady=5)
        clear_btn.pack(side='left', padx=5)
        
        # Results frame
        results_frame = tk.Frame(self.root, bg='#f0f0f0')
        results_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        # Results label
        results_label = tk.Label(results_frame, text="Analysis Results:", 
                                font=("Arial", 12, "bold"), bg='#f0f0f0')
        results_label.pack(anchor='w')
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, height=12, width=80, 
                                                     font=("Arial", 10), wrap=tk.WORD,
                                                     state='disabled')
        self.results_text.pack(pady=5, fill='both', expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to analyze")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             relief='sunken', anchor='w', bg='#ecf0f1')
        status_bar.pack(side='bottom', fill='x')
    
    def analyze_text(self):
        """Analyze the input text for fraud"""
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to analyze.")
            return
        
        try:
            self.status_var.set("Analyzing...")
            self.root.update()
            
            # Make prediction
            prediction, confidence, features = self.predict_fraud(text)
            
            # Display results
            self.display_results(prediction, confidence, features)
            
            self.status_var.set("Analysis complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.status_var.set("Analysis failed")
    
    def display_results(self, prediction, confidence, features):
        """Display the analysis results"""
        self.results_text.config(state='normal')
        self.results_text.delete("1.0", tk.END)
        
        # Create results text
        results = "🔍 FRAUD DETECTION ANALYSIS RESULTS\n"
        results += "=" * 50 + "\n\n"
        
        if prediction == 1:
            results += "🚨 FRAUD DETECTED!\n"
            results += f"   Confidence: {confidence*100:.2f}%\n"
            results += "   ⚠️  This financial statement shows signs of potential fraud.\n\n"
        else:
            results += "✅ NO FRAUD DETECTED\n"
            results += f"   Confidence: {confidence*100:.2f}%\n"
            results += "   ✅ This financial statement appears to be legitimate.\n\n"
        
        results += "📊 FEATURE ANALYSIS:\n"
        results += f"   • Positive words: {features['positive_words']}\n"
        results += f"   • Negative words: {features['negative_words']}\n"
        results += f"   • Uncertainty words: {features['uncertainty_words']}\n"
        results += f"   • Litigious words: {features['litigious_words']}\n"
        results += f"   • Sentiment polarity: {features['sentiment_polarity']:.3f}\n"
        results += f"   • Sentiment subjectivity: {features['sentiment_subjectivity']:.3f}\n"
        results += f"   • VADER compound score: {features['vader_compound']:.3f}\n"
        results += f"   • Fraud score: {features['fraud_score']:.3f}\n\n"
        
        results += "💡 INTERPRETATION:\n"
        if features['negative_words'] > features['positive_words']:
            results += "   • Higher negative sentiment detected\n"
        if features['uncertainty_words'] > 5:
            results += "   • High uncertainty language detected\n"
        if features['litigious_words'] > 3:
            results += "   • Litigious language detected\n"
        if abs(features['sentiment_polarity']) > 0.3:
            results += "   • Strong sentiment detected\n"
        if features['fraud_score'] > 0.5:
            results += "   • Multiple fraud indicators present\n"
        
        results += "\n" + "=" * 50 + "\n"
        results += "Note: This is a demonstration system. For real fraud detection,\n"
        results += "please consult with financial experts and use comprehensive\n"
        results += "auditing procedures.\n"
        
        self.results_text.insert("1.0", results)
        self.results_text.config(state='disabled')
    
    def clear_text(self):
        """Clear the input and results text areas"""
        self.text_input.delete("1.0", tk.END)
        self.results_text.config(state='normal')
        self.results_text.delete("1.0", tk.END)
        self.results_text.config(state='disabled')
        self.status_var.set("Ready to analyze")

def main():
    root = tk.Tk()
    app = FraudDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
