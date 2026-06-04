import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def extract_numerical_features(text):
    """
    Extract numerical features from text without sentiment analysis
    """
    if pd.isna(text):
        text = ""
    
    text = str(text).lower()
    
    # Basic text statistics
    features = {}
    
    # 1. Text length features
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    features['sentence_count'] = len([s for s in text.split('.') if s.strip()])
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)
    
    # 2. Character-based features
    features['char_count'] = len(text)
    features['digit_count'] = sum(c.isdigit() for c in text)
    features['uppercase_count'] = sum(c.isupper() for c in text)
    features['lowercase_count'] = sum(c.islower() for c in text)
    features['punctuation_count'] = sum(c in string.punctuation for c in text)
    features['whitespace_count'] = sum(c.isspace() for c in text)
    
    # 3. Ratio features
    features['digit_ratio'] = features['digit_count'] / max(features['char_count'], 1)
    features['uppercase_ratio'] = features['uppercase_count'] / max(features['char_count'], 1)
    features['punctuation_ratio'] = features['punctuation_count'] / max(features['char_count'], 1)
    features['whitespace_ratio'] = features['whitespace_count'] / max(features['char_count'], 1)
    
    # 4. Financial-specific numerical features
    # Count financial numbers and amounts
    currency_patterns = [
        r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)',  # Indian Rupee
        r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)',  # Dollar
        r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:crore|lakh|million|billion)',  # Amount with words
        r'(\d+(?:,\d+)*(?:\.\d+)?)\s*%',  # Percentage
    ]
    
    financial_numbers = 0
    for pattern in currency_patterns:
        financial_numbers += len(re.findall(pattern, text))
    
    features['financial_numbers_count'] = financial_numbers
    features['financial_numbers_ratio'] = financial_numbers / max(features['word_count'], 1)
    
    # 5. Financial keywords (without sentiment)
    financial_keywords = [
        'revenue', 'profit', 'loss', 'ebitda', 'cash', 'flow', 'receivables', 'payables',
        'inventory', 'assets', 'liabilities', 'equity', 'auditor', 'audit', 'caro',
        'statutory', 'compliance', 'related', 'party', 'transaction', 'fraud', 'irregularity',
        'balance', 'sheet', 'income', 'statement', 'financial', 'report', 'quarter',
        'annual', 'consolidated', 'subsidiary', 'investment', 'depreciation', 'amortization'
    ]
    
    financial_keyword_count = sum(1 for keyword in financial_keywords if keyword in text)
    features['financial_keywords_count'] = financial_keyword_count
    features['financial_keywords_ratio'] = financial_keyword_count / max(features['word_count'], 1)
    
    # 6. Red-flag indicators (numerical counts only)
    red_flag_phrases = [
        'operating cash flow', 'cash flow', 'gap versus net profit', 'profit without cash',
        'related party', 'related parties', 'related party transaction', 'rpt',
        'auditor', 'revenue recognition', 'emphasis of matter', 'qualified opinion', 'overdue receivables',
        'caro', 'statutory dues', 'statutory delay', 'non-compliance', 'gst dues', 'income tax dues',
        'trade receivables', 'overdue receivables', 'receivables increased', 'days sales outstanding'
    ]
    
    red_flag_count = sum(1 for phrase in red_flag_phrases if phrase in text)
    features['red_flag_count'] = red_flag_count
    features['red_flag_ratio'] = red_flag_count / max(features['word_count'], 1)
    
    # 7. Complexity features
    features['unique_words'] = len(set(text.split()))
    features['vocabulary_richness'] = features['unique_words'] / max(features['word_count'], 1)
    
    # 8. Readability features (simplified)
    # Average syllables per word (approximation)
    vowels = 'aeiou'
    syllable_count = sum(1 for char in text if char in vowels)
    features['syllable_count'] = syllable_count
    features['avg_syllables_per_word'] = syllable_count / max(features['word_count'], 1)
    
    # 9. Structural features
    features['paragraph_count'] = len([p for p in text.split('\n') if p.strip()])
    features['line_count'] = text.count('\n')
    
    # 10. Numerical patterns
    # Count specific number patterns
    features['decimal_count'] = len(re.findall(r'\d+\.\d+', text))
    features['comma_number_count'] = len(re.findall(r'\d{1,3}(?:,\d{3})+', text))
    features['percentage_count'] = len(re.findall(r'\d+(?:\.\d+)?%', text))
    
    return features

def create_numerical_dataset(df):
    """
    Create numerical features dataset from text data
    """
    print("Extracting numerical features from text...")
    
    # Extract features for each text
    feature_list = []
    for idx, text in enumerate(df['Fillings']):
        if idx % 50 == 0:
            print(f"Processing text {idx+1}/{len(df)}")
        
        features = extract_numerical_features(text)
        feature_list.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(feature_list)
    
    # Handle any NaN values
    features_df = features_df.fillna(0)
    
    print(f"Created {features_df.shape[1]} numerical features")
    print("Feature names:", features_df.columns.tolist())
    
    return features_df

def optimize_xgboost(X, y):
    """
    Optimize XGBoost hyperparameters for maximum accuracy
    """
    print("Optimizing XGBoost hyperparameters...")
    
    # Define parameter grid for optimization
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 0.5],
        'reg_lambda': [0, 0.01, 0.1, 0.5, 1.0],
        'min_child_weight': [1, 2, 3, 4, 5]
    }
    
    # Create base XGBoost model
    xgb = XGBClassifier(
        random_state=42,
        eval_metric='logloss'
    )
    
    # Use GridSearchCV for optimization
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, model_name="XGBoost"):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    print(f"ROC AUC: {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

def plot_feature_importance(model, feature_names, top_n=15):
    """
    Plot feature importance
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Feature Importances (XGBoost)')
    plt.barh(range(top_n), importance[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_numerical.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run the numerical-only XGBoost model
    """
    print("="*60)
    print("XGBoost Fraud Detection - Numerical Features Only")
    print("="*60)
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv('Final_Dataset.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud distribution: {df['Fraud'].value_counts()}")
    
    # Create numerical features
    X_numerical = create_numerical_dataset(df)
    
    # Prepare target variable
    y = df['Fraud'].map({'no': 0, 'yes': 1})
    
    # Handle class imbalance
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    
    print(f"Class distribution - Fraud: {pos_count}, No Fraud: {neg_count}")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_numerical, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    # Optimize XGBoost
    best_model = optimize_xgboost(X_train_scaled, y_train)
    
    # Add scale_pos_weight to handle class imbalance
    best_model.set_params(scale_pos_weight=scale_pos_weight)
    best_model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    results = evaluate_model(best_model, X_test_scaled, y_test)
    
    # Plot feature importance
    plot_feature_importance(best_model, X_numerical.columns)
    
    # Cross-validation score
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"✅ Final Accuracy: {results['accuracy']*100:.2f}%")
    
    if results['accuracy'] >= 0.90:
        print("🎉 SUCCESS: Achieved accuracy above 90%!")
    else:
        print("⚠️  Accuracy below 90%. Consider further optimization.")
    
    print(f"📊 Precision: {results['precision']*100:.2f}%")
    print(f"📊 Recall: {results['recall']*100:.2f}%")
    print(f"📊 F1-Score: {results['f1']*100:.2f}%")
    print(f"📊 ROC AUC: {results['roc_auc']*100:.2f}%")
    
    # Save model and scaler
    import joblib
    joblib.dump(best_model, 'xgboost_numerical_model.pkl')
    joblib.dump(scaler, 'numerical_scaler.pkl')
    print("\n💾 Model and scaler saved successfully!")
    
    return best_model, scaler, results

if __name__ == "__main__":
    model, scaler, results = main()
