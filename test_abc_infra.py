#!/usr/bin/env python3
"""
Test script to analyze the ABC Infra Ltd statement for fraud detection
"""

from advanced_fraud_detector import AdvancedFraudDetector

def test_abc_infra_statement():
    """Test the ABC Infra Ltd statement"""
    
    detector = AdvancedFraudDetector()
    
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
        
        print("\n🎯 EXPECTED FRAUD INDICATORS IN THIS STATEMENT:")
        print("   • Profit without cash (₹900 crore profit vs ₹120 crore cash flow)")
        print("   • Heavy related-party receivables (34% of ₹1,050 crore)")
        print("   • Auditor concerns over revenue recognition")
        print("   • Overdue receivables")
        print("   • Statutory dues delays (CARO 2020)")
        
    except Exception as e:
        print(f"❌ Error analyzing statement: {str(e)}")

if __name__ == "__main__":
    test_abc_infra_statement()
