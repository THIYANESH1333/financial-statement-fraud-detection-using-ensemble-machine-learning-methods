import webbrowser
import os

# Get the current directory
current_dir = os.getcwd()
svg_file = os.path.join(current_dir, 'financial_analysis_figure.svg')

print("Figure 1.1.1 - Financial Statement Analysis")
print("=" * 50)
print()
print("The diagram shows:")
print("• Revenue Recognition Phase (Purple) - Top Left")
print("  - Sales Revenue, Service Revenue, Interest Revenue, Accrual Accounting")
print()
print("• Expense Matching Phase (Orange) - Top Right") 
print("  - Cost of Goods Sold, Operating Expenses, Depreciation, Matching Principle")
print()
print("• Central Financial Statement Analysis (Blue) - Center")
print("  - Main analysis hub connecting all components")
print()
print("• Financial Statement Components (Gray) - Left")
print("  - Income Statement, Balance Sheet, Cash Flow Statement, Notes to Accounts")
print()
print("• Analysis Methods (Gray) - Right")
print("  - Ratio Analysis, Trend Analysis, Machine Learning, Statistical Models")
print()
print("• Accounting Cycle (Dashed Circle) - Center Bottom")
print("  - Shows the continuous nature of financial analysis")
print()
print("• Fraud Detection Categories (Red) - Bottom")
print("  - Normal | Suspicious | Manipulated | Fraudulent")
print()
print("Opening the SVG file in your default browser...")

# Open the SVG file in the default web browser
webbrowser.open(f'file://{svg_file}')
