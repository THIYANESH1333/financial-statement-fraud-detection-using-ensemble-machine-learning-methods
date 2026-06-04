import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set up the figure with high DPI for quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, ax = plt.subplots(1, 1, figsize=(18, 14))
ax.set_xlim(0, 18)
ax.set_ylim(0, 14)
ax.axis('off')

# Define professional colors
primary_color = '#1f77b4'      # Blue
secondary_color = '#ff7f0e'    # Orange
accent_color = '#2ca02c'       # Green
success_color = '#d62728'      # Red
light_blue = '#e6f3ff'
light_orange = '#fff2e6'
light_green = '#e6ffe6'
light_red = '#ffe6e6'

# Function to create professional rounded rectangle boxes
def create_box(x, y, width, height, text, color, text_color='black', fontsize=10):
    box = FancyBboxPatch((x, y), width, height, 
                         boxstyle="round,pad=0.15", 
                         facecolor=color, 
                         edgecolor='#333333', 
                         linewidth=2)
    ax.add_patch(box)
    
    # Add text with proper wrapping
    lines = text.split('\n')
    for i, line in enumerate(lines):
        y_pos = y + height/2 + (len(lines)-1)*0.15 - i*0.3
        ax.text(x + width/2, y_pos, line, 
                ha='center', va='center', fontsize=fontsize, 
                fontweight='bold', color=text_color)
    return box

# Function to create professional arrows
def create_arrow(start_x, start_y, end_x, end_y, color='#333333', width=2.5):
    arrow = ConnectionPatch((start_x, start_y), (end_x, end_y), 
                           "data", "data", 
                           arrowstyle="->", 
                           shrinkA=8, shrinkB=8, 
                           mutation_scale=25, 
                           fc=color, ec=color, linewidth=width)
    ax.add_patch(arrow)

# Title
ax.text(9, 13.5, 'ENHANCED FINANCIAL STATEMENT FRAUD DETECTION SYSTEM', 
        ha='center', va='center', fontsize=24, fontweight='bold', color=primary_color)

# Phase 1: Data Loading & Preprocessing
ax.text(9, 12.8, 'PHASE 1: DATA LOADING & PREPROCESSING', 
        ha='center', va='center', fontsize=16, fontweight='bold', color=secondary_color)

# Data Loading
create_box(1, 11.5, 3, 1, 'Load Final_Dataset.csv\n(Text + Numerical)', light_blue, primary_color, 10)

# Data Preprocessing
create_box(5, 11.5, 3, 1, 'Advanced Financial\nText Preprocessing', light_orange, secondary_color, 10)

# Feature Engineering
create_box(9, 11.5, 3, 1, 'Feature Engineering\n(Sentiment + Red-Flags)', light_green, accent_color, 10)

# BERT Processing
create_box(13, 11.5, 3, 1, 'BERT Embeddings\n(768 dimensions)', light_blue, primary_color, 10)

# Phase 2: Model Training
ax.text(9, 10.8, 'PHASE 2: MODEL TRAINING & ENSEMBLE', 
        ha='center', va='center', fontsize=16, fontweight='bold', color=secondary_color)

# Feature Sets
create_box(1, 9.5, 2.5, 0.8, 'TF-IDF\nFeatures', light_blue, primary_color, 9)
create_box(4, 9.5, 2.5, 0.8, 'Sentiment\nFeatures', light_orange, secondary_color, 9)
create_box(7, 9.5, 2.5, 0.8, 'Numerical\nFeatures', light_green, accent_color, 9)
create_box(10, 9.5, 2.5, 0.8, 'BERT\nFeatures', light_blue, primary_color, 9)
create_box(13.5, 9.5, 2.5, 0.8, 'Combined\nFeature Set', light_orange, secondary_color, 9)

# Model Training Boxes
create_box(0.5, 7.5, 3, 1, 'Logistic Regression\n(class_weight=balanced)', light_blue, primary_color, 9)
create_box(4.5, 7.5, 3, 1, 'SVM\n(class_weight=balanced)', light_orange, secondary_color, 9)
create_box(8.5, 7.5, 3, 1, 'XGBoost\n(scale_pos_weight)', light_green, accent_color, 9)
create_box(12.5, 7.5, 3, 1, 'Deep Learning\n(BCE + Focal Loss)', light_blue, primary_color, 9)

# Voting Classifier
create_box(8.5, 6, 3, 1, 'Voting Classifier\n(Ensemble)', success_color, 'white', 10)

# Phase 3: Model Evaluation
ax.text(9, 5.3, 'PHASE 3: MODEL EVALUATION & SELECTION', 
        ha='center', va='center', fontsize=16, fontweight='bold', color=secondary_color)

# Evaluation Metrics
create_box(1, 3.5, 2.5, 0.8, 'Accuracy\n95.00%', light_blue, primary_color, 9)
create_box(4, 3.5, 2.5, 0.8, 'Precision\n80.00%', light_orange, secondary_color, 9)
create_box(7, 3.5, 2.5, 0.8, 'Recall\n80.00%', light_green, accent_color, 9)
create_box(10, 3.5, 2.5, 0.8, 'F1-Score\n80.00%', light_blue, primary_color, 9)
create_box(13, 3.5, 2.5, 0.8, 'ROC AUC\n98.67%', light_orange, secondary_color, 9)

# Phase 4: Deployment
ax.text(9, 2.8, 'PHASE 4: DEPLOYMENT & INTERACTIVE INTERFACE', 
        ha='center', va='center', fontsize=16, fontweight='bold', color=secondary_color)

# Best Model Selection
create_box(8.5, 1.5, 3, 1, 'Best Model:\nXGBoost (95% Acc)', success_color, 'white', 10)

# Interactive Interface
create_box(8.5, 0, 3, 1, 'Interactive Fraud\nDetection Interface', light_green, accent_color, 10)

# Create professional arrows for Phase 1
create_arrow(2.5, 11.5, 5, 12, primary_color)
create_arrow(6.5, 11.5, 9, 12, secondary_color)
create_arrow(10.5, 11.5, 13, 12, accent_color)

# Create arrows from preprocessing to features
create_arrow(5, 11.5, 2.25, 9.5, secondary_color)
create_arrow(5, 11.5, 5.25, 9.5, secondary_color)
create_arrow(5, 11.5, 8.25, 9.5, secondary_color)
create_arrow(13, 11.5, 11.75, 9.5, primary_color)

# Create arrows to combined features
create_arrow(2.25, 9.5, 13.5, 9.9, primary_color)
create_arrow(5.25, 9.5, 13.5, 9.9, secondary_color)
create_arrow(8.25, 9.5, 13.5, 9.9, accent_color)
create_arrow(11.75, 9.5, 13.5, 9.9, primary_color)

# Create arrows from combined features to models
create_arrow(14.75, 9.5, 2, 7.5, light_orange)
create_arrow(14.75, 9.5, 6, 7.5, light_orange)
create_arrow(14.75, 9.5, 10, 7.5, light_orange)
create_arrow(14.75, 9.5, 14, 7.5, light_orange)

# Create arrows from models to voting classifier
create_arrow(2, 7.5, 8.5, 6.5, primary_color)
create_arrow(6, 7.5, 8.5, 6.5, secondary_color)
create_arrow(10, 7.5, 8.5, 6.5, accent_color)
create_arrow(14, 7.5, 8.5, 6.5, primary_color)

# Create arrows from voting classifier to evaluation
create_arrow(8.5, 6, 2.25, 3.5, success_color)
create_arrow(8.5, 6, 5.25, 3.5, success_color)
create_arrow(8.5, 6, 8.25, 3.5, success_color)
create_arrow(8.5, 6, 11.25, 3.5, success_color)
create_arrow(8.5, 6, 14.25, 3.5, success_color)

# Create arrows from evaluation to best model
create_arrow(2.25, 3.5, 8.5, 2.3, light_blue)
create_arrow(5.25, 3.5, 8.5, 2.3, light_orange)
create_arrow(8.25, 3.5, 8.5, 2.3, light_green)
create_arrow(11.25, 3.5, 8.5, 2.3, light_blue)
create_arrow(14.25, 3.5, 8.5, 2.3, light_orange)

# Create arrow from best model to interface
create_arrow(8.5, 1.5, 8.5, 1, success_color)

# Add key features box
key_features_box = FancyBboxPatch((0.5, 0.5), 6, 1.5, 
                                  boxstyle="round,pad=0.15", 
                                  facecolor=light_green, 
                                  edgecolor=accent_color, 
                                  linewidth=2)
ax.add_patch(key_features_box)

ax.text(3.5, 1.25, 'KEY FEATURES:', ha='center', va='center', 
        fontsize=14, fontweight='bold', color=accent_color)
ax.text(3.5, 0.9, '• Financial Red-Flag Detection', ha='center', va='center', 
        fontsize=11, color='black')
ax.text(3.5, 0.6, '• Sentiment Analysis (VADER + TextBlob)', ha='center', va='center', 
        fontsize=11, color='black')
ax.text(3.5, 0.3, '• Advanced Text Preprocessing', ha='center', va='center', 
        fontsize=11, color='black')

# Add performance comparison box
perf_box = FancyBboxPatch((11.5, 0.5), 6, 1.5, 
                          boxstyle="round,pad=0.15", 
                          facecolor=light_orange, 
                          edgecolor=secondary_color, 
                          linewidth=2)
ax.add_patch(perf_box)

ax.text(14.5, 1.25, 'PERFORMANCE COMPARISON:', ha='center', va='center', 
        fontsize=14, fontweight='bold', color=secondary_color)
ax.text(14.5, 0.9, '• Existing Model: ~75-80% Accuracy', ha='center', va='center', 
        fontsize=11, color='black')
ax.text(14.5, 0.6, '• Proposed Model: 95.00% Accuracy', ha='center', va='center', 
        fontsize=11, color='black')
ax.text(14.5, 0.3, '• Improvement: +17.5% Accuracy', ha='center', va='center', 
        fontsize=11, color='black')

# Add professional legend
legend_elements = [
    patches.Patch(color=light_blue, label='Data Processing & ML Models'),
    patches.Patch(color=light_orange, label='Feature Engineering & Evaluation'),
    patches.Patch(color=light_green, label='Advanced Features & Deployment'),
    patches.Patch(color=success_color, label='Best Model & Ensemble')
]

ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
          fontsize=12, frameon=True, fancybox=True, shadow=True)

# Add subtitle with key achievements
ax.text(9, 13.2, 'Achieving 95% Accuracy with Advanced ML Pipeline', 
        ha='center', va='center', fontsize=14, fontstyle='italic', color='#666666')

# Save the flowchart with high quality
plt.tight_layout()
plt.savefig('trained_model_flowchart.png', 
            bbox_inches='tight', dpi=300, facecolor='white', 
            edgecolor='none', pad_inches=0.2)
plt.show()

print("✅ Professional flowchart saved as 'trained_model_flowchart.png'")
print("📊 The flowchart shows the complete pipeline with clean design and proper alignment")
print("🎨 Features: Professional colors, proper spacing, clear arrows, and organized phases")
