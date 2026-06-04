import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up the figure with perfect proportions
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
fig, ax = plt.subplots(1, 1, figsize=(12, 9))
ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.axis('off')

# Professional colors
blue = '#1f77b4'
green = '#2ca02c'
orange = '#ff7f0e'
red = '#d62728'
light_gray = '#f0f0f0'

# Function to create perfectly aligned boxes
def create_box(x, y, width, height, text, color):
    rect = patches.Rectangle((x, y), width, height, linewidth=1.5, 
                           edgecolor='black', facecolor=color, alpha=0.9)
    ax.add_patch(rect)
    
    # Add text with perfect centering
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
            fontsize=9, fontweight='bold')

# Function to create clean arrows
def create_arrow(start_x, start_y, end_x, end_y):
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(facecolor='black', edgecolor='black',
                                shrinkA=0, shrinkB=0, linewidth=1.5,
                                arrowstyle='-|>', mutation_scale=15))

# Title - perfectly centered
ax.text(6, 8.5, 'Financial Statement Fraud Detection System', 
        ha='center', va='center', fontsize=14, fontweight='bold', color=blue)

# Phase 1: Data Processing - perfectly aligned row
create_box(0.5, 7, 2.2, 0.7, 'Load Data\n(Text+Numbers)', light_gray)
create_box(3.0, 7, 2.2, 0.7, 'Clean Text', light_gray)
create_box(5.5, 7, 2.2, 0.7, 'Extract\nFeatures', light_gray)
create_box(8.0, 7, 2.2, 0.7, 'BERT\nEmbeddings', light_gray)

# Phase 2: Model Training - perfectly aligned row
create_box(0.5, 5.5, 2.2, 0.7, 'Logistic\nRegression', blue)
create_box(3.0, 5.5, 2.2, 0.7, 'SVM', blue)
create_box(5.5, 5.5, 2.2, 0.7, 'XGBoost', blue)
create_box(8.0, 5.5, 2.2, 0.7, 'Deep\nLearning', blue)

# Ensemble - perfectly centered
create_box(4.9, 4, 2.2, 0.7, 'Voting\nClassifier', red)

# Results - perfectly aligned row
create_box(0.5, 2.5, 1.8, 0.5, 'Accuracy\n95%', green)
create_box(2.6, 2.5, 1.8, 0.5, 'Precision\n80%', green)
create_box(4.7, 2.5, 1.8, 0.5, 'Recall\n80%', green)
create_box(6.8, 2.5, 1.8, 0.5, 'F1-Score\n80%', green)
create_box(8.9, 2.5, 1.8, 0.5, 'ROC AUC\n98.7%', green)

# Final Model - perfectly centered
create_box(4.9, 1.5, 2.2, 0.7, 'Best Model:\nXGBoost', orange)

# Interface - perfectly centered
create_box(4.9, 0.5, 2.2, 0.7, 'Interactive\nInterface', green)

# Perfect arrows - horizontal flow in Phase 1 (from right edge to left edge)
create_arrow(0.5 + 2.2, 7 + 0.7/2, 3.0, 7 + 0.7/2)
create_arrow(3.0 + 2.2, 7 + 0.7/2, 5.5, 7 + 0.7/2)
create_arrow(5.5 + 2.2, 7 + 0.7/2, 8.0, 7 + 0.7/2)

# Perfect arrows - from Phase 1 to Phase 2 (from center bottom to center top)
create_arrow(0.5 + 2.2/2, 7, 0.5 + 2.2/2, 5.5 + 0.7)
create_arrow(3.0 + 2.2/2, 7, 3.0 + 2.2/2, 5.5 + 0.7)
create_arrow(5.5 + 2.2/2, 7, 5.5 + 2.2/2, 5.5 + 0.7)
create_arrow(8.0 + 2.2/2, 7, 8.0 + 2.2/2, 5.5 + 0.7)

# Perfect arrows - from models to ensemble (from center bottom to center top)
create_arrow(0.5 + 2.2/2, 5.5, 4.9 + 2.2/2, 4 + 0.7)
create_arrow(3.0 + 2.2/2, 5.5, 4.9 + 2.2/2, 4 + 0.7)
create_arrow(5.5 + 2.2/2, 5.5, 4.9 + 2.2/2, 4 + 0.7)
create_arrow(8.0 + 2.2/2, 5.5, 4.9 + 2.2/2, 4 + 0.7)

# Perfect arrows - from ensemble to evaluation (from center bottom to center top)
create_arrow(4.9 + 2.2/2, 4, 0.5 + 1.8/2, 2.5 + 0.5)
create_arrow(4.9 + 2.2/2, 4, 2.6 + 1.8/2, 2.5 + 0.5)
create_arrow(4.9 + 2.2/2, 4, 4.7 + 1.8/2, 2.5 + 0.5)
create_arrow(4.9 + 2.2/2, 4, 6.8 + 1.8/2, 2.5 + 0.5)
create_arrow(4.9 + 2.2/2, 4, 8.9 + 1.8/2, 2.5 + 0.5)

# Perfect arrows - from evaluation to best model (from center bottom to center top)
create_arrow(0.5 + 1.8/2, 2.5, 4.9 + 2.2/2, 1.5 + 0.7)
create_arrow(2.6 + 1.8/2, 2.5, 4.9 + 2.2/2, 1.5 + 0.7)
create_arrow(4.7 + 1.8/2, 2.5, 4.9 + 2.2/2, 1.5 + 0.7)
create_arrow(6.8 + 1.8/2, 2.5, 4.9 + 2.2/2, 1.5 + 0.7)
create_arrow(8.9 + 1.8/2, 2.5, 4.9 + 2.2/2, 1.5 + 0.7)

# Perfect arrow - from best model to interface (from center bottom to center top)
create_arrow(4.9 + 2.2/2, 1.5, 4.9 + 2.2/2, 0.5 + 0.7)

# Phase labels - perfectly positioned
ax.text(6, 6.7, 'PHASE 1: DATA PROCESSING', ha='center', va='center', 
        fontsize=11, fontweight='bold', color=blue)
ax.text(6, 5.2, 'PHASE 2: MODEL TRAINING', ha='center', va='center', 
        fontsize=11, fontweight='bold', color=blue)
ax.text(6, 3.7, 'PHASE 3: EVALUATION', ha='center', va='center', 
        fontsize=11, fontweight='bold', color=blue)
ax.text(6, 0.2, 'PHASE 4: DEPLOYMENT', ha='center', va='center', 
        fontsize=11, fontweight='bold', color=blue)

# Add subtitle
ax.text(6, 8.2, 'Achieving 95% Accuracy with Advanced ML Pipeline', 
        ha='center', va='center', fontsize=10, fontstyle='italic', color='#666666')

# Save with perfect layout
plt.tight_layout()
plt.savefig('simple_flowchart.png', bbox_inches='tight', dpi=300, facecolor='white')
# plt.show()  # Commented out to avoid display issues

print("✅ Perfectly aligned flowchart saved as 'simple_flowchart.png'")
print("🎯 Ready for mentor presentation - clean, professional, and perfectly aligned!")
print("📏 Size: 12x9 inches with exact positioning")
