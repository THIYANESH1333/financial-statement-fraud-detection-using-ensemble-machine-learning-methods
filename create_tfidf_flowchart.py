import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_tfidf_flowchart():
    """Create a clear TF-IDF Feature Extraction Process flowchart"""
    
    # Set up the figure with high DPI for clarity
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors (using grayscale for clarity)
    box_color = 'white'
    border_color = 'black'
    text_color = 'black'
    
    # Define box properties
    box_style = dict(boxstyle="round,pad=0.3", facecolor=box_color, 
                     edgecolor=border_color, linewidth=2)
    
    # Title
    ax.text(5, 11.5, 'TF-IDF Feature Extraction Process', 
            fontsize=16, fontweight='bold', ha='center', color=text_color)
    
    # 1. Raw Financial Text
    ax.text(1.5, 10.5, 'Raw Financial Text', fontsize=12, fontweight='bold', 
            ha='center', color=text_color)
    ax.text(1.5, 10.2, 'MD&A, notes,\naudit remarks', fontsize=10, 
            ha='center', color=text_color)
    raw_box = FancyBboxPatch((0.5, 9.8), 2, 1, boxstyle="round,pad=0.1",
                            facecolor=box_color, edgecolor=border_color, linewidth=2)
    ax.add_patch(raw_box)
    
    # Arrow 1: Raw Text to Preprocessing
    ax.annotate('', xy=(2.5, 10.3), xytext=(2.5, 10.3),
                arrowprops=dict(arrowstyle='->', lw=2, color=border_color))
    
    # 2. Preprocessing
    ax.text(4.5, 10.5, 'Preprocessing', fontsize=12, fontweight='bold', 
            ha='center', color=text_color)
    ax.text(4.5, 10.2, 'cleaning, lemmatize,\nkeep %, FY', fontsize=10, 
            ha='center', color=text_color)
    prep_box = FancyBboxPatch((3.5, 9.8), 2, 1, boxstyle="round,pad=0.1",
                             facecolor=box_color, edgecolor=border_color, linewidth=2)
    ax.add_patch(prep_box)
    
    # Arrow 2: Preprocessing to Tokenization
    ax.annotate('', xy=(5.5, 9.8), xytext=(5.5, 9.8),
                arrowprops=dict(arrowstyle='->', lw=2, color=border_color))
    
    # 3. Tokenization
    ax.text(7.5, 10.5, 'Tokenization', fontsize=12, fontweight='bold', 
            ha='center', color=text_color)
    ax.text(7.5, 10.2, 'words/tokens', fontsize=10, 
            ha='center', color=text_color)
    token_box = FancyBboxPatch((6.5, 9.8), 2, 1, boxstyle="round,pad=0.1",
                              facecolor=box_color, edgecolor=border_color, linewidth=2)
    ax.add_patch(token_box)
    
    # Arrow 3: Tokenization to TF
    ax.annotate('', xy=(7.5, 9.2), xytext=(7.5, 9.2),
                arrowprops=dict(arrowstyle='->', lw=2, color=border_color))
    
    # 4. Term Frequency (TF)
    ax.text(7.5, 8.8, 'Term Frequency (TF)', fontsize=12, fontweight='bold', 
            ha='center', color=text_color)
    ax.text(7.5, 8.5, 'count(term) / total terms', fontsize=10, 
            ha='center', color=text_color)
    tf_box = FancyBboxPatch((6.5, 8.1), 2, 1, boxstyle="round,pad=0.1",
                           facecolor=box_color, edgecolor=border_color, linewidth=2)
    ax.add_patch(tf_box)
    
    # Arrow 4: Tokenization to IDF
    ax.annotate('', xy=(6.5, 9.2), xytext=(6.5, 9.2),
                arrowprops=dict(arrowstyle='->', lw=2, color=border_color))
    
    # 5. Corpus
    ax.text(4.5, 8.8, 'Corpus', fontsize=12, fontweight='bold', 
            ha='center', color=text_color)
    ax.text(4.5, 8.5, 'All documents\nin dataset', fontsize=10, 
            ha='center', color=text_color)
    corpus_box = FancyBboxPatch((3.5, 8.1), 2, 1, boxstyle="round,pad=0.1",
                               facecolor=box_color, edgecolor=border_color, linewidth=2)
    ax.add_patch(corpus_box)
    
    # Arrow 5: Corpus to IDF
    ax.annotate('', xy=(4.5, 7.5), xytext=(4.5, 7.5),
                arrowprops=dict(arrowstyle='->', lw=2, color=border_color))
    
    # 6. Inverse Document Frequency (IDF)
    ax.text(4.5, 7.1, 'Inverse Document\nFrequency (IDF)', fontsize=12, fontweight='bold', 
            ha='center', color=text_color)
    ax.text(4.5, 6.7, 'log(N / df(term))', fontsize=10, 
            ha='center', color=text_color)
    idf_box = FancyBboxPatch((3.5, 6.3), 2, 1, boxstyle="round,pad=0.1",
                            facecolor=box_color, edgecolor=border_color, linewidth=2)
    ax.add_patch(idf_box)
    
    # Arrow 6: IDF to TF-IDF Calculation
    ax.annotate('', xy=(5.5, 6.8), xytext=(5.5, 6.8),
                arrowprops=dict(arrowstyle='->', lw=2, color=border_color))
    
    # Arrow 7: TF to TF-IDF Calculation
    ax.annotate('', xy=(6.5, 7.5), xytext=(6.5, 7.5),
                arrowprops=dict(arrowstyle='->', lw=2, color=border_color))
    
    # 7. TF-IDF Calculation
    ax.text(6, 5.5, 'TF-IDF Calculation', fontsize=12, fontweight='bold', 
            ha='center', color=text_color)
    ax.text(6, 5.2, 'TF(term, doc) x IDF(term)', fontsize=10, 
            ha='center', color=text_color)
    tfidf_box = FancyBboxPatch((5, 4.8), 2, 1, boxstyle="round,pad=0.1",
                              facecolor=box_color, edgecolor=border_color, linewidth=2)
    ax.add_patch(tfidf_box)
    
    # Arrow 8: TF-IDF Calculation to Sparse Vector
    ax.annotate('', xy=(6, 4.2), xytext=(6, 4.2),
                arrowprops=dict(arrowstyle='->', lw=2, color=border_color))
    
    # 8. Sparse Vector Representation
    ax.text(6, 3.8, 'Sparse Vector\nRepresentation', fontsize=12, fontweight='bold', 
            ha='center', color=text_color)
    ax.text(6, 3.4, 'input to SVM / XGBoost / Logistic', fontsize=10, 
            ha='center', color=text_color)
    vector_box = FancyBboxPatch((5, 3.1), 2, 1, boxstyle="round,pad=0.1",
                              facecolor=box_color, edgecolor=border_color, linewidth=2)
    ax.add_patch(vector_box)
    
    # Pipeline description at bottom
    ax.text(5, 1.5, 'Pipeline: Raw Text → Preprocessing → Tokenization → TF → IDF → TF-IDF → Vector', 
            fontsize=11, fontweight='bold', ha='center', color=text_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', edgecolor=border_color, linewidth=1))
    
    # Set background to white
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('TF-IDF_Feature_Extraction_Process.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('TF-IDF_Feature_Extraction_Process.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("TF-IDF Feature Extraction Process flowchart created successfully!")
    print("Files saved:")
    print("- TF-IDF_Feature_Extraction_Process.png (high resolution)")
    print("- TF-IDF_Feature_Extraction_Process.pdf (vector format)")
    
    plt.show()

if __name__ == "__main__":
    create_tfidf_flowchart()
