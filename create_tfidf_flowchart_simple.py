from PIL import Image, ImageDraw, ImageFont
import os

def create_tfidf_flowchart_simple():
    """Create a simple TF-IDF flowchart using PIL"""
    
    # Create a white image
    width, height = 1200, 800
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font_title = ImageFont.truetype("arial.ttf", 24)
        font_label = ImageFont.truetype("arial.ttf", 16)
        font_desc = ImageFont.truetype("arial.ttf", 12)
    except:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
        font_desc = ImageFont.load_default()
    
    # Title
    draw.text((width//2 - 200, 30), "TF-IDF Feature Extraction Process", 
              fill='black', font=font_title)
    
    # Define box positions and sizes
    boxes = [
        # (x, y, width, height, label, description)
        (50, 100, 200, 80, "Raw Financial Text", "MD&A, notes,\naudit remarks"),
        (300, 100, 200, 80, "Preprocessing", "cleaning, lemmatize,\nkeep %, FY"),
        (550, 100, 200, 80, "Tokenization", "words/tokens"),
        (800, 100, 200, 80, "Term Frequency (TF)", "count(term) / total terms"),
        (300, 250, 200, 80, "Corpus", "All documents\nin dataset"),
        (550, 250, 200, 80, "Inverse Document\nFrequency (IDF)", "log(N / df(term))"),
        (800, 400, 200, 80, "TF-IDF Calculation", "TF(term, doc) x IDF(term)"),
        (550, 550, 200, 80, "Sparse Vector\nRepresentation", "input to SVM / XGBoost / Logistic")
    ]
    
    # Draw boxes
    for x, y, w, h, label, desc in boxes:
        # Draw box with black border
        draw.rectangle([x, y, x+w, y+h], outline='black', width=3)
        # Draw label
        draw.text((x + w//2 - 50, y + 10), label, fill='black', font=font_label)
        # Draw description
        draw.text((x + w//2 - 50, y + 35), desc, fill='black', font=font_desc)
    
    # Draw arrows
    arrows = [
        # From Raw Text to Preprocessing
        (250, 140, 300, 140),
        # From Preprocessing to Tokenization
        (500, 140, 550, 140),
        # From Tokenization to TF
        (650, 140, 800, 140),
        # From Tokenization to IDF
        (650, 190, 650, 250),
        # From Corpus to IDF
        (400, 290, 550, 290),
        # From IDF to TF-IDF
        (650, 330, 800, 400),
        # From TF to TF-IDF
        (900, 190, 900, 400),
        # From TF-IDF to Vector
        (900, 480, 650, 550)
    ]
    
    for x1, y1, x2, y2 in arrows:
        # Draw arrow line
        draw.line([x1, y1, x2, y2], fill='black', width=3)
        # Draw arrowhead
        if x2 > x1:  # Right arrow
            draw.polygon([(x2-10, y2-5), (x2, y2), (x2-10, y2+5)], fill='black')
        elif x2 < x1:  # Left arrow
            draw.polygon([(x2+10, y2-5), (x2, y2), (x2+10, y2+5)], fill='black')
        elif y2 > y1:  # Down arrow
            draw.polygon([(x2-5, y2-10), (x2, y2), (x2+5, y2-10)], fill='black')
        elif y2 < y1:  # Up arrow
            draw.polygon([(x2-5, y2+10), (x2, y2), (x2+5, y2+10)], fill='black')
    
    # Pipeline description at bottom
    pipeline_text = "Pipeline: Raw Text → Preprocessing → Tokenization → TF → IDF → TF-IDF → Vector"
    draw.text((width//2 - 300, height - 50), pipeline_text, fill='black', font=font_label)
    
    # Save the image
    img.save('TF-IDF_Feature_Extraction_Process.png')
    print("TF-IDF Feature Extraction Process flowchart created successfully!")
    print("File saved: TF-IDF_Feature_Extraction_Process.png")

if __name__ == "__main__":
    create_tfidf_flowchart_simple()
