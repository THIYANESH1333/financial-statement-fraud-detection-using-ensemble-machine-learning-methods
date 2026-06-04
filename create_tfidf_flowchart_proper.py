from PIL import Image, ImageDraw, ImageFont
import math

def create_properly_aligned_tfidf_flowchart():
    """Create a properly aligned TF-IDF flowchart with precise arrows and text"""
    
    # Create a larger image for better clarity
    width, height = 1600, 1000
    img = Image.new('L', (width, height), 255)  # White background
    draw = ImageDraw.Draw(img)
    
    # Font setup
    try:
        font_title = ImageFont.truetype("arial.ttf", 32)
        font_label = ImageFont.truetype("arial.ttf", 20)
        font_desc = ImageFont.truetype("arial.ttf", 16)
        font_pipeline = ImageFont.truetype("arial.ttf", 18)
    except:
        try:
            font_title = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 32)
            font_label = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
            font_desc = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
            font_pipeline = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
        except:
            font_title = ImageFont.load_default()
            font_label = ImageFont.load_default()
            font_desc = ImageFont.load_default()
            font_pipeline = ImageFont.load_default()
    
    # Box dimensions
    box_width, box_height = 250, 120
    border_width = 3
    arrow_width = 4
    
    # Function to get text dimensions
    def get_text_size(text, font):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    # Function to draw centered text
    def draw_centered_text(x, y, text, font, fill=0):
        text_width, text_height = get_text_size(text, font)
        draw.text((x - text_width//2, y - text_height//2), text, fill=fill, font=font)
    
    # Function to draw multi-line centered text
    def draw_centered_multiline(x, y, lines, font, fill=0):
        total_height = sum(get_text_size(line, font)[1] for line in lines)
        start_y = y - total_height//2
        
        for line in lines:
            text_width, text_height = get_text_size(line, font)
            draw.text((x - text_width//2, start_y), line, fill=fill, font=font)
            start_y += text_height
    
    # Function to draw arrow with proper arrowhead
    def draw_arrow(x1, y1, x2, y2, fill=0, width=4):
        # Draw main line
        draw.line([(x1, y1), (x2, y2)], fill=fill, width=width)
        
        # Calculate arrowhead
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_length = 15
        arrow_angle = math.pi / 6  # 30 degrees
        
        # Arrowhead points
        x3 = x2 - arrow_length * math.cos(angle - arrow_angle)
        y3 = y2 - arrow_length * math.sin(angle - arrow_angle)
        x4 = x2 - arrow_length * math.cos(angle + arrow_angle)
        y4 = y2 - arrow_length * math.sin(angle + arrow_angle)
        
        # Draw arrowhead
        draw.polygon([(x2, y2), (x3, y3), (x4, y4)], fill=fill, outline=fill)
    
    # Title
    title_text = "TF-IDF Feature Extraction Process"
    draw_centered_text(width//2, 50, title_text, font_title)
    
    # Define box positions (centered coordinates)
    # Row 1
    raw_x, raw_y = 200, 200
    prep_x, prep_y = 500, 200
    token_x, token_y = 800, 200
    
    # Row 2
    corpus_x, corpus_y = 350, 400
    tf_x, tf_y = 800, 400
    
    # Row 3
    idf_x, idf_y = 575, 600
    
    # Row 4
    tfidf_x, tfidf_y = 575, 800
    
    # Row 5
    vector_x, vector_y = 575, 900
    
    # Draw boxes with precise positioning
    boxes = [
        (raw_x, raw_y, "Raw Financial Text", ["MD&A, notes,", "audit remarks"]),
        (prep_x, prep_y, "Preprocessing", ["cleaning, lemmatize,", "keep %, FY"]),
        (token_x, token_y, "Tokenization", ["words/tokens"]),
        (corpus_x, corpus_y, "Corpus", ["All documents", "in dataset"]),
        (tf_x, tf_y, "Term Frequency (TF)", ["count(term) /", "total terms"]),
        (idf_x, idf_y, "Inverse Document\nFrequency (IDF)", ["log(N / df(term))"]),
        (tfidf_x, tfidf_y, "TF-IDF Calculation", ["TF(term, doc) x IDF(term)"]),
        (vector_x, vector_y, "Sparse Vector\nRepresentation", ["input to SVM /", "XGBoost / Logistic"])
    ]
    
    for center_x, center_y, label, desc in boxes:
        # Draw box
        x1 = center_x - box_width//2
        y1 = center_y - box_height//2
        x2 = center_x + box_width//2
        y2 = center_y + box_height//2
        
        draw.rectangle([x1, y1, x2, y2], outline=0, width=border_width)
        
        # Draw label
        draw_centered_text(center_x, center_y - 20, label, font_label)
        
        # Draw description
        draw_centered_multiline(center_x, center_y + 20, desc, font_desc)
    
    # Draw arrows with precise connections
    # Raw Text -> Preprocessing
    draw_arrow(raw_x + box_width//2, raw_y, prep_x - box_width//2, prep_y)
    
    # Preprocessing -> Tokenization
    draw_arrow(prep_x + box_width//2, prep_y, token_x - box_width//2, token_y)
    
    # Tokenization -> TF (vertical down)
    draw_arrow(token_x, token_y + box_height//2, tf_x, tf_y - box_height//2)
    
    # Tokenization -> Corpus (diagonal)
    draw_arrow(token_x - box_width//2, token_y + box_height//2, corpus_x + box_width//2, corpus_y - box_height//2)
    
    # Corpus -> IDF (horizontal)
    draw_arrow(corpus_x + box_width//2, corpus_y, idf_x - box_width//2, idf_y)
    
    # TF -> IDF (vertical up)
    draw_arrow(tf_x, tf_y + box_height//2, idf_x, idf_y - box_height//2)
    
    # IDF -> TF-IDF (vertical down)
    draw_arrow(idf_x, idf_y + box_height//2, tfidf_x, tfidf_y - box_height//2)
    
    # TF-IDF -> Vector (vertical down)
    draw_arrow(tfidf_x, tfidf_y + box_height//2, vector_x, vector_y - box_height//2)
    
    # Pipeline description at bottom
    pipeline_text = "Pipeline: Raw Text → Preprocessing → Tokenization → TF → IDF → TF-IDF → Vector"
    pipeline_y = height - 50
    draw_centered_text(width//2, pipeline_y, pipeline_text, font_pipeline)
    
    # Save the image
    img.save('TF-IDF_Feature_Extraction_Process_Proper.png')
    print("Properly aligned TF-IDF flowchart created successfully!")
    print("File saved: TF-IDF_Feature_Extraction_Process_Proper.png")

if __name__ == "__main__":
    create_properly_aligned_tfidf_flowchart()
