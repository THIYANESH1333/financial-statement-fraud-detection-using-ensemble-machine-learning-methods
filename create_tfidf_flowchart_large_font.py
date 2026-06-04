from PIL import Image, ImageDraw, ImageFont
import math

def create_tfidf_flowchart_large_font():
    """Create a TF-IDF flowchart with larger font sizes for better readability"""
    
    # Create a larger image for better clarity
    width, height = 2000, 1400
    img = Image.new('L', (width, height), 255)  # White background
    draw = ImageDraw.Draw(img)
    
    # Font setup - LARGER FONT SIZES
    try:
        font_title = ImageFont.truetype("arial.ttf", 50)
        font_label = ImageFont.truetype("arial.ttf", 32)
        font_desc = ImageFont.truetype("arial.ttf", 24)
        font_pipeline = ImageFont.truetype("arial.ttf", 28)
    except:
        try:
            font_title = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 50)
            font_label = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 32)
            font_desc = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
            font_pipeline = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
        except:
            font_title = ImageFont.load_default()
            font_label = ImageFont.load_default()
            font_desc = ImageFont.load_default()
            font_pipeline = ImageFont.load_default()
    
    # Box dimensions
    box_width, box_height = 350, 160
    border_width = 4
    arrow_width = 6
    arrowhead_size = 25
    
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
        line_heights = []
        total_height = 0
        
        for line in lines:
            _, line_height = get_text_size(line, font)
            line_heights.append(line_height)
            total_height += line_height
        
        start_y = y - total_height//2
        
        for i, line in enumerate(lines):
            text_width, _ = get_text_size(line, font)
            draw.text((x - text_width//2, start_y), line, fill=fill, font=font)
            start_y += line_heights[i]
    
    # Function to draw arrow with proper arrowhead
    def draw_arrow(x1, y1, x2, y2, fill=0, width=6):
        # Draw main line
        draw.line([(x1, y1), (x2, y2)], fill=fill, width=width)
        
        # Calculate arrowhead
        angle = math.atan2(y2 - y1, x2 - x1)
        
        # Arrowhead points
        x3 = x2 - arrowhead_size * math.cos(angle - math.pi / 6)
        y3 = y2 - arrowhead_size * math.sin(angle - math.pi / 6)
        x4 = x2 - arrowhead_size * math.cos(angle + math.pi / 6)
        y4 = y2 - arrowhead_size * math.sin(angle + math.pi / 6)
        
        # Draw arrowhead
        draw.polygon([(x2, y2), (x3, y3), (x4, y4)], fill=fill, outline=fill)
    
    # Function to draw L-shaped arrow
    def draw_l_arrow(x1, y1, x2, y2, x3, y3, fill=0, width=6):
        # Draw first segment
        draw.line([(x1, y1), (x2, y2)], fill=fill, width=width)
        # Draw second segment
        draw.line([(x2, y2), (x3, y3)], fill=fill, width=width)
        # Draw arrowhead at end
        angle = math.atan2(y3 - y2, x3 - x2)
        x4 = x3 - arrowhead_size * math.cos(angle - math.pi / 6)
        y4 = y3 - arrowhead_size * math.sin(angle - math.pi / 6)
        x5 = x3 - arrowhead_size * math.cos(angle + math.pi / 6)
        y5 = y3 - arrowhead_size * math.sin(angle + math.pi / 6)
        draw.polygon([(x3, y3), (x4, y4), (x5, y5)], fill=fill, outline=fill)
    
    # Title
    title_text = "TF-IDF Feature Extraction Process"
    draw_centered_text(width//2, 80, title_text, font_title)
    
    # Define precise box positions
    # Row 1: Raw Text, Preprocessing, Tokenization
    row1_y = 250
    raw_x = 250
    prep_x = 700
    token_x = 1150
    
    # Row 2: Corpus, Term Frequency
    row2_y = 500
    corpus_x = 475
    tf_x = 1150
    
    # Row 3: IDF
    row3_y = 750
    idf_x = 812
    
    # Row 4: TF-IDF Calculation
    row4_y = 1000
    tfidf_x = 812
    
    # Row 5: Sparse Vector
    row5_y = 1250
    vector_x = 812
    
    # Draw boxes with precise positioning
    boxes = [
        (raw_x, row1_y, "Raw Financial Text", ["MD&A, notes,", "audit remarks"]),
        (prep_x, row1_y, "Preprocessing", ["cleaning, lemmatize,", "keep %, FY"]),
        (token_x, row1_y, "Tokenization", ["words/tokens"]),
        (corpus_x, row2_y, "Corpus", ["All documents", "in dataset"]),
        (tf_x, row2_y, "Term Frequency (TF)", ["count(term) /", "total terms"]),
        (idf_x, row3_y, "Inverse Document\nFrequency (IDF)", ["log(N / df(term))"]),
        (tfidf_x, row4_y, "TF-IDF Calculation", ["TF(term, doc) x IDF(term)"]),
        (vector_x, row5_y, "Sparse Vector\nRepresentation", ["input to SVM /", "XGBoost / Logistic"])
    ]
    
    for center_x, center_y, label, desc in boxes:
        # Draw box
        x1 = center_x - box_width//2
        y1 = center_y - box_height//2
        x2 = center_x + box_width//2
        y2 = center_y + box_height//2
        
        draw.rectangle([x1, y1, x2, y2], outline=0, width=border_width)
        
        # Draw label
        draw_centered_text(center_x, center_y - 30, label, font_label)
        
        # Draw description
        draw_centered_multiline(center_x, center_y + 30, desc, font_desc)
    
    # Draw arrows with precise connections
    # Raw Text -> Preprocessing
    draw_arrow(raw_x + box_width//2, row1_y, prep_x - box_width//2, row1_y)
    
    # Preprocessing -> Tokenization
    draw_arrow(prep_x + box_width//2, row1_y, token_x - box_width//2, row1_y)
    
    # Tokenization -> TF (vertical down)
    draw_arrow(token_x, row1_y + box_height//2, tf_x, row2_y - box_height//2)
    
    # Tokenization -> Corpus (L-shaped)
    draw_l_arrow(token_x - box_width//2, row1_y + box_height//2, 
                token_x - box_width//2, row2_y - box_height//2,
                corpus_x + box_width//2, row2_y - box_height//2)
    
    # Corpus -> IDF (L-shaped)
    draw_l_arrow(corpus_x + box_width//2, row2_y + box_height//2,
                corpus_x + box_width//2, row3_y - box_height//2,
                idf_x - box_width//2, row3_y - box_height//2)
    
    # TF -> IDF (vertical up)
    draw_arrow(tf_x, row2_y + box_height//2, idf_x, row3_y - box_height//2)
    
    # IDF -> TF-IDF (vertical down)
    draw_arrow(idf_x, row3_y + box_height//2, tfidf_x, row4_y - box_height//2)
    
    # TF-IDF -> Vector (vertical down)
    draw_arrow(tfidf_x, row4_y + box_height//2, vector_x, row5_y - box_height//2)
    
    # Pipeline description at bottom
    pipeline_text = "Pipeline: Raw Text → Preprocessing → Tokenization → TF → IDF → TF-IDF → Vector"
    draw_centered_text(width//2, height - 60, pipeline_text, font_pipeline)
    
    # Save the image
    img.save('TF-IDF_Feature_Extraction_Process_Large_Font.png')
    print("TF-IDF flowchart with large font created successfully!")
    print("File saved: TF-IDF_Feature_Extraction_Process_Large_Font.png")

if __name__ == "__main__":
    create_tfidf_flowchart_large_font()
