from PIL import Image, ImageDraw, ImageFont
import os

def create_improved_tfidf_flowchart():
    """Create an improved TF-IDF flowchart with better layout and clarity"""
    
    # Create a larger image for better clarity
    width, height = 1400, 900
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Try to use better fonts
    try:
        font_title = ImageFont.truetype("arial.ttf", 28)
        font_label = ImageFont.truetype("arial.ttf", 18)
        font_desc = ImageFont.truetype("arial.ttf", 14)
        font_pipeline = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font_title = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
            font_label = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
            font_desc = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
            font_pipeline = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
        except:
            font_title = ImageFont.load_default()
            font_label = ImageFont.load_default()
            font_desc = ImageFont.load_default()
            font_pipeline = ImageFont.load_default()
    
    # Title
    title_text = "TF-IDF Feature Extraction Process"
    title_bbox = draw.textbbox((0, 0), title_text, font=font_title)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text((width//2 - title_width//2, 20), title_text, fill='black', font=font_title)
    
    # Define improved box positions and sizes
    box_width, box_height = 200, 100
    
    # Row 1: Raw Text -> Preprocessing -> Tokenization
    row1_y = 120
    boxes_row1 = [
        (50, row1_y, "Raw Financial Text", "MD&A, notes,\naudit remarks"),
        (300, row1_y, "Preprocessing", "cleaning, lemmatize,\nkeep %, FY"),
        (550, row1_y, "Tokenization", "words/tokens")
    ]
    
    # Row 2: TF and Corpus
    row2_y = 280
    boxes_row2 = [
        (550, row2_y, "Term Frequency (TF)", "count(term) / total terms"),
        (300, row2_y, "Corpus", "All documents\nin dataset")
    ]
    
    # Row 3: IDF
    row3_y = 440
    boxes_row3 = [
        (425, row3_y, "Inverse Document\nFrequency (IDF)", "log(N / df(term))")
    ]
    
    # Row 4: TF-IDF Calculation
    row4_y = 600
    boxes_row4 = [
        (425, row4_y, "TF-IDF Calculation", "TF(term, doc) x IDF(term)")
    ]
    
    # Row 5: Sparse Vector
    row5_y = 720
    boxes_row5 = [
        (425, row5_y, "Sparse Vector\nRepresentation", "input to SVM / XGBoost / Logistic")
    ]
    
    # Draw all boxes
    all_boxes = boxes_row1 + boxes_row2 + boxes_row3 + boxes_row4 + boxes_row5
    
    for x, y, label, desc in all_boxes:
        # Draw box with thick black border
        draw.rectangle([x, y, x+box_width, y+box_height], outline='black', width=3)
        
        # Draw label (centered)
        label_bbox = draw.textbbox((0, 0), label, font=font_label)
        label_width = label_bbox[2] - label_bbox[0]
        draw.text((x + box_width//2 - label_width//2, y + 10), label, fill='black', font=font_label)
        
        # Draw description (centered)
        desc_bbox = draw.textbbox((0, 0), desc, font=font_desc)
        desc_width = desc_bbox[2] - desc_bbox[0]
        draw.text((x + box_width//2 - desc_width//2, y + 45), desc, fill='black', font=font_desc)
    
    # Draw improved arrows with arrowheads
    def draw_arrow(x1, y1, x2, y2, arrow_size=8):
        # Draw main arrow line
        draw.line([x1, y1, x2, y2], fill='black', width=4)
        
        # Calculate arrowhead direction
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2)**0.5
        
        if length > 0:
            # Normalize direction
            dx_norm = dx / length
            dy_norm = dy / length
            
            # Arrowhead points
            arrow_x1 = x2 - arrow_size * dx_norm + arrow_size * 0.3 * dy_norm
            arrow_y1 = y2 - arrow_size * dy_norm - arrow_size * 0.3 * dx_norm
            arrow_x2 = x2 - arrow_size * dx_norm - arrow_size * 0.3 * dy_norm
            arrow_y2 = y2 - arrow_size * dy_norm + arrow_size * 0.3 * dx_norm
            
            # Draw arrowhead
            draw.polygon([(x2, y2), (arrow_x1, arrow_y1), (arrow_x2, arrow_y2)], fill='black')
    
    # Draw arrows between boxes
    # Row 1 arrows
    draw_arrow(250, row1_y + box_height//2, 300, row1_y + box_height//2)  # Raw -> Preprocessing
    draw_arrow(500, row1_y + box_height//2, 550, row1_y + box_height//2)  # Preprocessing -> Tokenization
    
    # From Tokenization to TF
    draw_arrow(650, row1_y + box_height//2, 650, row2_y + box_height//2)  # Tokenization -> TF
    
    # From Tokenization to IDF
    draw_arrow(650, row1_y + box_height, 650, row3_y)  # Tokenization -> IDF
    
    # From Corpus to IDF
    draw_arrow(500, row2_y + box_height//2, 425, row3_y)  # Corpus -> IDF
    
    # From TF to TF-IDF
    draw_arrow(650, row2_y + box_height, 650, row4_y)  # TF -> TF-IDF
    
    # From IDF to TF-IDF
    draw_arrow(525, row3_y + box_height, 525, row4_y)  # IDF -> TF-IDF
    
    # From TF-IDF to Sparse Vector
    draw_arrow(525, row4_y + box_height, 525, row5_y)  # TF-IDF -> Vector
    
    # Pipeline description at bottom
    pipeline_text = "Pipeline: Raw Text → Preprocessing → Tokenization → TF → IDF → TF-IDF → Vector"
    pipeline_bbox = draw.textbbox((0, 0), pipeline_text, font=font_pipeline)
    pipeline_width = pipeline_bbox[2] - pipeline_bbox[0]
    
    # Draw background for pipeline text
    draw.rectangle([width//2 - pipeline_width//2 - 20, height - 60, 
                   width//2 + pipeline_width//2 + 20, height - 20], 
                  fill='lightgray', outline='black', width=2)
    
    draw.text((width//2 - pipeline_width//2, height - 50), pipeline_text, 
              fill='black', font=font_pipeline)
    
    # Add some decorative elements
    # Draw a subtle border around the entire flowchart
    draw.rectangle([20, 80, width-20, height-80], outline='gray', width=2)
    
    # Save the improved image
    img.save('TF-IDF_Feature_Extraction_Process_Improved.png')
    print("Improved TF-IDF Feature Extraction Process flowchart created successfully!")
    print("File saved: TF-IDF_Feature_Extraction_Process_Improved.png")
    
    # Also create a PDF version
    try:
        img.save('TF-IDF_Feature_Extraction_Process_Improved.pdf')
        print("PDF version also saved: TF-IDF_Feature_Extraction_Process_Improved.pdf")
    except:
        print("PDF version could not be created (PIL PDF support may not be available)")

if __name__ == "__main__":
    create_improved_tfidf_flowchart()
