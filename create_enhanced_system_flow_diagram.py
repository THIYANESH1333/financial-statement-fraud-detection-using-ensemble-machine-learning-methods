from PIL import Image, ImageDraw, ImageFont

def create_enhanced_system_flow_diagram():
    """Create an enhanced system flow diagram for fraud prediction."""
    
    # Create image
    width, height = 1200, 800
    img = Image.new('L', (width, height), 255)  # White background
    draw = ImageDraw.Draw(img)
    
    # Font setup
    try:
        font_title = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 30)
        font_main = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
        font_desc = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    except:
        font_title = ImageFont.load_default()
        font_main = ImageFont.load_default()
        font_desc = ImageFont.load_default()

    text_color = 0  # Black
    box_border_width = 3
    arrow_width = 4
    arrowhead_size = 15
    
    # Box dimensions
    box_width, box_height = 200, 80
    
    # Calculate positions
    title_y = 30
    row1_y = 100
    row2_y = row1_y + box_height + 50
    row3_y = row2_y + box_height + 50
    row4_y = row3_y + box_height + 50
    row5_y = row4_y + box_height + 50
    
    center_x = width // 2
    
    # Title
    title_text = "Enhanced Financial Statement Fraud Detection System"
    title_width, title_height = draw.textsize(title_text, font=font_title)
    draw.text(((width - title_width) // 2, title_y), title_text, fill=text_color, font=font_title)

    # Data Source (Oval)
    data_x = center_x - box_width // 2
    data_coords = (data_x, row1_y, data_x + box_width, row1_y + box_height)
    draw.ellipse(data_coords, outline=text_color, width=box_border_width)
    data_text = "Financial Data"
    data_text_width, data_text_height = draw.textsize(data_text, font=font_main)
    draw.text((data_x + (box_width - data_text_width) // 2, row1_y + (box_height - data_text_height) // 2), 
              data_text, fill=text_color, font=font_main)

    # Preprocessing
    prep_coords = (data_x, row2_y, data_x + box_width, row2_y + box_height)
    draw.rectangle(prep_coords, outline=text_color, width=box_border_width)
    prep_text = "Data Preprocessing"
    prep_text_width, prep_text_height = draw.textsize(prep_text, font=font_main)
    draw.text((data_x + (box_width - prep_text_width) // 2, row2_y + (box_height - prep_text_height) // 2), 
              prep_text, fill=text_color, font=font_main)

    # Feature Extraction
    feat_coords = (data_x, row3_y, data_x + box_width, row3_y + box_height)
    draw.rectangle(feat_coords, outline=text_color, width=box_border_width)
    feat_text = "Feature Extraction"
    feat_text_width, feat_text_height = draw.textsize(feat_text, font=font_main)
    draw.text((data_x + (box_width - feat_text_width) // 2, row3_y + (box_height - feat_text_height) // 2), 
              feat_text, fill=text_color, font=font_main)

    # Base Classifiers (3 boxes side by side)
    classifier_spacing = 50
    classifier_start_x = center_x - (3 * box_width + 2 * classifier_spacing) // 2
    
    # SVM
    svm_coords = (classifier_start_x, row4_y, classifier_start_x + box_width, row4_y + box_height)
    draw.rectangle(svm_coords, outline=text_color, width=box_border_width)
    svm_text = "SVM"
    svm_text_width, svm_text_height = draw.textsize(svm_text, font=font_main)
    draw.text((classifier_start_x + (box_width - svm_text_width) // 2, row4_y + (box_height - svm_text_height) // 2), 
              svm_text, fill=text_color, font=font_main)

    # XGBoost
    xgb_x = classifier_start_x + box_width + classifier_spacing
    xgb_coords = (xgb_x, row4_y, xgb_x + box_width, row4_y + box_height)
    draw.rectangle(xgb_coords, outline=text_color, width=box_border_width)
    xgb_text = "XGBoost"
    xgb_text_width, xgb_text_height = draw.textsize(xgb_text, font=font_main)
    draw.text((xgb_x + (box_width - xgb_text_width) // 2, row4_y + (box_height - xgb_text_height) // 2), 
              xgb_text, fill=text_color, font=font_main)

    # Logistic Regression
    lr_x = xgb_x + box_width + classifier_spacing
    lr_coords = (lr_x, row4_y, lr_x + box_width, row4_y + box_height)
    draw.rectangle(lr_coords, outline=text_color, width=box_border_width)
    lr_text = "Logistic Reg"
    lr_text_width, lr_text_height = draw.textsize(lr_text, font=font_main)
    draw.text((lr_x + (box_width - lr_text_width) // 2, row4_y + (box_height - lr_text_height) // 2), 
              lr_text, fill=text_color, font=font_main)

    # Ensemble
    ensemble_coords = (data_x, row5_y, data_x + box_width, row5_y + box_height)
    draw.rectangle(ensemble_coords, outline=text_color, width=box_border_width)
    ensemble_text = "Ensemble"
    ensemble_text_width, ensemble_text_height = draw.textsize(ensemble_text, font=font_main)
    draw.text((data_x + (box_width - ensemble_text_width) // 2, row5_y + (box_height - ensemble_text_height) // 2), 
              ensemble_text, fill=text_color, font=font_main)

    # Final Prediction
    pred_coords = (data_x, row5_y + box_height + 50, data_x + box_width, row5_y + box_height + 50 + box_height)
    draw.rectangle(pred_coords, outline=text_color, width=box_border_width)
    pred_text = "Fraud Prediction"
    pred_text_width, pred_text_height = draw.textsize(pred_text, font=font_main)
    draw.text((data_x + (box_width - pred_text_width) // 2, row5_y + box_height + 50 + (box_height - pred_text_height) // 2), 
              pred_text, fill=text_color, font=font_main)

    # Draw arrows
    # Data -> Preprocessing
    draw.line([data_x + box_width // 2, row1_y + box_height, data_x + box_width // 2, row2_y], fill=text_color, width=arrow_width)
    draw.polygon([(data_x + box_width // 2 - arrowhead_size // 2, row2_y), (data_x + box_width // 2 + arrowhead_size // 2, row2_y), (data_x + box_width // 2, row2_y - arrowhead_size)], fill=text_color)

    # Preprocessing -> Feature Extraction
    draw.line([data_x + box_width // 2, row2_y + box_height, data_x + box_width // 2, row3_y], fill=text_color, width=arrow_width)
    draw.polygon([(data_x + box_width // 2 - arrowhead_size // 2, row3_y), (data_x + box_width // 2 + arrowhead_size // 2, row3_y), (data_x + box_width // 2, row3_y - arrowhead_size)], fill=text_color)

    # Feature Extraction -> Classifiers (3 arrows)
    # To SVM
    draw.line([data_x + box_width // 2, row3_y + box_height, classifier_start_x + box_width // 2, row4_y], fill=text_color, width=arrow_width)
    draw.polygon([(classifier_start_x + box_width // 2 - arrowhead_size // 2, row4_y), (classifier_start_x + box_width // 2 + arrowhead_size // 2, row4_y), (classifier_start_x + box_width // 2, row4_y - arrowhead_size)], fill=text_color)
    
    # To XGBoost
    draw.line([data_x + box_width // 2, row3_y + box_height, xgb_x + box_width // 2, row4_y], fill=text_color, width=arrow_width)
    draw.polygon([(xgb_x + box_width // 2 - arrowhead_size // 2, row4_y), (xgb_x + box_width // 2 + arrowhead_size // 2, row4_y), (xgb_x + box_width // 2, row4_y - arrowhead_size)], fill=text_color)
    
    # To Logistic Regression
    draw.line([data_x + box_width // 2, row3_y + box_height, lr_x + box_width // 2, row4_y], fill=text_color, width=arrow_width)
    draw.polygon([(lr_x + box_width // 2 - arrowhead_size // 2, row4_y), (lr_x + box_width // 2 + arrowhead_size // 2, row4_y), (lr_x + box_width // 2, row4_y - arrowhead_size)], fill=text_color)

    # Classifiers -> Ensemble (3 arrows)
    # From SVM
    draw.line([classifier_start_x + box_width // 2, row4_y + box_height, data_x + box_width // 2, row5_y], fill=text_color, width=arrow_width)
    draw.polygon([(data_x + box_width // 2 - arrowhead_size // 2, row5_y), (data_x + box_width // 2 + arrowhead_size // 2, row5_y), (data_x + box_width // 2, row5_y - arrowhead_size)], fill=text_color)
    
    # From XGBoost
    draw.line([xgb_x + box_width // 2, row4_y + box_height, data_x + box_width // 2, row5_y], fill=text_color, width=arrow_width)
    
    # From Logistic Regression
    draw.line([lr_x + box_width // 2, row4_y + box_height, data_x + box_width // 2, row5_y], fill=text_color, width=arrow_width)

    # Ensemble -> Final Prediction
    draw.line([data_x + box_width // 2, row5_y + box_height, data_x + box_width // 2, row5_y + box_height + 50], fill=text_color, width=arrow_width)
    draw.polygon([(data_x + box_width // 2 - arrowhead_size // 2, row5_y + box_height + 50), (data_x + box_width // 2 + arrowhead_size // 2, row5_y + box_height + 50), (data_x + box_width // 2, row5_y + box_height + 50 - arrowhead_size)], fill=text_color)

    # Save the image
    output_path = "Enhanced_System_Flow_Diagram.png"
    print(f"Creating enhanced system flow diagram...")
    img.save(output_path)
    print(f"Enhanced system flow diagram saved as {output_path}")

if __name__ == '__main__':
    create_enhanced_system_flow_diagram()
