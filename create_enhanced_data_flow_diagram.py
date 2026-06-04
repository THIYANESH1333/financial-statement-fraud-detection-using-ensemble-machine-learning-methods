from PIL import Image, ImageDraw, ImageFont

def create_enhanced_data_flow_diagram():
    """Create an enhanced data flow diagram for financial fraud prediction system."""
    
    # Create a larger image for better clarity
    width, height = 1600, 1200
    img = Image.new('L', (width, height), 255)  # White background
    draw = ImageDraw.Draw(img)
    
    # Font setup
    try:
        font_title = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 35)
        font_main = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
        font_sub = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
        font_desc = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    except:
        font_title = ImageFont.load_default()
        font_main = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_desc = ImageFont.load_default()

    # Colors and properties
    text_color = 0  # Black
    box_border_width = 3
    arrow_width = 4
    arrowhead_size = 15
    
    # Box dimensions
    main_box_width, main_box_height = 280, 100
    classifier_box_width, classifier_box_height = 200, 80
    data_box_width, data_box_height = 300, 60
    
    # Calculate positions
    title_y = 40
    row1_y = 120
    row2_y = row1_y + data_box_height + 60
    row3_y = row2_y + main_box_height + 60
    row4_y = row3_y + main_box_height + 60
    row5_y = row4_y + classifier_box_height + 60
    row6_y = row5_y + classifier_box_height + 60
    row7_y = row6_y + main_box_height + 60
    
    center_x = width // 2
    
    # Helper function to draw a box with centered text
    def draw_box_with_text(draw_obj, coords, main_text, sub_text="", font_main, font_sub, text_color, border_width):
        x1, y1, x2, y2 = coords
        draw_obj.rectangle([x1, y1, x2, y2], outline=text_color, width=border_width)
        
        # Center main text
        main_width, main_height = draw_obj.textsize(main_text, font=font_main)
        main_x = x1 + (x2 - x1 - main_width) // 2
        main_y = y1 + 15
        draw_obj.text((main_x, main_y), main_text, fill=text_color, font=font_main)
        
        # Center sub text (if any)
        if sub_text:
            sub_width, sub_height = draw_obj.textsize(sub_text, font=font_sub)
            sub_x = x1 + (x2 - x1 - sub_width) // 2
            sub_y = main_y + main_height + 5
            draw_obj.text((sub_x, sub_y), sub_text, fill=text_color, font=font_sub)

    # Helper function to draw an arrow
    def draw_arrow(draw_obj, x1, y1, x2, y2, arrow_w, head_s, direction='down'):
        draw_obj.line([x1, y1, x2, y2], fill=text_color, width=arrow_w)
        
        # Draw arrowhead
        if direction == 'down':
            draw_obj.polygon([(x2 - head_s // 2, y2), (x2 + head_s // 2, y2), (x2, y2 + head_s)], fill=text_color)
        elif direction == 'right':
            draw_obj.polygon([(x2, y2 - head_s // 2), (x2, y2 + head_s // 2), (x2 + head_s, y2)], fill=text_color)
        elif direction == 'left':
            draw_obj.polygon([(x2, y2 - head_s // 2), (x2, y2 + head_s // 2), (x2 - head_s, y2)], fill=text_color)
        elif direction == 'up':
            draw_obj.polygon([(x2 - head_s // 2, y2), (x2 + head_s // 2, y2), (x2, y2 - head_s)], fill=text_color)

    # Title
    title_text = "Enhanced Financial Statement Fraud Detection System - Data Flow"
    title_width, title_height = draw.textsize(title_text, font=font_title)
    draw.text(((width - title_width) // 2, title_y), title_text, fill=text_color, font=font_title)

    # Data Source (Oval shape)
    data_x = center_x - data_box_width // 2
    data_coords = (data_x, row1_y, data_x + data_box_width, row1_y + data_box_height)
    draw.ellipse(data_coords, outline=text_color, width=box_border_width)
    data_text = "Financial Data"
    data_text_width, data_text_height = draw.textsize(data_text, font=font_main)
    draw.text((data_x + (data_box_width - data_text_width) // 2, row1_y + (data_box_height - data_text_height) // 2), 
              data_text, fill=text_color, font=font_main)

    # Data Preprocessing
    prep_x = center_x - main_box_width // 2
    prep_coords = (prep_x, row2_y, prep_x + main_box_width, row2_y + main_box_height)
    draw_box_with_text(draw, prep_coords, "Data Preprocessing", 
                       "Text Cleaning & Numerical Processing", 
                       font_main, font_sub, text_color, box_border_width)

    # Feature Engineering
    feat_x = center_x - main_box_width // 2
    feat_coords = (feat_x, row3_y, feat_x + main_box_width, row3_y + main_box_height)
    draw_box_with_text(draw, feat_coords, "Feature Engineering", 
                       "Multi-Modal Feature Extraction", 
                       font_main, font_sub, text_color, box_border_width)

    # Feature Types (3 boxes side by side)
    feature_spacing = 20
    feature_start_x = center_x - (3 * 180 + 2 * feature_spacing) // 2
    
    # TF-IDF Features
    tfidf_x = feature_start_x
    tfidf_coords = (tfidf_x, row4_y, tfidf_x + 180, row4_y + 60)
    draw_box_with_text(draw, tfidf_coords, "TF-IDF Features", 
                       "Text Vectorization", 
                       font_main, font_sub, text_color, box_border_width)

    # Numerical Features
    num_x = tfidf_x + 180 + feature_spacing
    num_coords = (num_x, row4_y, num_x + 180, row4_y + 60)
    draw_box_with_text(draw, num_coords, "Numerical Features", 
                       "Financial Ratios", 
                       font_main, font_sub, text_color, box_border_width)

    # Sentiment Features
    sent_x = num_x + 180 + feature_spacing
    sent_coords = (sent_x, row4_y, sent_x + 180, row4_y + 60)
    draw_box_with_text(draw, sent_coords, "Sentiment Features", 
                       "VADER & TextBlob", 
                       font_main, font_sub, text_color, box_border_width)

    # Base Classifiers (3 boxes side by side)
    classifier_spacing = 40
    classifier_start_x = center_x - (3 * classifier_box_width + 2 * classifier_spacing) // 2
    
    # SVM Classifier
    svm_x = classifier_start_x
    svm_coords = (svm_x, row5_y, svm_x + classifier_box_width, row5_y + classifier_box_height)
    draw_box_with_text(draw, svm_coords, "SVM Classifier", 
                       "RBF Kernel", 
                       font_main, font_sub, text_color, box_border_width)

    # XGBoost Classifier
    xgb_x = svm_x + classifier_box_width + classifier_spacing
    xgb_coords = (xgb_x, row5_y, xgb_x + classifier_box_width, row5_y + classifier_box_height)
    draw_box_with_text(draw, xgb_coords, "XGBoost Classifier", 
                       "Gradient Boosting", 
                       font_main, font_sub, text_color, box_border_width)

    # Logistic Regression
    lr_x = xgb_x + classifier_box_width + classifier_spacing
    lr_coords = (lr_x, row5_y, lr_x + classifier_box_width, row5_y + classifier_box_height)
    draw_box_with_text(draw, lr_coords, "Logistic Regression", 
                       "Linear Classifier", 
                       font_main, font_sub, text_color, box_border_width)

    # Voting Classifier Ensemble
    ensemble_x = center_x - main_box_width // 2
    ensemble_coords = (ensemble_x, row6_y, ensemble_x + main_box_width, row6_y + main_box_height)
    draw_box_with_text(draw, ensemble_coords, "Voting Classifier", 
                       "Hard & Soft Voting", 
                       font_main, font_sub, text_color, box_border_width)

    # Final Prediction
    pred_x = center_x - main_box_width // 2
    pred_coords = (pred_x, row7_y, pred_x + main_box_width, row7_y + main_box_height)
    draw_box_with_text(draw, pred_coords, "Fraud Prediction", 
                       "Binary Classification", 
                       font_main, font_sub, text_color, box_border_width)

    # Draw arrows
    # Data -> Preprocessing
    draw_arrow(draw, data_x + data_box_width // 2, row1_y + data_box_height,
               prep_x + main_box_width // 2, row2_y,
               arrow_width, arrowhead_size, direction='down')

    # Preprocessing -> Feature Engineering
    draw_arrow(draw, prep_x + main_box_width // 2, row2_y + main_box_height,
               feat_x + main_box_width // 2, row3_y,
               arrow_width, arrowhead_size, direction='down')

    # Feature Engineering -> Feature Types (3 arrows)
    # To TF-IDF
    draw_arrow(draw, feat_x + main_box_width // 2, row3_y + main_box_height,
               tfidf_x + 90, row4_y,
               arrow_width, arrowhead_size, direction='down')
    
    # To Numerical
    draw_arrow(draw, feat_x + main_box_width // 2, row3_y + main_box_height,
               num_x + 90, row4_y,
               arrow_width, arrowhead_size, direction='down')
    
    # To Sentiment
    draw_arrow(draw, feat_x + main_box_width // 2, row3_y + main_box_height,
               sent_x + 90, row4_y,
               arrow_width, arrowhead_size, direction='down')

    # Feature Types -> Classifiers (3 arrows from each feature type)
    # From TF-IDF to all classifiers
    draw_arrow(draw, tfidf_x + 90, row4_y + 60, svm_x + classifier_box_width // 2, row5_y, arrow_width, arrowhead_size, direction='down')
    draw_arrow(draw, tfidf_x + 90, row4_y + 60, xgb_x + classifier_box_width // 2, row5_y, arrow_width, arrowhead_size, direction='down')
    draw_arrow(draw, tfidf_x + 90, row4_y + 60, lr_x + classifier_box_width // 2, row5_y, arrow_width, arrowhead_size, direction='down')
    
    # From Numerical to all classifiers
    draw_arrow(draw, num_x + 90, row4_y + 60, svm_x + classifier_box_width // 2, row5_y, arrow_width, arrowhead_size, direction='down')
    draw_arrow(draw, num_x + 90, row4_y + 60, xgb_x + classifier_box_width // 2, row5_y, arrow_width, arrowhead_size, direction='down')
    draw_arrow(draw, num_x + 90, row4_y + 60, lr_x + classifier_box_width // 2, row5_y, arrow_width, arrowhead_size, direction='down')
    
    # From Sentiment to all classifiers
    draw_arrow(draw, sent_x + 90, row4_y + 60, svm_x + classifier_box_width // 2, row5_y, arrow_width, arrowhead_size, direction='down')
    draw_arrow(draw, sent_x + 90, row4_y + 60, xgb_x + classifier_box_width // 2, row5_y, arrow_width, arrowhead_size, direction='down')
    draw_arrow(draw, sent_x + 90, row4_y + 60, lr_x + classifier_box_width // 2, row5_y, arrow_width, arrowhead_size, direction='down')

    # Classifiers -> Ensemble (3 arrows)
    # From SVM
    draw_arrow(draw, svm_x + classifier_box_width // 2, row5_y + classifier_box_height,
               ensemble_x + main_box_width // 2, row6_y,
               arrow_width, arrowhead_size, direction='down')
    
    # From XGBoost
    draw_arrow(draw, xgb_x + classifier_box_width // 2, row5_y + classifier_box_height,
               ensemble_x + main_box_width // 2, row6_y,
               arrow_width, arrowhead_size, direction='down')
    
    # From Logistic Regression
    draw_arrow(draw, lr_x + classifier_box_width // 2, row5_y + classifier_box_height,
               ensemble_x + main_box_width // 2, row6_y,
               arrow_width, arrowhead_size, direction='down')

    # Ensemble -> Final Prediction
    draw_arrow(draw, ensemble_x + main_box_width // 2, row6_y + main_box_height,
               pred_x + main_box_width // 2, row7_y,
               arrow_width, arrowhead_size, direction='down')

    # Add performance metrics box on the right side
    metrics_x = width - 350
    metrics_y = row3_y
    metrics_coords = (metrics_x, metrics_y, metrics_x + 320, metrics_y + 200)
    draw.rectangle(metrics_coords, outline=text_color, width=box_border_width)
    
    # Performance metrics text
    metrics_title = "Performance Metrics"
    metrics_title_width, metrics_title_height = draw.textsize(metrics_title, font=font_sub)
    draw.text((metrics_x + (320 - metrics_title_width) // 2, metrics_y + 15), 
              metrics_title, fill=text_color, font=font_sub)
    
    metrics_text = " XGBoost: 95.0% Accuracy\n Voting Classifier: 90.0%\n SVM: 85.0%\n Logistic Regression: 85.0%\n Deep Learning: 81.0%"
    metrics_lines = metrics_text.split('\n')
    current_y = metrics_y + 40
    for line in metrics_lines:
        line_width, line_height = draw.textsize(line, font=font_desc)
        draw.text((metrics_x + (320 - line_width) // 2, current_y), line, fill=text_color, font=font_desc)
        current_y += line_height + 5

    # Add data flow indicators at the bottom
    flow_text = "Data Flow: Raw Data  Preprocessing  Feature Engineering  Multi-Modal Features  Base Classifiers  Ensemble  Prediction"
    flow_width, flow_height = draw.textsize(flow_text, font=font_desc)
    draw.text(((width - flow_width) // 2, height - 30), flow_text, fill=text_color, font=font_desc)

    # Save the image
    output_path = "Enhanced_Data_Flow_Diagram.png"
    print(f"Creating enhanced data flow diagram...")
    img.save(output_path)
    print(f"Enhanced data flow diagram saved as {output_path}")

if __name__ == '__main__':
    create_enhanced_data_flow_diagram()
