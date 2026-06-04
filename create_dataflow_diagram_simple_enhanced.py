from PIL import Image, ImageDraw, ImageFont

width, height = 1400, 1200
img = Image.new('L', (width, height), 255)
draw = ImageDraw.Draw(img)

# Fonts
try:
    ft = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 36)
    fm = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 24)
    fs = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
except:
    ft = fm = fs = ImageFont.load_default()

black = 0
BW = 3

# Helpers
def box(x, y, w, h, text, sub=None):
    draw.rectangle([x, y, x+w, y+h], outline=black, width=BW)
    tw, th = draw.textbbox((0,0), text, font=fm)[2:]
    draw.text((x + (w-tw)//2, y + 12), text, fill=black, font=fm)
    if sub:
        sw, sh = draw.textbbox((0,0), sub, font=fs)[2:]
        draw.text((x + (w-sw)//2, y + 12 + th + 6), sub, fill=black, font=fs)

def arrow(x1,y1,x2,y2,dir='down'):
    draw.line([x1,y1,x2,y2], fill=black, width=4)
    if dir=='down':
        draw.polygon([(x2-10,y2),(x2+10,y2),(x2,y2+18)], fill=black)
    elif dir=='right':
        draw.polygon([(x2,y2-10),(x2,y2+10),(x2+18,y2)], fill=black)

# Title
title = 'Enhanced Fraud Detection Data Flow'
TW = draw.textbbox((0,0), title, font=ft)[2]
draw.text(((width-TW)//2, 40), title, fill=black, font=ft)

# Layout
cx = width//2
w,h = 320,100
small_w, small_h = 220,72

# Data (oval)
ox, oy, ow, oh = cx-150, 120, 300, 70
draw.ellipse([ox,oy,ox+ow,oy+oh], outline=black, width=BW)
label = 'Financial Data'
LW = draw.textbbox((0,0), label, font=fm)[2]
draw.text((ox + (ow-LW)//2, oy + (oh-24)//2), label, fill=black, font=fm)

# Preprocessing
px, py = cx-w//2, 230
box(px, py, w, h, 'Financial Data Preprocessing', 'Cleaning, standardization')
arrow(cx, oy+oh, cx, py, 'down')

# Feature Extraction
fx, fy = px, py+h+60
box(fx, fy, w, h, 'Feature Extraction', '(TF-IDF, Numerical, Sentiment)')
arrow(cx, py+h, cx, fy, 'down')

# Training Base Classifiers
bx, by = fx, fy+h+60
box(bx, by, w, h, 'Training Base Classifiers')
arrow(cx, fy+h, cx, by, 'down')

# Three classifiers row
gap = 40
svx = cx - small_w - gap//2
xgbx = cx - small_w//2
lrx = cx + gap//2
cy = by + h + 40
box(svx, cy, small_w, small_h, 'SVM Classifier', 'RBF kernel')
box(xgbx, cy, small_w, small_h, 'XGBoost Classifier', 'Gradient boosting')
box(lrx, cy, small_w, small_h, 'Logistic Regression', 'Linear model')
arrow(cx, by+h, cx, cy, 'down')

# Voting Ensemble
vy = cy + small_h + 60
box(cx-w//2, vy, w, h, 'Voting Classifier Ensemble', 'Hard/Soft voting')
# arrows up from three classifiers
arrow(svx+small_w//2, cy+small_h, cx, vy, 'down')
arrow(xgbx+small_w//2, cy+small_h, cx, vy, 'down')
arrow(lrx+small_w//2, cy+small_h, cx, vy, 'down')

# Prediction
py2 = vy + h + 60
box(cx-w//2, py2, w, h, 'Fraud Prediction', 'Binary output + score')
arrow(cx, vy+h, cx, py2, 'down')

img.save('Enhanced_Dataflow.png')
print('Saved Enhanced_Dataflow.png')
