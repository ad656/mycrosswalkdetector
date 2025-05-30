import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
#from pipeline import load_enhanced_crosswalk_model
# Configuration
MODEL_PATH = 'crosswalk_model.h5'
IMAGE_DIR = 'primitive/dont_walk'
IMAGE_SIZE = (256, 256)  # Match training size
OUTPUT_DIR = 'predictions'
THRESHOLD = 0.5

# Load model
model = load_model(MODEL_PATH)
#model = load_enhanced_crosswalk_model('enhanced_shape_crosswalk_model_final.h5')
print(f"Model loaded. Input shape: {model.input_shape}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def predict_image(img_path):
    """Run classifier on resized image, return prediction and original image"""
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize(IMAGE_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0][0]
    label = "walk" if prediction < THRESHOLD else "don't walk"
    confidence = 1 - prediction if label == "walk" else prediction

    return label, confidence, np.array(img)

def detect_signal_box(image, label):
    """
    Narrow down detection using color filters:
    - For 'walk': white
    - For 'don't walk': red/orange
    """
    # Focus on top-center region (adjust if needed)
    h, w, _ = image.shape
    roi = image[0:h//2, w//4:3*w//4]  # crop region of interest

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    if label == "walk":
        # White detection
        lower = np.array([0, 0, 180])
        upper = np.array([180, 50, 255])
    else:
        # Red detection (in HSV, red wraps around)
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 100, 100])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    if label == "walk":
        mask = cv2.inRange(hsv, lower, upper)

    # Find contours in filtered mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 100]

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(largest)

    # Adjust box to original image coordinates
    return (x + w//4, y, w_box, h_box)

# Process images
results = []
for filename in os.listdir(IMAGE_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(IMAGE_DIR, filename)
        label, confidence, orig_img = predict_image(img_path)

        output_img = orig_img.copy()
        box = detect_signal_box(output_img, label)
        color = (0, 255, 0) if label == "walk" else (0, 0, 255)
        text = f"{label} ({confidence:.2f})"

        # Draw box if found
        if box:
            x, y, w, h = box
            cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output_img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            # No box, still add text at top-left
            cv2.putText(output_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Save annotated image
        save_path = os.path.join(OUTPUT_DIR, f"pred_{filename}")
        cv2.imwrite(save_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
        results.append((filename, label, confidence))
        print(f"{filename}: {label} ({confidence:.2f})")

# Save report
if results:
    with open(os.path.join(OUTPUT_DIR, "predictions_report.txt"), 'w') as f:
        f.write("Image,Prediction,Confidence\n")
        for name, label, conf in results:
            f.write(f"{name},{label},{conf:.4f}\n")
    print(f"\nDone! Results saved in: {OUTPUT_DIR}")
else:
    print("No valid images found.")
