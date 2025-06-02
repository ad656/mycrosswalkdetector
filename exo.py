import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
#from pipeline import load_enhanced_crosswalk_model
# Configuration
MODEL_PATH = 'crosswalk_model_cropped.h5'
IMAGE_DIR = 'crosswalk_crops/resized_crops_100x100'
IMAGE_SIZE = (100, 100)  # Match training size
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

def detect_signal_box(image, label, debug=False):
    """
    Detects the signal light in the image using HSV color filtering and contour analysis.
    Now with tighter thresholds and shape filtering.
    """
    h, w, _ = image.shape

    # Adjust ROI: top-middle region where signals usually appear
    roi_x1 = w // 3
    roi_x2 = 2 * w // 3
    roi_y1 = 0
    roi_y2 = h // 2
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    if label == "walk":
        # Tighter white detection
        lower = np.array([0, 0, 200])
        upper = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower, upper)
    else:
        # Red detection using two ranges (red wraps in HSV)
        lower1 = np.array([0, 120, 120])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 120, 120])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

    # Optional debug view
    if debug:
        cv2.imshow("Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small and weird-shaped blobs
    filtered = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        x, y, w_box, h_box = cv2.boundingRect(c)
        aspect_ratio = w_box / h_box if h_box > 0 else 0
        if 0.4 < aspect_ratio < 1.5:  # typical square-ish light shape
            filtered.append((x, y, w_box, h_box))

    if not filtered:
        return None

    # Choose the largest remaining contour
    x, y, w_box, h_box = max(filtered, key=lambda b: b[2] * b[3])

    # Adjust coordinates back to original image space
    return (x + roi_x1, y + roi_y1, w_box, h_box)


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
