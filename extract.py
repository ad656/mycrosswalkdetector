import cv2
import numpy as np
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from pathlib import Path
from tensorflow.keras.models import load_model
import os
from PIL import Image
# Paths
input_directory = "crosswalk_dataset"  # Directory containing input images
output_directory = "crosswalk_crops"  # Directory to save cropped images
yolo_weights = "runs/train/crosswalk_detector3/weights/best.pt"

# YOLOv5 setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DetectMultiBackend(yolo_weights, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = 640

print(f"YOLOv5 model loaded on {device}")
print(f"Processing images from: {input_directory}")
print(f"Saving cropped images to: {output_directory}")

# Ensure the output directory exists
Path(output_directory).mkdir(parents=True, exist_ok=True)

# Create subdirectories for different crop types
original_crops_dir = Path(output_directory) / "original_crops"
resized_crops_dir = Path(output_directory) / "resized_crops_100x100"
original_crops_dir.mkdir(parents=True, exist_ok=True)
resized_crops_dir.mkdir(parents=True, exist_ok=True)

# Process each image in the input directory (handling both .png and .jpg)
image_paths = list(Path(input_directory).glob("*.png")) + list(Path(input_directory).glob("*.jpg"))
processed_count = 0
crop_count = 0

for image_path in image_paths:
    print(f"Processing image: {image_path.name}")
    
    # Load image
    original_img = cv2.imread(str(image_path))
    if original_img is None:
        print(f"  Error: Could not load image {image_path}")
        continue
    
    img = cv2.resize(original_img, (imgsz, imgsz))
    img_tensor = torch.from_numpy(img).to(device).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    
    # YOLOv5 inference
    with torch.no_grad():
        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]
    
    # Proceed only if detection occurred
    if pred is not None and len(pred) > 0:
        # Get highest confidence detection
        best_detection = pred[pred[:, 4].argmax()]  # Pick the top-1 detection
        x1, y1, x2, y2, conf, cls = best_detection[:6]
        
        # Scale back coordinates to original image size
        scale_x = original_img.shape[1] / imgsz
        scale_y = original_img.shape[0] / imgsz
        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, original_img.shape[1] - 1))
        y1 = max(0, min(y1, original_img.shape[0] - 1))
        x2 = max(0, min(x2, original_img.shape[1] - 1))
        y2 = max(0, min(y2, original_img.shape[0] - 1))
        
        # Crop detected region
        detected_region = original_img[y1:y2, x1:x2]
        
        if detected_region.size > 0 and detected_region.shape[0] > 10 and detected_region.shape[1] > 10:
            # Generate base filename from original image name (without extension)
            base_name = image_path.stem
            
            # Save original crop
            original_crop_path = original_crops_dir / f"{base_name}_crop_{crop_count:04d}.jpg"
            cv2.imwrite(str(original_crop_path), detected_region)
            
            # Save resized crop (100x100 as in your classifier preprocessing)
            resized = cv2.resize(detected_region, (100, 100))
            resized_crop_path = resized_crops_dir / f"{base_name}_crop_{crop_count:04d}.jpg"
            cv2.imwrite(str(resized_crop_path), resized)
            
            print(f"  ✓ Saved crop {crop_count:04d}: {detected_region.shape} -> (100,100)")
            print(f"    Confidence: {conf:.3f}, Coords: ({x1},{y1}) to ({x2},{y2})")
            
            crop_count += 1
        else:
            print(f"  ✗ Detected region too small or empty")
    else:
        print(f"  ✗ No crosswalk signal detected")
    
    processed_count += 1

print(f"\nProcessing complete!")
print(f"Images processed: {processed_count}")
print(f"Crosswalk crops saved: {crop_count}")
print(f"Original crops saved to: {original_crops_dir}")
print(f"Resized crops (100x100) saved to: {resized_crops_dir}")

if crop_count > 0:
    print(f"\nSuccess rate: {crop_count/processed_count*100:.1f}%")
else:
    print("\nNo crops were extracted. Check your YOLOv5 model and input images.")

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
