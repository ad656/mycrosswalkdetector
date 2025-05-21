import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
MODEL_PATH = 'crosswalk_model.h5'  # Path to your trained model
IMAGE_DIR = 'Raw_data/validation/dont_walk'  # Directory containing images to predict
IMAGE_SIZE = (128, 128)  # Must match training size
OUTPUT_DIR = 'predictions'  # Where to save results
THRESHOLD = 0.5  # Confidence threshold

# Load model
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully. Input shape: {model.input_shape}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def predict_image(img_path):
    """Predict single image and return label and confidence"""
    try:
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array, verbose=0)[0][0]
        confidence = abs(prediction - 0.5) * 2  # Convert to 0-1 scale
        
        if prediction < THRESHOLD:
            return "walk", 1 - prediction, img_array[0]
        else:
            return "don't walk", prediction, img_array[0]
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, None, None

# Process all images in directory
results = []
for filename in os.listdir(IMAGE_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(IMAGE_DIR, filename)
        label, confidence, img = predict_image(img_path)
        
        if label:
            # Save annotated image
            output_img = (img * 255).astype('uint8')
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            
            # Add prediction text
            color = (0, 255, 0) if label == "walk" else (0, 0, 255)
            text = f"{label} ({confidence:.2f})"
            cv2.putText(output_img, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Save result
            output_path = os.path.join(OUTPUT_DIR, f"pred_{filename}")
            cv2.imwrite(output_path, output_img)
            
            results.append((filename, label, confidence))
            print(f"{filename}: {label} (confidence: {confidence:.2f})")

# Generate summary report
if results:
    report_path = os.path.join(OUTPUT_DIR, "predictions_report.txt")
    with open(report_path, 'w') as f:
        f.write("Image,Prediction,Confidence\n")
        for filename, label, confidence in results:
            f.write(f"{filename},{label},{confidence:.4f}\n")
    
    print(f"\nProcessed {len(results)} images. Results saved to {OUTPUT_DIR}")
    print(f"Summary report: {report_path}")
else:
    print("No valid images found in directory.")