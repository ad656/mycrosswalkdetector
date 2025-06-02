import cv2
import numpy as np
import torch
from tensorflow.keras.models import load_model
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from pathlib import Path

# Paths
input_directory = "Raw_data/images/training"  # Directory containing input images
output_directory = "crosswalk_dataset/output_images"  # Directory to save output images
yolo_weights = "runs/train/crosswalk_detector3/weights/best.pt"
classifier_path = "crosswalk_model_cropped.h5"

# YOLOv5 setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DetectMultiBackend(yolo_weights, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = 640

# Load classifier
classifier = load_model(classifier_path)
print("Classifier loaded.")

# Ensure the output directory exists
Path(output_directory).mkdir(parents=True, exist_ok=True)

# Process each image in the input directory (handling both .png and .jpg)
image_paths = list(Path(input_directory).glob("*.png")) + list(Path(input_directory).glob("*.jpg"))

for image_path in image_paths:
    print(f"Processing image: {image_path}")

    # Load image
    original_img = cv2.imread(str(image_path))
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
        pred = pred[pred[:, 4].argmax()]  # Pick the top-1 detection
        x1, y1, x2, y2, conf, cls = map(int, pred[:6])

        # Scale back coordinates (if original was resized)
        scale_x = original_img.shape[1] / imgsz
        scale_y = original_img.shape[0] / imgsz
        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)

        # Crop detected region
        detected_region = original_img[y1:y2, x1:x2]
        if detected_region.size == 0:
            print("Detected region is empty!")
        else:
            # Preprocess for classifier
            resized = cv2.resize(detected_region, (100, ))
            normalized = resized / 255.0
            input_tensor = np.expand_dims(normalized, axis=0)

            # Classify
            prediction = classifier.predict(input_tensor)[0]
            if len(prediction) == 2:
                class_idx = np.argmax(prediction)
                confidence = prediction[class_idx]
                final_label = "WALK" if class_idx == 0 else "DON'T WALK"
            else:
                confidence = prediction[0]
                final_label = "WALK" if confidence < 0.5 else "DON'T WALK"
                confidence = max(confidence, 1 - confidence)

            # Annotate image
            color = (0, 255, 0) if final_label == "WALK" else (0, 0, 255)
            cv2.rectangle(original_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(original_img, f"{final_label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save the modified image to the output directory
        output_image_path = Path(output_directory) / image_path.name
        cv2.imwrite(str(output_image_path), original_img)

    else:
        print(f"No crosswalk signal detected in {image_path}")

print("Processing complete.")
