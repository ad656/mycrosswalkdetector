import cv2
import numpy as np
import torch
import time
from pathlib import Path
from tensorflow.keras.models import load_model
from models.common import DetectMultiBackend
from utils.general import non_max_suppression

# --- Settings ---
SAVE_DIR = Path("captured_frames")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_INTERVAL = 0.5  # seconds (2 frames per second)
FRAME_SIZE = (256, 256)

# Load YOLOv5 model
yolo_weights = 'runs/train/crosswalk_detector3/weights/best.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DetectMultiBackend(yolo_weights, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = 640

# Load classifier
classifier = load_model('primitive.h5')
print(f"Classifier loaded. Input shape: {classifier.input_shape}")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Starting webcam. Press 'q' to quit.")
frame_count = 0
last_capture_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    if current_time - last_capture_time >= SAVE_INTERVAL:
        last_capture_time = current_time

        original_frame = frame.copy()
        img = cv2.resize(frame, (imgsz, imgsz))
        img_tensor = torch.from_numpy(img).to(device).permute(2, 0, 1).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # YOLOv5 inference
        pred = model(img_tensor, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

        if pred is not None and len(pred) and pred.ndim == 2:
            for *xyxy, conf, cls in reversed(pred):
                x1, y1, x2, y2 = map(int, xyxy)
                signal_region = original_frame[y1:y2, x1:x2]

                if signal_region.size > 0:
                    resized = cv2.resize(signal_region, FRAME_SIZE)
                    normalized = resized / 255.0
                    input_tensor = np.expand_dims(normalized, axis=0)

                    pred = classifier.predict(input_tensor)[0]
                    if len(pred) == 2:
                        class_idx = np.argmax(pred)
                        final_label = "WALK" if class_idx == 0 else "DON'T WALK"
                        confidence = pred[class_idx]
                    else:
                        confidence = pred[0]
                        final_label = "WALK" if confidence < 0.5 else "DON'T WALK"
                        confidence = max(confidence, 1 - confidence)

                    color = (0, 255, 0) if final_label == "WALK" else (0, 0, 255)
                    cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(original_frame, f"{final_label} ({confidence:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save processed image
        filename = SAVE_DIR / f"frame_{frame_count:04d}.jpg"
        cv2.imwrite(str(filename), original_frame)
        print(f"Saved: {filename}")
        frame_count += 1

    # Display preview
    cv2.imshow("Crosswalk Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
