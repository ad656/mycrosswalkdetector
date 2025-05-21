import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import time
from collections import deque

# Parameters
FPS = 15  # Target frames per second for processing (adjust as needed)
HALF_SECOND_FRAMES = int(FPS / 2)  # Number of frames in half a second
frame_interval = 1.0 / FPS  # Time between frames in seconds

# Debug: Check webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Webcam not accessible.")
    exit()

# Debug: Check model
try:
    model = load_model('crosswalk_model.h5')
    print(f"Model loaded. Expected input shape: {model.input_shape}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    exit()

# Initialize variables for bounding box
box_x, box_y, box_w, box_h = 0, 0, 0, 0
confidence_threshold = 0.6  # Minimum confidence to display box
found_signal = False

# Create directory for saved frames if it doesn't exist
save_dir = "saved_frames"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Create a deque to store recent predictions
recent_predictions = deque(maxlen=HALF_SECOND_FRAMES)
recent_confidences = deque(maxlen=HALF_SECOND_FRAMES)
recent_labels = deque(maxlen=HALF_SECOND_FRAMES)

# Main loop
print(f"Starting webcam feed at {FPS} FPS... Press 'q' to quit, 'r' to reset box, arrow keys to adjust box.")
frame_count = 0
last_frame_time = time.time()

while True:
    current_time = time.time()
    elapsed = current_time - last_frame_time
    
    # Check if it's time to capture a new frame based on target FPS
    if elapsed >= frame_interval:
        # Capture a new frame
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame.")
            break

        # Save the frame
        frame_path = os.path.join(save_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        # Create a copy of the frame for displaying
        display_frame = frame.copy()
        
        # Processing for model input
        resized = cv2.resize(frame, (128, 128))
        normalized = resized / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)

        # Prediction for this frame
        try:
            prediction = model.predict(input_tensor, verbose=0)
            
            # Handle different prediction shapes
            if prediction.shape == (1, 1):
                # Binary classification with a single output
                pred_value = prediction[0][0]
                confidence = max(pred_value, 1 - pred_value)  # Convert to confidence score (0.0-1.0)
                label = "WALK" if pred_value < 0.5 else "DON'T WALK"
            elif prediction.shape == (1, 2):
                # Two-class output (softmax)
                class_idx = np.argmax(prediction[0])
                pred_value = prediction[0][class_idx]
                confidence = pred_value  # Softmax output is already a confidence score
                label = "WALK" if class_idx == 0 else "DON'T WALK"
            else:
                print(f"Unexpected prediction shape: {prediction.shape}")
                pred_value = 0.5  # Default
                confidence = 0.5
                label = "UNKNOWN"
            
            # Add this prediction to recent predictions
            recent_predictions.append(pred_value)
            recent_confidences.append(confidence)
            recent_labels.append(label)
            
            # Calculate average prediction over the last half second
            if len(recent_predictions) > 0:
                # For binary classification, count votes for each class
                walk_votes = recent_labels.count("WALK")
                dont_walk_votes = recent_labels.count("DON'T WALK")
                
                # Determine the final label by majority vote
                final_label = "WALK" if walk_votes > dont_walk_votes else "DON'T WALK"
                
                # Calculate average confidence
                avg_confidence = sum(recent_confidences) / len(recent_confidences)
                
                # Print debug info
                print(f"Frame {frame_count}: Current={label} ({confidence:.2f}), "
                      f"Average={final_label} ({avg_confidence:.2f}), "
                      f"Votes: WALK={walk_votes}, DON'T WALK={dont_walk_votes}")
                
                # Set color based on final prediction
                color = (0, 255, 0) if final_label == "WALK" else (0, 0, 255)
                
                # Display combined prediction info
                cv2.putText(display_frame, f"{final_label} ({avg_confidence:.2f})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Add current frame info
                cv2.putText(display_frame, f"Current: {label} ({confidence:.2f})", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Add frame count and buffer size
                cv2.putText(display_frame, f"Frames: {len(recent_predictions)}/{HALF_SECOND_FRAMES}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # If we have high confidence, attempt to locate the signal in the frame
                if avg_confidence > confidence_threshold:
                    # If we haven't found a signal yet or want to update, try to locate it
                    if not found_signal:
                        # Simple approach: use the whole frame initially
                        frame_height, frame_width = frame.shape[:2]
                        box_x, box_y = int(frame_width * 0.25), int(frame_height * 0.25)
                        box_w, box_h = int(frame_width * 0.5), int(frame_height * 0.5)
                        found_signal = True
                    
                    # Draw bounding box around the signal area
                    cv2.rectangle(display_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 2)
                    
                    # Draw confidence level next to the box
                    cv2.putText(display_frame, f"Conf: {avg_confidence:.2f}", (box_x, box_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Extract signal region for detailed analysis
                    signal_region = frame[box_y:box_y+box_h, box_x:box_x+box_w]
                    if signal_region.size > 0:  # Check if region is valid
                        # Create a small preview of the detected region
                        preview_size = (128, 128)
                        signal_preview = cv2.resize(signal_region, preview_size)
                        
                        # Place preview in top-right corner
                        preview_x = display_frame.shape[1] - preview_size[0] - 10
                        preview_y = 10
                        display_frame[preview_y:preview_y+preview_size[1], 
                                     preview_x:preview_x+preview_size[0]] = signal_preview
                        
                        # Add border to preview
                        cv2.rectangle(display_frame, (preview_x, preview_y), 
                                     (preview_x + preview_size[0], preview_y + preview_size[1]), 
                                     color, 2)
            
        except Exception as e:
            print(f"ERROR during prediction: {e}")
            cv2.putText(display_frame, f"Error: {str(e)[:20]}...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Display the processed frame
        cv2.imshow('Crosswalk Signal Detector', display_frame)
        
        # Update counters for next frame
        frame_count += 1
        last_frame_time = current_time
    
    # Check for key presses (do this regardless of frame timing)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # Quit
        break
    elif key == ord('r'):
        # Reset box
        found_signal = False
    elif key == ord('c'):
        # Clear recent predictions
        recent_predictions.clear()
        recent_confidences.clear()
        recent_labels.clear()
    elif key == 82:  # Up arrow
        box_y = max(0, box_y - 5)
    elif key == 84:  # Down arrow
        box_y = min(frame.shape[0] - box_h, box_y + 5)
    elif key == 81:  # Left arrow
        box_x = max(0, box_x - 5)
    elif key == 83:  # Right arrow
        box_x = min(frame.shape[1] - box_w, box_x + 5)
    elif key == ord('w'):  # Increase box height
        box_h = min(frame.shape[0] - box_y, box_h + 5)
    elif key == ord('s'):  # Decrease box height
        box_h = max(10, box_h - 5)
    elif key == ord('a'):  # Decrease box width
        box_w = max(10, box_w - 5)
    elif key == ord('d'):  # Increase box width
        box_w = min(frame.shape[1] - box_x, box_w + 5)
    
    # Small sleep to prevent CPU overuse when not capturing frames
    time.sleep(0.001)

cap.release()
cv2.destroyAllWindows()
print(f"Webcam released. {frame_count} frames were saved to {save_dir}/")

# You might want to clean up old frames here
# Uncomment the following to delete all saved frames:
# import shutil
# shutil.rmtree(save_dir)
# os.makedirs(save_dir)