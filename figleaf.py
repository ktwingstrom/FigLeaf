import cv2
import numpy as np
from ultralytics import YOLO

# Load the locally saved model instead of downloading it each time
model = YOLO("/Users/kevintwingstrom/github/FigLeaf/models/erax-anti-nsfw-yolo11m-v1.1.pt")

def detect_and_blur(image_path):
    """Detects NSFW content and applies a circular blur using an elliptical mask,
    skipping any detections labeled 'make_love'."""
    frame = cv2.imread(image_path)
    results = model(frame)

    for result in results:
        for box in result.boxes:
            # Get bounding box coordinates
            coords = box.xyxy.squeeze().tolist()
            x1, y1, x2, y2 = map(int, coords)
            confidence = box.conf.squeeze().item()

            # Retrieve the class label
            cls_idx = int(box.cls.squeeze().item())
            label = result.names[cls_idx]
            print(f"Label: {label}, Coordinates: ({x1}, {y1}), ({x2}, {y2}) with confidence: {confidence}")

            # Skip processing for 'make_love' detections
            if label == "make_love":
                print("Skipping blur for make_love detection")
                continue

            # Validate ROI dimensions
            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]
                print(f"ROI shape: {roi.shape}")
                
                # Create a blurred version of the ROI
                blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
                
                # Create an elliptical mask for a circular blur effect
                mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
                center = (roi.shape[1] // 2, roi.shape[0] // 2)
                axes = (roi.shape[1] // 2, roi.shape[0] // 2)
                cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                
                # Blend the blurred ROI with the original ROI using the mask
                roi_masked = roi.copy()
                roi_masked[mask == 255] = blurred_roi[mask == 255]
                
                # Replace the ROI in the original frame with the blended version
                frame[y1:y2, x1:x2] = roi_masked
            else:
                print("Invalid ROI dimensions; skipping blur.")

    cv2.imwrite("blurred_output.jpg", frame)
    print("NSFW content detected and blurred.")

# Test with an image
detect_and_blur("test_image.jpg")
