import cv2
import numpy as np
from ultralytics import YOLO

# Load the locally saved model instead of downloading it each time
model = YOLO("/Users/kevintwingstrom/github/FigLeaf/models/640m.pt")
model2 = YOLO("/Users/kevintwingstrom/github/FigLeaf/models/erax-anti-nsfw-yolo11m-v1.1.pt")

def detect_and_blur(image_path):
    """Detects NSFW content and applies a circular pixelation using an elliptical mask,
    skipping any detections labeled 'make_love'."""
    frame = cv2.imread(image_path)
    results = model(frame)
    results2 = model2(frame)

    for results_set in [results, results2]:
        for result in results_set:
            for box in result.boxes:
                coords = box.xyxy.squeeze().tolist()
                x1, y1, x2, y2 = map(int, coords)
                confidence = box.conf.squeeze().item()

                cls_idx = int(box.cls.squeeze().item())
                label = result.names[cls_idx]
                print(f"Label: {label}, Coordinates: ({x1}, {y1}), ({x2}, {y2}) with confidence: {confidence}")

                if label == "make_love":
                    print("Skipping blur for make_love detection")
                    continue

                if x2 > x1 and y2 > y1:
                    roi = frame[y1:y2, x1:x2]
                    print(f"ROI shape: {roi.shape}")
                    
                    # Create pixelation effect by scaling down and up
                    height, width = roi.shape[:2]
                    pixel_size = min(20, min(width, height))  # Ensure pixel_size doesn't exceed ROI dimensions
                    
                    # Ensure minimum dimensions of 2 pixels
                    new_width = max(2, width // pixel_size)
                    new_height = max(2, height // pixel_size)
                    
                    # Scale down
                    temp = cv2.resize(roi, (new_width, new_height),
                                    interpolation=cv2.INTER_LINEAR)
                    
                    # Scale up
                    pixelated_roi = cv2.resize(temp, (width, height),
                                             interpolation=cv2.INTER_NEAREST)
                    
                    # Create an elliptical mask for a circular effect
                    mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
                    center = (roi.shape[1] // 2, roi.shape[0] // 2)
                    axes = (roi.shape[1] // 2, roi.shape[0] // 2)
                    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                    
                    # Blend the pixelated ROI with the original ROI using the mask
                    roi_masked = roi.copy()
                    roi_masked[mask == 255] = pixelated_roi[mask == 255]
                    
                    frame[y1:y2, x1:x2] = roi_masked
                else:
                    print("Invalid ROI dimensions; skipping blur.")

    cv2.imwrite("blurred_output.jpg", frame)
    print("NSFW content detected and pixelated.")

# Test with an image
detect_and_blur("test_image.jpg")
