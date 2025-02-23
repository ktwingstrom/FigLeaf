import cv2
import numpy as np
from ultralytics import YOLO
import torch
import concurrent.futures
import ffmpeg

# Change path for production; this is only for testing.
model = YOLO("/Users/kevintwingstrom/github/FigLeaf/models/erax-anti-nsfw-yolo11m-v1.1.pt")
# Use Apple’s MPS if available (for Apple Silicon)
if torch.backends.mps.is_available():
    model.to("mps")

def process_batch(frames):
    """Processes a batch of frames using YOLO detection and applies elliptical blur,
    skipping detections labeled 'make_love'."""
    results = model(frames)  # Batch inference
    # Process each frame's results
    for i, result in enumerate(results):
        for box in result.boxes:
            # Convert bounding box tensor to list of coordinates
            coords = box.xyxy.squeeze().tolist()
            # Flatten if the coordinates are nested
            if isinstance(coords[0], list):
                coords = coords[0]
            x1, y1, x2, y2 = map(int, coords)
            confidence = box.conf.squeeze().item()

            # Retrieve the class label from the model's names mapping
            cls_idx = int(box.cls.squeeze().item())
            label = result.names[cls_idx]

            # Skip detection labeled "make_love"
            if label == "make_love":
                continue

            # Validate ROI dimensions
            if x2 > x1 and y2 > y1:
                # Extract the region of interest from the current frame
                roi = frames[i][y1:y2, x1:x2]
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

                # Replace the ROI in the frame with the blended version
                frames[i][y1:y2, x1:x2] = roi_masked
    return frames

def process_video(input_video, output_video, batch_size=32, max_workers=10):
    """Detects and blurs NSFW content in a video using concurrent batch processing
    and an elliptical mask, skipping any detections labeled 'make_love'."""
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Setup VideoWriter with the same properties as the input video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    batch_futures = []   # List to store futures along with their batch index
    frames_batch = []    # Current batch of frames
    batch_index = 0      # To preserve order

    # Use ThreadPoolExecutor to process multiple batches concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_batch.append(frame)

            # Once we have a full batch, submit it for processing
            if len(frames_batch) == batch_size:
                future = executor.submit(process_batch, frames_batch.copy())
                batch_futures.append((batch_index, future))
                batch_index += 1
                frames_batch = []  # Reset batch

        # Process any remaining frames as a final batch
        if frames_batch:
            future = executor.submit(process_batch, frames_batch.copy())
            batch_futures.append((batch_index, future))

    cap.release()

    # Ensure batches are written in order
    batch_futures.sort(key=lambda x: x[0])
    for idx, future in batch_futures:
        processed_frames = future.result()
        for pf in processed_frames:
            out.write(pf)
    out.release()
    print("Processing complete. Output saved as", output_video)

def mux_audio(processed_video, input_video, final_output):
    """Uses ffmpeg to mux the audio from the original input video into the processed video without re-encoding."""
    video_input = ffmpeg.input(processed_video)
    audio_input = ffmpeg.input(input_video)
    try:
        (
            ffmpeg
            .output(video_input, audio_input, final_output, vcodec="copy", acodec="copy")
            .global_args('-map', '0:v:0', '-map', '1:a:0')
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
        print("Audio muxed successfully into", final_output)
    except ffmpeg.Error as e:
        print("FFmpeg error:", e.stderr.decode('utf-8'))
        raise

def main():
    input_video = "test_video.mp4"
    processed_video = "output_blurred.mp4"
    final_output = "final_output.mp4"
    
    # Process video frames and apply blurring
    process_video(input_video, processed_video, batch_size=32, max_workers=4)
    # Mux the original audio from input_video into the processed video
    mux_audio(processed_video, input_video, final_output)

if __name__ == "__main__":
    main()
