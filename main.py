import cv2
import time
from ultralytics import YOLO

# ==============================
# 1. Load YOLO model
# ==============================
model = YOLO("yolo11n.pt")

# ==============================
# 2. Choose input
# ==============================
# 0 = webcam
# "videos/test.mp4" = video file
SOURCE = "videos/test.mp4"

# ==============================
# 3. Open video source
# ==============================
cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# ==============================
# 4. Get video properties
# ==============================
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
input_fps = cap.get(cv2.CAP_PROP_FPS)

# If FPS information is unavailable
if input_fps <= 0:
    input_fps = 30

# ==============================
# 5. Create output video
# ==============================
output_path = "outputs/tracked_output.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    input_fps,
    (width, height)
)

# ==============================
# 6. Processing statistics
# ==============================
frame_count = 0
start_time = time.time()

print("Processing started...")
print(f"Input: {SOURCE}")
print(f"Output: {output_path}")

# ==============================
# 7. Process video
# ==============================
while True:

    ret, frame = cap.read()

    if not ret:
        print("Video processing completed.")
        break

    frame_count += 1

    # --------------------------
    # YOLO detection + tracking
    # --------------------------
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    # --------------------------
    # Draw bounding boxes
    # --------------------------
    annotated_frame = result.plot()

    # --------------------------
    # Current frame object count
    # --------------------------
    object_count = len(result.boxes)

    # --------------------------
    # Calculate FPS
    # --------------------------
    elapsed_time = time.time() - start_time

    if elapsed_time > 0:
        fps = frame_count / elapsed_time
    else:
        fps = 0

    # --------------------------
    # Display information
    # --------------------------
    cv2.putText(
        annotated_frame,
        f"Objects: {object_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Frame: {frame_count}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # --------------------------
    # Save processed frame
    # --------------------------
    out.write(annotated_frame)

    # --------------------------
    # Display
    # --------------------------
    cv2.imshow(
        "Object Detection and Tracking",
        annotated_frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Processing stopped by user.")
        break

# ==============================
# 8. Release resources
# ==============================
cap.release()
out.release()
cv2.destroyAllWindows()

# ==============================
# 9. Final statistics
# ==============================
total_time = time.time() - start_time

if total_time > 0:
    average_fps = frame_count / total_time
else:
    average_fps = 0

print("\n========== PROCESSING SUMMARY ==========")
print(f"Total frames processed: {frame_count}")
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Average FPS: {average_fps:.2f}")
print(f"Output saved to: {output_path}")
print("========================================")