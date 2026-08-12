import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Choose input
# 0 = webcam
# "videos/test.mp4" = video file
SOURCE = "videos/test.mp4"

cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video ended or frame could not be read.")
        break

    # Object detection + tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # Get result
    result = results[0]

    # Draw boxes and tracking IDs
    annotated_frame = result.plot()

    # Count detected objects
    object_count = len(result.boxes)

    # Display object count
    cv2.putText(
        annotated_frame,
        f"Objects: {object_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display frame
    cv2.imshow("Object Detection and Tracking", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()