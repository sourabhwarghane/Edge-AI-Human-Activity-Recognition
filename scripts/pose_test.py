import cv2
import time
from ultralytics import YOLO

model = YOLO("models/yolov8n-pose.pt")
cap = cv2.VideoCapture(0)

previous_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera frame not received.")
        break

    results = model(frame, verbose=False)
    output = results[0].plot()

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(output, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        keypoints = results[0].keypoints.xy[0]
        cv2.putText(output, f"Keypoints: {len(keypoints)}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Pose Test", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
