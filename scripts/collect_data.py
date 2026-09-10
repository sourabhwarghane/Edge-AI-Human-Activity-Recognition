import csv
import cv2
import time
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

activities = ["standing", "walking", "sitting", "falling"]

print("Activities:")
for i, activity in enumerate(activities, 1):
    print(f"{i}. {activity}")

choice = int(input("Select activity (1-4): "))
label = activities[choice - 1]

model = YOLO("models/yolov8n-pose.pt")
cap = cv2.VideoCapture(0)

save_dir = Path("data/raw") / label
save_dir.mkdir(parents=True, exist_ok=True)

sequence = []
recording = False
last_sample_time = 0
saved_count = 0
target_frames = 20
sample_interval = 0.1

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)
    output = results[0].plot()

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        person_id = areas.argmax()

        xy = results[0].keypoints.xy[person_id].cpu().numpy()
        conf = results[0].keypoints.conf[person_id].cpu().numpy()
        box = boxes[person_id]

        if recording and time.time() - last_sample_time >= sample_interval:
            width = max(box[2] - box[0], 1)
            height = max(box[3] - box[1], 1)
            pose = []

            for (x, y), c in zip(xy, conf):
                if c < 0.2:
                    pose.extend([0, 0, c])
                else:
                    nx = (x - box[0]) / width
                    ny = (y - box[1]) / height
                    pose.extend([nx, ny, c])

            sequence.append(pose)
            last_sample_time = time.time()

    cv2.putText(output, f"Activity: {label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(output, f"Saved: {saved_count}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if recording:
        cv2.putText(output, f"Recording: {len(sequence)}/{target_frames}",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(output, "Press R to record | Q to quit", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if len(sequence) == target_frames:
        filename = save_dir / \
            f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        header = ["frame"]

        for i in range(17):
            header.extend([f"x{i}", f"y{i}", f"c{i}"])

        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)

            for i, pose in enumerate(sequence):
                writer.writerow([i] + pose)

        saved_count += 1
        sequence = []
        recording = False
        print(f"Saved: {filename}")

    cv2.imshow("Activity Data Collection", output)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r") and not recording:
        sequence = []
        recording = True
        last_sample_time = 0

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
