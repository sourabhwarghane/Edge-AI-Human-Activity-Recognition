import csv
import cv2
import joblib
import numpy as np
import time
import torch
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

pose_model = YOLO("models/yolov8n-pose.pt")
activity_model = joblib.load("models/activity_classifier.joblib")

pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1640,height=1232,framerate=30/1,format=NV12 ! nvvidconv ! video/x-raw,width=640,height=480,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Camera failed to open.")
    exit()

sequence = []
last_sample_time = 0
sample_interval = 0.1
target_frames = 12
frame_count = 0
warmup_frames = 30

fps_values = []
pose_latencies = []
classifier_latencies = []

previous_time = time.perf_counter()

def extract_features(sequence):
    sequence = np.array(sequence)
    features = []

    for i in range(17):
        x = sequence[:, i * 3]
        y = sequence[:, i * 3 + 1]
        c = sequence[:, i * 3 + 2]
        features.append(np.mean(x))
        features.append(np.mean(y))
        features.append(np.std(x))
        features.append(np.std(y))
        features.append(np.mean(np.abs(np.diff(x))))
        features.append(np.mean(np.abs(np.diff(y))))
        features.append(np.mean(c))

    return np.array(features)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not received.")
        break

    frame_count += 1

    torch.cuda.synchronize()
    pose_start = time.perf_counter()
    results = pose_model(frame, device=0, verbose=False)
    torch.cuda.synchronize()
    pose_latency = (time.perf_counter() - pose_start) * 1000

    output = results[0].plot()
    classifier_latency = 0

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        person_id = areas.argmax()

        xy = results[0].keypoints.xy[person_id].cpu().numpy()
        conf = results[0].keypoints.conf[person_id].cpu().numpy()
        box = boxes[person_id]

        if time.time() - last_sample_time >= sample_interval:
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

            if len(sequence) > target_frames:
                sequence.pop(0)

            if len(sequence) == target_frames:
                features = extract_features(sequence)
                classifier_start = time.perf_counter()
                activity_model.predict_proba([features])
                classifier_latency = (time.perf_counter() - classifier_start) * 1000

                if frame_count > warmup_frames:
                    classifier_latencies.append(classifier_latency)

    current_time = time.perf_counter()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    if frame_count > warmup_frames:
        fps_values.append(fps)
        pose_latencies.append(pose_latency)

    cv2.putText(output, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(output, f"Pose: {pose_latency:.1f} ms", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(output, f"Classifier: {classifier_latency:.2f} ms", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(output, "Press Q to finish benchmark", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Jetson PyTorch Benchmark", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if len(fps_values) == 0 or len(pose_latencies) == 0:
    print("No benchmark data collected.")
    exit()

avg_fps = np.mean(fps_values)
avg_pose = np.mean(pose_latencies)
avg_classifier = np.mean(classifier_latencies) if classifier_latencies else 0

print(f"\nAverage FPS: {avg_fps:.2f}")
print(f"Average pose latency: {avg_pose:.2f} ms")
print(f"Average classifier latency: {avg_classifier:.2f} ms")

benchmark_dir = Path("benchmarks")
benchmark_dir.mkdir(exist_ok=True)
file_path = benchmark_dir / "jetson_orin_pytorch.csv"
file_exists = file_path.exists()

with open(file_path, "a", newline="") as file:
    writer = csv.writer(file)

    if not file_exists:
        writer.writerow(["date", "device", "backend", "pose_model", "classifier", "frames", "avg_fps", "avg_pose_ms", "avg_classifier_ms"])

    writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "NVIDIA Jetson Orin Nano", "PyTorch CUDA", "YOLOv8n-pose", "Random Forest", len(fps_values), round(avg_fps, 2), round(avg_pose, 2), round(avg_classifier, 2)])

print(f"\nBenchmark saved to {file_path}")