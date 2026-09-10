import cv2
import joblib
import numpy as np
import time
from ultralytics import YOLO

pose_model = YOLO("models/yolov8n-pose.pt")
activity_model = joblib.load("models/activity_classifier.joblib")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

sequence = []
probability_history = []
last_sample_time = 0
sample_interval = 0.1
target_frames = 20
smooth_predictions = 5

activity = "Collecting poses..."
confidence = 0
classifier_latency = 0
previous_time = time.time()


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
        break

    pose_start = time.perf_counter()
    results = pose_model(frame, device=0, verbose=False)
    pose_latency = (time.perf_counter() - pose_start) * 1000

    output = results[0].plot()

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
                probabilities = activity_model.predict_proba([features])[0]
                classifier_latency = (
                    time.perf_counter() - classifier_start) * 1000

                probability_history.append(probabilities)

                if len(probability_history) > smooth_predictions:
                    probability_history.pop(0)

                average_probabilities = np.mean(probability_history, axis=0)
                activity = activity_model.classes_[
                    np.argmax(average_probabilities)]
                confidence = np.max(average_probabilities)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(output, f"Activity: {activity}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if len(sequence) == target_frames:
        cv2.putText(output, f"Confidence: {confidence * 100:.1f}%",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(output, f"Buffer: {len(sequence)}/{target_frames}",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 150), 2)

    cv2.putText(output, f"FPS: {fps:.1f}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 150), 2)
    cv2.putText(output, f"Pose: {pose_latency:.1f} ms", (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 150), 2)
    cv2.putText(output, f"Classifier: {classifier_latency:.2f} ms", (
        20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 150), 2)

    cv2.imshow("Edge AI Human Activity Recognition", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
