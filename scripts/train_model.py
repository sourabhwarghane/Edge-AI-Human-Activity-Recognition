from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

data_dir = Path("data/raw")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

activities = ["standing", "walking", "sitting", "falling"]


def extract_features(file):
    df = pd.read_csv(file)
    features = []

    for i in range(17):
        x = df[f"x{i}"].values
        y = df[f"y{i}"].values
        c = df[f"c{i}"].values

        features.append(np.mean(x))
        features.append(np.mean(y))
        features.append(np.std(x))
        features.append(np.std(y))
        features.append(np.mean(np.abs(np.diff(x))))
        features.append(np.mean(np.abs(np.diff(y))))
        features.append(np.mean(c))

    return features


X = []
y = []

for activity in activities:
    files = list((data_dir / activity).glob("*.csv"))

    for file in files:
        X.append(extract_features(file))
        y.append(activity)

X = np.array(X)
y = np.array(y)

print("Sequences:", len(X))
print("Features per sequence:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, model_dir / "activity_classifier.joblib")
print("\nModel saved to models/activity_classifier.joblib")
