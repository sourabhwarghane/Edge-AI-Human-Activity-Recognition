import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

data_dir = Path("data/raw")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

activities = ["standing", "walking", "sitting", "falling"]

torch.manual_seed(42)
np.random.seed(42)

X = []
y = []

for label_id, activity in enumerate(activities):
    files = sorted((data_dir / activity).glob("*.csv"))

    for file in files:
        df = pd.read_csv(file)
        sequence = []

        for i in range(17):
            sequence.append(df[f"x{i}"].values)
            sequence.append(df[f"y{i}"].values)
            sequence.append(df[f"c{i}"].values)

        sequence = np.array(sequence).T
        X.append(sequence)
        y.append(label_id)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("Sequences:", len(X))
print("Sequence shape:", X.shape[1:])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


class ActivityLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=51, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 4)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


model = ActivityLSTM().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")

model.eval()

with torch.no_grad():
    outputs = model(X_test.to(device))
    predictions = torch.argmax(outputs, dim=1).cpu().numpy()

y_test_np = y_test.numpy()

print("\nAccuracy:", round(accuracy_score(y_test_np, predictions), 3))
print("\nClassification Report:")
print(classification_report(y_test_np, predictions, target_names=activities))
print("Confusion Matrix:")
print(confusion_matrix(y_test_np, predictions))

torch.save(model.state_dict(), model_dir / "activity_lstm.pth")

with open(model_dir / "activity_labels.json", "w") as file:
    json.dump(activities, file)

print("\nModel saved to models/activity_lstm.pth")
