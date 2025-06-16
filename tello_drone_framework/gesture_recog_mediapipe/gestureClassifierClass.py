import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import os
import csv

class GestureClassifier:
    """
    Classe pour entraîner et tester un classificateur de gestes à partir d'un fichier CSV.

    Attributs:
        save_mode (str): True to save the model, False to test it.
    """

    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.input_dim = None
        self.num_classes = None

    def load_data(self, path="gestures.csv"):

        path = os.path.join("data/", path)

        df = pd.read_csv(path)
        X = df.drop(columns=["label"]).values
        y = df["label"].values

        y_encoded = self.label_encoder.fit_transform(y)
        self.num_classes = len(self.label_encoder.classes_)
        self.input_dim = X.shape[1]
        print(f"Number of classes : {self.num_classes}")

        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42
        )

        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
        train_tensor = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )

        self.train_loader = DataLoader(train_tensor, shuffle=True)

    def sequential_model(self):
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes)
        )

    def train(self, epochs=3000, lr=0.001, print_every=100):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            for X_batch, y_batch in self.train_loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % print_every == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    def evaluate(self):
        if self.model is None:
            raise ValueError("The model is not loaded. Please load the model before recognizing gestures.")
        
        self.model.eval()
        with torch.no_grad():
            preds = self.model(self.X_test_tensor)
            predicted_classes = torch.argmax(preds, dim=1).numpy()
            acc = accuracy_score(self.y_test_tensor.numpy(), predicted_classes)
            print(f"\nAccuracy : {acc * 100:.2f}%")

    def save(self, path="gesture_bundle.pt"):
        saved_path = os.path.join("models/", path)
        torch.save({
            "model_state": self.model.state_dict(),
            "label_classes": self.label_encoder.classes_,
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes
        }, saved_path)
        print(f"Saved at : {saved_path}")

    def load(self, path="gesture_bundle.pt"):
        load_path = os.path.join("models/", path)
        checkpoint = torch.load(load_path, weights_only=False)

        self.input_dim = checkpoint["input_dim"]
        self.num_classes = checkpoint["num_classes"]

        self.sequential_model()
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.label_encoder.classes_ = checkpoint["label_classes"]
        self.scaler.mean_ = checkpoint["scaler_mean"]
        self.scaler.scale_ = checkpoint["scaler_scale"]

        print(f"Model charged from {load_path}")

    @staticmethod
    def count_label_rows(file_path, target_label):
        count = 0
        with open(file_path, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["label"] == target_label:
                    count += 1
        return count

    @staticmethod
    def collect_hand_data(mode="new", camera_id=0, output_path="gesture.csv", label="unknown"):
        """
        Collects hand landmarks from camera and saves them to a CSV file with labels.

        Args:
            mode (str): "new" = error if file exists, "append" = add to existing file, "empty" = overwrite.
            camera_id (int): Camera device ID.
            output_path (str): Path to the CSV file to save data.
            label (str): The label/class name for the gesture being recorded.
        """

        output_path = os.path.join("data/", output_path)

        try:
            cap = cv2.VideoCapture(camera_id)
        except Exception as e:
            print(f"Error opening the camera with id {camera_id} : {e}")
            return
        

        if mode not in ["new", "append", "empty"]:
            raise ValueError("collect_hand_data : Mode must be 'new', 'append' or 'empty'")

        file_exists = os.path.exists(output_path)

        if mode == "new" and file_exists:
            raise FileExistsError(f"File '{output_path}' already exists. Use 'append' or 'empty' instead.")
        elif mode == "empty":
            with open(output_path, "w") as f:
                writer = csv.writer(f)
                header = [f"{coord}_{i}" for i in range(21) for coord in ["x", "y", "z"]]
                header.append("label")
                writer.writerow(header)
            print(f"File emptied and ready : {output_path}")
        elif mode == "new" or (mode == "append" and not file_exists):
            with open(output_path, "w") as f:
                writer = csv.writer(f)
                header = [f"{coord}_{i}" for i in range(21) for coord in ["x", "y", "z"]]
                header.append("label")
                writer.writerow(header)
            print(f"New File initialized : {output_path}")
        else:
            print(f"Add to an existing file : {output_path}")
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False,
                            max_num_hands=1,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)
        mp_draw = mp.solutions.drawing_utils

        print("-----------------------------------------------------------------------------------")
        print("|  You're in collect_hand_data mode. Press 'Space' to save a vector, 'q' to quit  |")
        print("-----------------------------------------------------------------------------------")

        if mode == "append" and file_exists:
            nb = GestureClassifier.count_label_rows(output_path, label)
            print(f"Number of already existing rows for label '{label}': {nb}")
        else:
            nb = 0
            print(f"No existing rows for label '{label}'")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error collect_hand_data : Could not read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"Label: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Data collecting", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                row = []
                for lm in landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                row.append(label)
                with open(output_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                nb += 1
                print(f"Data number {nb} saved successfully.")
                

            elif key == ord('q'):
                print("Collect hand data exited.")
                break

        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def print_hand_data(path="gesture.csv"):
        """
        Prints the hand data from the CSV file.
        
        Args:
            path (str): Path to the CSV file containing hand data.
        """
        print_path = os.path.join("data/", path)

        try:
            df = pd.read_csv(print_path)
            print(df["label"].value_counts())
        except FileNotFoundError:
            print(f"File not found: {print_path}")
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")

    def recognize_from_cam(self,seuil=0.99, temperature=2.0, camera_id=0):
        if self.model is None:
            raise ValueError("The model is not loaded. Please load the model before recognizing gestures.")
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_draw = mp.solutions.drawing_utils

        try:
            cap = cv2.VideoCapture(camera_id)
        except Exception as e:
            print(f"Error opening the camera with id {camera_id} : {e}")
            return
        

        print("Detection... press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture = ""
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                hand = results.multi_hand_landmarks[0]
                data = []
                for lm in hand.landmark:
                    data.extend([lm.x, lm.y, lm.z])
                X_input = np.array(data).reshape(1, -1)
                X_scaled = self.scaler.transform(X_input)

                with torch.no_grad():
                    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                    output = self.model(X_tensor)
                    probs = torch.softmax(output / temperature, dim=1)
                    conf_value, predicted = torch.max(probs, dim=1)
                    conf_value = conf_value.item()
                    predicted = predicted.item()

                    if conf_value >= seuil:
                        gesture = self.label_encoder.inverse_transform([predicted])[0]
                    else:
                        gesture = "Aucun geste"

                cv2.putText(frame, f"{gesture} ({conf_value:.2f})", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.imshow("Gesture recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

 