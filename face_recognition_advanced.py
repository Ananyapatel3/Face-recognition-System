import cv2
import os
import numpy as np
import pickle

class FaceRecognizer:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.labels = {}
        self.label_ids = {}

    def train(self, data_dir):
        print("📂 Training...")

        x_train = []
        y_labels = []
        current_id = 0

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):

                    path = os.path.join(root, file)
                    label = os.path.basename(root)

                    if label not in self.label_ids:
                        self.label_ids[label] = current_id
                        self.labels[current_id] = label
                        current_id += 1

                    id_ = self.label_ids[label]

                    img = cv2.imread(path)
                    if img is None:
                        continue

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # 🔥 IMPORTANT: use full image (no face detection)
                    roi = cv2.resize(gray, (200, 200))

                    x_train.append(roi)
                    y_labels.append(id_)

        print(f"👉 Images used for training: {len(x_train)}")

        if len(x_train) == 0:
            print("❌ No images found in training_data")
            return False

        self.recognizer.train(x_train, np.array(y_labels))
        self.recognizer.save("face_trainer.yml")

        with open("labels.pkl", "wb") as f:
            pickle.dump(self.labels, f)

        print("✅ Training done")
        return True

    def recognize(self):
        # Train automatically if model not exists
        if not os.path.exists("face_trainer.yml"):
            print("⚠️ No model → training first...")
            if not self.train("training_data"):
                return

        self.recognizer.read("face_trainer.yml")

        # Load labels
        if os.path.exists("labels.pkl"):
            with open("labels.pkl", "rb") as f:
                self.labels = pickle.load(f)

        cap = cv2.VideoCapture(0)

        print("🎥 Running... Press Q to exit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            # Show number of people
            cv2.putText(frame, f"People: {len(faces)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            for (x,y,w,h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (200, 200))

                try:
                    id_, conf = self.recognizer.predict(roi)
                    name = self.labels.get(id_, "Unknown")

                    # Optional confidence filter
                    if conf > 80:
                        name = "Unknown"

                except:
                    name = "Unknown"

                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,name,(x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

            cv2.imshow("Face Recognition System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()