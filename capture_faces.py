import cv2
import os

def capture_faces(person_name):
    # Create folder for person
    path = f"training_data/{person_name}"
    os.makedirs(path, exist_ok=True)

    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    count = 0

    print("📸 Press SPACE to capture face")
    print("❌ Press ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw rectangle
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        cv2.imshow("Auto Face Capture", frame)

        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break

        elif key == 32:  # SPACE
            if len(faces) == 0:
                print("⚠️ No face detected!")
                continue

            # Take first detected face
            (x, y, w, h) = faces[0]
            face_img = frame[y:y+h, x:x+w]

            face_img = cv2.resize(face_img, (200, 200))

            filename = f"{path}/{count}.jpg"
            cv2.imwrite(filename, face_img)

            print(f"✅ Saved: {filename}")
            count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    name = input("Enter person name: ")
    capture_faces(name)