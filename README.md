# 👁️ Face Recognition Project

## 📌 Overview
This is a Face Recognition system built using Python and Deep Learning.  
It detects and recognizes faces in real-time using webcam input.

---

## ⚙️ Features
- Real-time face detection
- Face recognition using trained model
- Image dataset training
- Easy to run using Python

---

## 🧠 Tech Stack
- Python
- OpenCV
- TensorFlow / Keras
- NumPy

---

## 📸 Output
![Project Output](output.png)

---

## 📁 How to add your face images
To train the model, add your face images in the following folder structure:

```text
training_data/
  your_name/
    1.jpg
    2.jpg
    3.jpg
```

Each image file should be a clear face photo in `.jpg`, `.jpeg`, or `.png` format.
The folder name becomes the person label used by the recognizer.

Example:
```bash
mkdir -p training_data/ananya
# add your image files here
```

Then run:
```bash
python capture_faces.py
```

This will capture face images and save them under `training_data/<person_name>/` automatically.

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
python main.py
