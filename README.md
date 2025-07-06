# Computer VIsion Mid-Semester Project
# Vehicle License Plate Color Classification using Computer Vision

---

## Background

In recent years, the implementation of Computer Vision technology has expanded significantly across transportation systems and traffic management. One of its key applications is in the automatic detection and identification of vehicles through camera systems. While most existing systems focus on license plate character recognition (OCR), an important feature is often overlooked: the color of the license plate.

Indonesia, license plate colors:
- **Black** → Private Vehicle  
- **Yellow** → Public/commercial transport 
- **Red** → Government-owned vehicles  
- **Putih** → Private Vehicle (2022 - Now)  
- **Blue** → Electric Vehicle  

---

## Objective

The main objective of this project is to automatically classify the color of vehicle license plates using computer vision techniques.

---

## Project Benefits

Identifying the color of a license plate can serve as a valuable feature in systems such as traffic surveillance, smart parking, automated toll systems, and electronic law enforcement (ETLE).

---

## Tools 

- Roboflow 
- Google Colab
- Platform : Web Base

---

## Dataset

| Description           | Details                                                            |
| --------------------- | ------------------------------------------------------------------ |
| **Dataset Name**      | Motorcycle License Plate (SKRDR)                                   |
| **Source**            | [Roboflow Universe](https://universe.roboflow.com/zeroexperiments/motorcycle-license-plate-skrdr/) |
| **Number of Images**  | 1845+ images                                                       |

---

## Programs 

[🔗 Google Colab](https://colab.research.google.com/drive/1Lqr99H-JAdBJbAuDfzkTR1ms1BIUpqvO?usp=drive_link)

## 1. Needed Dependencies
```python
!pip install ultralytics roboflow opencv-python-headless matplotlib
!pip install ultralytics
pip install streamlit ultralytics opencv-python-headless
```

## 2. Roboflow Dataset
```python
rf = Roboflow(api_key="--")
project = rf.workspace("--").project("--")
dataset = project.version(1).download("yolov8")
from roboflow import Roboflow
```

## 3. Training
```python
from ultralytics import YOLO
model = YOLO("yolov8n.yaml")
model.train(
    data="motorcycle-license-plate-1/data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    name="yolov8n_plate")
```

## 4. Preprocessing
```python
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

model = YOLO("runs/detect/yolov8n_plate2/weights/best.pt")
input_dir = "motorcycle-license-plate-1/valid/images"
output_dir = "cropped_plates"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".jpg"):
        img_path = os.path.join(input_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            continue  # Lewati jika gambar gagal dimuat
        results = model(img)[0]
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            crop_name = f"{file}_plate{i}.jpg"
            cv2.imwrite(os.path.join(output_dir, crop_name), crop)
print("✅ Semua plat nomor berhasil dipotong dan disimpan di folder:", output_dir)
```

## 5. Classification
```python
import numpy as np
import csv
def classify_plate_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_mean, s_mean, v_mean = np.mean(h), np.mean(s), np.mean(v)
    if v_mean > 180 and s_mean < 60:
        return "Putih"
    elif v_mean < 80 and s_mean < 60:
        return "Hitam"
    elif h_mean < 10 or h_mean > 160:
        return "Merah"
    elif 20 < h_mean < 35:
        return "Kuning"
    else:
        return "Tidak diketahui"
output_csv = "plate_colors.csv"
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "color"])
    for fname in os.listdir(output_dir):
        if fname.endswith(".jpg"):
            img = cv2.imread(os.path.join(output_dir, fname))
            if img is not None:
                label = classify_plate_color(img)
                writer.writerow([fname, label])
print("✅ CSV warna plat dibuat:", output_csv)
```
---

## Output
[Streamlit](https://computervision-9tur2wxvbn8anwcioqhfxy.streamlit.app/)

![Image](Plate/p1.png)
![Result](Plate/p2.png)

![Image](Plate/b1.png)
![Result](Plate/b2.png)

---
