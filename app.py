import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from ultralytics import YOLO

# Load model YOLOv8
model = YOLO("best.pt")

# Fungsi klasifikasi warna berdasarkan HSV
def classify_plate_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_mean, s_mean, v_mean = np.mean(h), np.mean(s), np.mean(v)

    if v_mean > 170 and s_mean < 50:
        return "Putih"
    elif v_mean < 90 and s_mean < 50:
        return "Hitam"
    elif h_mean < 10 or h_mean > 160:
        return "Merah"
    elif 20 < h_mean < 35:
        return "Kuning"
    else:
        return "Abu / Tidak diketahui"


# Title
st.title("Klasifikasi Warna Plat Nomor Kendaraan")

# Upload image
uploaded_file = st.file_uploader("Upload gambar kendaraan", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Buka gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption='Gambar Asli', use_column_width=True)

    # Deteksi plat nomor
    results = model(image)[0]

    if len(results.boxes) == 0:
        st.warning("⚠️ Tidak ditemukan plat nomor pada gambar.")
    else:
        st.subheader("Hasil Deteksi & Klasifikasi Warna:")
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = image[y1:y2, x1:x2]

            color_label = classify_plate_color(plate_crop)

            st.image(plate_crop, caption=f'Plat #{i+1} - Warna: {color_label}', width=300)
