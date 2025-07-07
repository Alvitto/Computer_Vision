import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from ultralytics import YOLO

# Load model YOLOv8
model = YOLO("best.pt")  # Pastikan file best.pt berada di folder yang sama

# Fungsi klasifikasi warna berdasarkan HSV
def classify_plate_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_mean, s_mean, v_mean = np.mean(h), np.mean(s), np.mean(v)

    if v_mean > 160 and s_mean < 80:
        return "Putih"
    elif v_mean < 100 and s_mean < 80:
        return "Hitam"
    elif (h_mean < 15 or h_mean > 150) and s_mean > 50:
        return "Merah"
    elif 15 < h_mean < 40 and s_mean > 50:
        return "Kuning"
    else:
        return "Tidak diketahui"


# Title
st.title("Klasifikasi Warna Plat Nomor Kendaraan")

# Upload image
uploaded_file = st.file_uploader("Upload gambar kendaraan", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Buka gambar dan konversi ke OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Convert BGR to RGB untuk ditampilkan di Streamlit
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption='Gambar Asli', use_container_width=True)

    # Deteksi plat nomor menggunakan YOLO
    results = model(image)[0]

    if len(results.boxes) == 0:
        st.warning("⚠️ Tidak ditemukan plat nomor pada gambar.")
    else:
        st.subheader("Hasil Deteksi & Klasifikasi Warna:")
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop plat dari gambar asli
            plate_crop = image[y1:y2, x1:x2]

            # Klasifikasi warna plat
            color_label = classify_plate_color(plate_crop)

            # Tampilkan hasil dengan konversi ke RGB untuk Streamlit
            plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
            st.image(plate_crop_rgb, caption=f'Plat #{i+1} - Warna: {color_label}', width=300)
