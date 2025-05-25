# Computer VIsion
# 🚗 Klasifikasi Warna Plat Nomor Kendaraan dengan Computer Vision

Proyek ini bertujuan untuk mendeteksi dan mengklasifikasikan **warna plat nomor kendaraan** menggunakan teknik **Computer Vision** berbasis Python dan OpenCV. Dataset yang digunakan adalah _Motorcycle License Plate (SKRDR)_ dari Roboflow.

---

## 🔍 Deskripsi

Warna plat nomor di Indonesia memiliki arti penting:
- **Hitam** → Kendaraan pribadi  
- **Kuning** → Kendaraan umum  
- **Merah** → Kendaraan dinas  
- **Putih** → Kendaraan pribadi (baru)  
- **Hijau** → Kendaraan listrik  
- **Biru** → Kendaraan diplomatik

Dengan mengenali warna, sistem ini dapat membantu dalam klasifikasi jenis kendaraan otomatis, seperti pada sistem parkir, tilang elektronik (ETLE), dan analisis lalu lintas.

---

## 🧰 Tools & Teknologi

- Python 3.x
- OpenCV
- NumPy
- Roboflow API
- Matplotlib (opsional)
- Jupyter Notebook / IDE

---

## 📦 Dataset

Dataset digunakan dari Roboflow:

> 🖼 [Motorcycle License Plate (SKRDR) Dataset](https://universe.roboflow.com/zeroexperiments/motorcycle-license-plate-skrdr)

Format: YOLOv8, sudah berisi anotasi bounding box untuk plat nomor.

---

## 🛠️ Langkah-Langkah

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/username/project-plat-warna.git
cd project-plat-warna
pip install -r requirements.txt
