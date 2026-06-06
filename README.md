# 🧠🏗️ ConcreteVision: Sistem Analisis Retak Beton Berbasis AI

ConcreteVision merupakan aplikasi berbasis Artificial Intelligence (AI) yang digunakan untuk melakukan analisis dan klasifikasi kondisi permukaan beton menggunakan teknologi Convolutional Neural Network (CNN). Sistem ini mampu mengidentifikasi apakah permukaan beton mengalami retak atau tidak berdasarkan citra yang diunggah pengguna.

---

## 👨‍💻 Informasi Pengembang

**Nama** : Muhammad Reval Denta  
**NIM** : 032400048  
**Program Studi** : Elektro Mekanika

---

## 🎯 Tujuan Proyek

Proyek ini dikembangkan untuk membantu proses inspeksi visual permukaan beton secara otomatis menggunakan metode Deep Learning sehingga proses identifikasi retak dapat dilakukan lebih cepat dan efisien.

---

## ✨ Fitur Utama

- Upload model CNN (.h5)
- Upload satu atau banyak gambar beton
- Klasifikasi otomatis Retak / Tidak Retak
- Menampilkan nilai confidence prediksi
- Dashboard analisis hasil
- Statistik jumlah gambar
- Persentase retak dan tidak retak
- Grafik distribusi hasil klasifikasi
- Filter hasil prediksi
- Download hasil analisis ke CSV

---

## 🧠 Teknologi yang Digunakan

- Python
- Streamlit
- TensorFlow 2.18
- NumPy
- Pandas
- Pillow
- H5Py

---

## 🚀 Cara Menjalankan Secara Lokal

Install dependency:

```bash
pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
streamlit run app.py
```

---

## 📂 Struktur Proyek

```text
project/
│
├── app.py
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## 📊 Output Klasifikasi

| Kelas | Deskripsi |
|---------|---------|
| Retak | Permukaan beton terdeteksi memiliki retakan |
| Tidak_Retak | Permukaan beton terdeteksi tidak memiliki retakan |

---

## 📌 Cara Penggunaan

1. Upload file model CNN (.h5)
2. Upload satu atau beberapa gambar beton
3. Tunggu proses prediksi selesai
4. Lihat hasil klasifikasi dan dashboard analisis
5. Download hasil dalam format CSV

---

## 📜 Lisensi

Proyek ini dibuat untuk tujuan pendidikan, penelitian, dan pengembangan sistem inspeksi beton berbasis Artificial Intelligence.
