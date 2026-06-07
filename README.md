# 🔬 Sistem Analisis Retak Beton Berbasis AI

Aplikasi web berbasis **Convolutional Neural Network (CNN)** untuk mendeteksi dan mengklasifikasikan keretakan pada permukaan beton secara otomatis melalui analisis citra digital. Dibangun menggunakan **Streamlit** dan **TensorFlow**, serta dapat di-deploy langsung di **Streamlit Cloud**.

---

## 🚀 Demo Aplikasi

> Setelah deploy ke Streamlit Cloud, tambahkan link aplikasi Anda di sini.

🔗 **Link Aplikasi:** `https://share.streamlit.io/username/nama-repo/main/streamlit_app.py`

---

## ✨ Fitur Utama

- 🧠 **Upload model CNN (.h5)** langsung dari browser tanpa konfigurasi tambahan
- 🖼️ **Analisis multi-gambar** beton sekaligus dalam satu sesi
- ⚙️ **Threshold confidence** yang bisa diatur (30–90%) sesuai kebutuhan analisis
- 🔄 **Auto-detect urutan label** (sigmoid & softmax) secara otomatis
- 🔁 **Koreksi manual label** jika urutan prediksi terbalik
- 🏗️ **Info arsitektur model** — jumlah layer, parameter, input/output shape
- 📊 **Dashboard analitik interaktif** dengan pie chart, bar chart, dan histogram confidence (Plotly)
- 🔎 **Filter data** berdasarkan confidence dan kelas prediksi
- 📄 **Export laporan PDF otomatis** lengkap dengan grafik visualisasi
- 📥 **Export CSV** hasil prediksi seluruh sesi
- 💾 **Data tersimpan antar menu** dalam satu sesi

---

## 🗂️ Struktur Repository

```
├── streamlit_app.py       # File utama aplikasi Streamlit
├── requirements.txt       # Daftar dependencies Python
└── README.md              # Dokumentasi project
```

---

## 🛠️ Teknologi yang Digunakan

| Library | Versi | Kegunaan |
|---|---|---|
| Streamlit | Latest | Framework UI web |
| TensorFlow (CPU) | 2.18.0 | Load & inferensi model CNN |
| Plotly | Latest | Grafik interaktif |
| Matplotlib | Latest | Grafik untuk laporan PDF |
| ReportLab | Latest | Generate laporan PDF |
| Pandas | Latest | Pengolahan data tabular |
| NumPy | Latest | Operasi array & numerik |
| Pillow | Latest | Preprocessing gambar |

---

## 💻 Menjalankan Secara Lokal

### 1. Clone Repository

```bash
git clone https://github.com/username/nama-repo.git
cd nama-repo
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi

```bash
streamlit run streamlit_app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`

---

## ☁️ Deploy ke Streamlit Cloud

1. Push repository ini ke **GitHub**
2. Buka [share.streamlit.io](https://share.streamlit.io) dan login dengan akun GitHub
3. Klik **New app** → pilih repository ini
4. Set **Main file path** ke `streamlit_app.py`
5. Klik **Deploy** dan tunggu proses selesai

> ⚠️ **Catatan:** Proses deploy pertama membutuhkan waktu beberapa menit karena TensorFlow berukuran besar (~500 MB). Setelah itu aplikasi dapat diakses kapan saja.

---

## 📖 Cara Penggunaan

### 1. Upload Model
- Buka sidebar kiri
- Upload file model CNN dalam format `.h5`
- Informasi arsitektur model akan otomatis tampil di sidebar

### 2. Prediksi Gambar
- Pilih menu **🔍 Predict**
- Upload satu atau lebih gambar beton (JPG/PNG)
- Atur **threshold confidence** sesuai kebutuhan (default: 50%)
- Hasil prediksi tampil dalam grid lengkap dengan confidence score

### 3. Analisis Hasil
- Pilih menu **📊 Analytics**
- Lihat distribusi hasil prediksi dalam bentuk grafik interaktif
- Filter data berdasarkan confidence minimum dan kelas prediksi

### 4. Export Laporan
- Di menu **Predict**, klik **Download PDF** untuk laporan lengkap dengan grafik
- Klik **Download CSV** untuk data mentah hasil prediksi

---

## 📋 Catatan Teknis

- Gambar akan di-resize otomatis ke **150×150 px** sesuai input model
- Mendukung model dengan output **sigmoid** (binary) maupun **softmax** (multi-class)
- Jika prediksi terbalik, aktifkan toggle **Balik urutan label** di sidebar

---

## 👤 Identitas Mahasiswa

| Keterangan | Detail |
|---|---|
| **Nama** | Muhammad Reval Denta |
| **NIM** | 032400048 |
| **Program Studi** | Elektro Mekanika |

---

## 📄 Lisensi

Project ini dibuat untuk keperluan Tugas Akhir. Silakan gunakan dan modifikasi sesuai kebutuhan dengan mencantumkan sumber.
