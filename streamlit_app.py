import streamlit as st
import numpy as np
import os
import pandas as pd
from PIL import Image

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Image Classification Beton",
    page_icon="🏗️",
    layout="centered"
)

st.title("🏗️ Klasifikasi Retakan Beton")
st.markdown("""
**Nama :** Muhammad Reval Denta  
**NIM   :** 032400048  
**Prodi :** Elektro Mekanika  

Aplikasi ini menggunakan model CNN untuk mendeteksi apakah sebuah gambar beton **Retak** atau **Tidak Retak**.
""")
st.divider()

# =====================================================
# KONFIGURASI MODEL
# =====================================================
MODEL_PATH  = "model_crack_beton.h5"
IMG_HEIGHT  = 150
IMG_WIDTH   = 150
CLASS_NAMES = ["Retak", "Tidak_Retak"]

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.header("ℹ️ Informasi Model")
    st.markdown(f"""
    - **Arsitektur:** CNN (Sequential)
    - **Input:** {IMG_HEIGHT}×{IMG_WIDTH} px
    - **Kelas:** {", ".join(CLASS_NAMES)}
    - **Optimizer:** Adam
    - **Epochs:** 10
    """)
    st.divider()
    st.subheader("📊 Performa Training")
    st.markdown("""
    | Metrik | Nilai |
    |---|---|
    | Akurasi Train | ~99.00% |
    | Akurasi Val   | ~98.75% |
    | Loss Train    | ~0.034  |
    | Loss Val      | ~0.078  |
    """)
    st.divider()
    st.caption("Letakkan `model_crack_beton.h5` di folder yang sama dengan `streamlit_app.py`.")

# =====================================================
# LOAD MODEL (lazy, hanya load sekali)
# =====================================================
@st.cache_resource
def load_model_cached(path):
    # Gunakan tensorflow.keras agar kompatibel di semua environment
    try:
        from tensorflow import keras
        model = keras.models.load_model(path, compile=False)
    except Exception:
        import tensorflow as tf
        model = tf.keras.models.load_model(path, compile=False)
    return model

# =====================================================
# UPLOAD MODEL jika belum ada
# =====================================================
if not os.path.exists(MODEL_PATH):
    st.warning("⚠️ File model `model_crack_beton.h5` tidak ditemukan.")
    uploaded_model = st.file_uploader("Upload file model (.h5)", type=["h5"])
    if uploaded_model is not None:
        with open(MODEL_PATH, "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.success("✅ Model berhasil diupload! Refresh otomatis...")
        st.rerun()
    st.stop()

try:
    model = load_model_cached(MODEL_PATH)
    st.success("✅ Model berhasil dimuat!")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

st.divider()

# =====================================================
# FUNGSI PREDIKSI
# =====================================================
def predict_image(img_pil):
    img_resized = img_pil.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array   = np.array(img_resized, dtype=np.float32)
    img_array   = np.expand_dims(img_array, axis=0)   # shape: [1, 150, 150, 3]
    probs       = model.predict(img_array, verbose=0)[0]
    idx         = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]) * 100, probs

# =====================================================
# UPLOAD GAMBAR
# =====================================================
st.subheader("📷 Upload Gambar Beton")
uploaded_files = st.file_uploader(
    "Pilih satu atau beberapa gambar beton (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.subheader("🔍 Hasil Prediksi")
    results = []

    cols_per_row = 2
    for i in range(0, len(uploaded_files), cols_per_row):
        row_files = uploaded_files[i:i+cols_per_row]
        cols = st.columns(len(row_files))
        for col, f in zip(cols, row_files):
            with col:
                img_pil = Image.open(f).convert("RGB")
                st.image(img_pil, caption=f.name, use_container_width=True)

                label, conf, probs = predict_image(img_pil)

                if label == "Retak":
                    st.error(f"**{label}** 🔴")
                else:
                    st.success(f"**{label}** 🟢")

                st.metric("Confidence", f"{conf:.2f}%")

                with st.expander("Detail probabilitas"):
                    for j, cls in enumerate(CLASS_NAMES):
                        p = float(probs[j]) * 100
                        st.write(f"{cls}: **{p:.2f}%**")
                        st.progress(p / 100)

                results.append({
                    "Nama File"      : f.name,
                    "Prediksi"       : label,
                    "Confidence (%)" : f"{conf:.2f}"
                })

    # Ringkasan tabel
    st.divider()
    st.subheader("📋 Ringkasan Hasil")
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    total = len(df)
    retak = (df["Prediksi"] == "Retak").sum()
    tidak = total - retak
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Gambar", total)
    c2.metric("Retak 🔴", retak)
    c3.metric("Tidak Retak 🟢", tidak)

else:
    st.info("Silakan upload gambar beton untuk memulai prediksi.")

# =====================================================
# PEMBAHASAN
# =====================================================
st.divider()
with st.expander("📈 Pembahasan Hasil Training"):
    st.markdown("""
    | Metrik | Awal | Akhir |
    |---|---|---|
    | Training Accuracy   | ~71.63% | ~99.00% |
    | Validation Accuracy | ~92.25% | ~98.75% |
    | Training Loss       | 0.546   | 0.034   |
    | Validation Loss     | 0.201   | 0.078   |

    Model berhasil mempelajari pola gambar beton dengan sangat baik tanpa tanda-tanda *overfitting* yang signifikan.
    """)

with st.expander("📝 Kesimpulan Akhir"):
    st.markdown("""
    1. **Akurasi Tinggi:** Model mencapai akurasi mendekati 99% dalam 10 epoch.
    2. **Confidence Tinggi:** Prediksi pada gambar baru mencapai >90%, bahkan banyak mendekati 99%.
    3. Model ini cocok untuk mendeteksi retakan beton secara otomatis.
    """)
