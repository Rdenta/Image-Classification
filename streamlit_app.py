import streamlit as st
import numpy as np
import os
import pandas as pd
from PIL import Image

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

MODEL_PATH  = "model_crack_beton.h5"
IMG_HEIGHT  = 150
IMG_WIDTH   = 150
CLASS_NAMES = ["Retak", "Tidak_Retak"]

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

# Layer augmentasi yang punya keyword tidak kompatibel
AUGMENTATION_LAYERS = {'RandomFlip', 'RandomRotation', 'RandomZoom', 'RandomCrop',
                       'RandomContrast', 'RandomBrightness', 'RandomTranslation'}

def fix_config(obj, parent_class=None):
    if isinstance(obj, dict):
        # Ganti DTypePolicy dengan string float32
        if obj.get('class_name') == 'DTypePolicy':
            return 'float32'

        current_class = obj.get('class_name', parent_class)
        cleaned = {}

        for k, v in obj.items():
            # Selalu hapus key ini
            if k in ('optional', 'registered_name', 'module'):
                continue

            # batch_shape -> input_shape
            if k == 'batch_shape' and isinstance(v, list) and len(v) > 1:
                cleaned['input_shape'] = v[1:]
                continue

            # Hapus data_format dari semua layer augmentasi
            if k == 'data_format' and current_class in AUGMENTATION_LAYERS:
                continue

            cleaned[k] = fix_config(v, current_class)

        return cleaned

    elif isinstance(obj, list):
        return [fix_config(i, parent_class) for i in obj]

    return obj

@st.cache_resource
def load_model_cached(path):
    import tensorflow as tf
    import json

    with open(path, 'rb') as f:
        content = f.read()

    start = content.find(b'{"class_name": "Sequential"')
    if start == -1:
        start = content.find(b'{"class_name":"Sequential"')

    chunk = content[start:start+100000].decode('utf-8', errors='ignore')
    depth, end = 0, 0
    for i, c in enumerate(chunk):
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    config = json.loads(chunk[:end])
    config = fix_config(config)

    model = tf.keras.models.model_from_json(json.dumps(config))
    model.load_weights(path)
    return model

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

def predict_image(img_pil):
    img_resized = img_pil.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array   = np.array(img_resized, dtype=np.float32)
    img_array   = np.expand_dims(img_array, axis=0)
    probs       = model.predict(img_array, verbose=0)[0]
    idx         = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]) * 100, probs

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
