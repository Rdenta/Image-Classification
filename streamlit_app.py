import streamlit as st
import numpy as np
import pandas as pd
import tempfile

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(
    page_title="Deteksi Retak Beton",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Deteksi Retak Beton Menggunakan CNN")

st.markdown("""
### Cara Penggunaan
1. Upload model `.h5`
2. Upload satu atau banyak gambar beton
3. Sistem akan melakukan klasifikasi otomatis
""")

# ==========================
# Upload Model
# ==========================

uploaded_model = st.file_uploader(
    "Upload Model CNN (.h5)",
    type=["h5"]
)

model = None

if uploaded_model is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(uploaded_model.read())
            model = load_model(tmp.name, compile=False)

        st.success("✅ Model berhasil dimuat")

    except Exception as e:
        st.error(f"Gagal memuat model: {e}")

# ==========================
# Upload Banyak Gambar
# ==========================

uploaded_images = st.file_uploader(
    "Upload Gambar Beton",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ==========================
# Prediksi
# ==========================

if model is not None and uploaded_images:

    hasil_prediksi = []

    st.header("Hasil Prediksi")

    for uploaded_image in uploaded_images:

        image = Image.open(uploaded_image).convert("RGB")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(
                image,
                caption=uploaded_image.name,
                use_container_width=True
            )

        img = image.resize((150, 150))

        img_array = img_to_array(img)

        # Jika saat training menggunakan normalisasi,
        # hapus komentar pada baris berikut:
        # img_array = img_array / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)

        class_names = ["Retak", "Tidak_Retak"]

        predicted_index = np.argmax(prediction[0])
        predicted_class = class_names[predicted_index]

        confidence = float(prediction[0][predicted_index]) * 100

        with col2:
            st.subheader(uploaded_image.name)
            st.success(f"Hasil: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%")

        hasil_prediksi.append({
            "Nama File": uploaded_image.name,
            "Prediksi": predicted_class,
            "Confidence (%)": round(confidence, 2)
        })

        st.divider()

    # ==========================
    # Tabel Hasil
    # ==========================

    st.header("Ringkasan Hasil")

    df = pd.DataFrame(hasil_prediksi)

    st.dataframe(
        df,
        use_container_width=True
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Hasil CSV",
        data=csv,
        file_name="hasil_prediksi_beton.csv",
        mime="text/csv"
    )
