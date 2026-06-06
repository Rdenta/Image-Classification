import streamlit as st
import numpy as np
import pandas as pd
import tempfile

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# =====================================
# CONFIG
# =====================================

st.set_page_config(
    page_title="ConcreteVision AI",
    page_icon="🧠",
    layout="wide"
)

# =====================================
# SIDEBAR NAVIGATION
# =====================================

st.sidebar.title("🧠 ConcreteVision")
menu = st.sidebar.radio(
    "Navigasi",
    ["🏠 Home", "🔍 Predict", "📊 Analytics", "ℹ️ About"]
)

st.sidebar.markdown("---")

# =====================================
# HOME
# =====================================

if menu == "🏠 Home":
    st.title("🏗️ ConcreteVision AI")
    st.markdown("""
    Sistem analisis retak beton berbasis Artificial Intelligence (CNN)

    ### ✨ Fitur:
    - Klasifikasi Retak / Tidak Retak
    - Upload model AI (.h5)
    - Analisis gambar banyak sekaligus
    - Dashboard statistik
    - UI modern & interaktif
    """)

    st.success("Silakan masuk ke menu Predict untuk mulai analisis")

# =====================================
# LOAD MODEL (GLOBAL)
# =====================================

uploaded_model = st.sidebar.file_uploader(
    "📁 Upload Model (.h5)",
    type=["h5"]
)

model = None

if uploaded_model is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(uploaded_model.read())
            model = load_model(tmp.name, compile=False)

        st.sidebar.success("Model loaded")

    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# =====================================
# PREDICT PAGE
# =====================================

if menu == "🔍 Predict":

    st.title("🔍 AI Prediction")

    uploaded_images = st.file_uploader(
        "Upload gambar beton",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if model is not None and uploaded_images:

        hasil = []

        st.markdown("## 📸 Hasil Analisis")

        col_count = st.slider("Jumlah kolom grid", 2, 4, 3)

        cols = st.columns(col_count)

        for i, img_file in enumerate(uploaded_images):

            image = Image.open(img_file).convert("RGB")

            with st.spinner("Menganalisis gambar..."):
                img = image.resize((150, 150))
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array, verbose=0)

                class_names = ["Retak", "Tidak_Retak"]

                idx = np.argmax(prediction[0])
                label = class_names[idx]
                conf = float(prediction[0][idx]) * 100

            # GRID DISPLAY
            with cols[i % col_count]:

                if label == "Retak":
                    st.error("⚠️ Retak")
                else:
                    st.success("✔ Tidak Retak")

                st.image(image, use_container_width=True)
                st.write(f"Confidence: {conf:.2f}%")
                st.progress(conf / 100)

            hasil.append({
                "File": img_file.name,
                "Prediksi": label,
                "Confidence": conf
            })

        df = pd.DataFrame(hasil)

        st.markdown("---")

        st.subheader("📥 Download Hasil")
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode(),
            "hasil_prediksi.csv",
            "text/csv"
        )

# =====================================
# ANALYTICS
# =====================================

if menu == "📊 Analytics":

    st.title("📊 Dashboard Analisis")

    if "df" not in locals():
        st.info("Lakukan prediksi terlebih dahulu")
    else:

        total = len(df)
        retak = len(df[df["Prediksi"] == "Retak"])
        normal = len(df[df["Prediksi"] == "Tidak_Retak"])

        col1, col2, col3 = st.columns(3)

        col1.metric("Total", total)
        col2.metric("Retak", retak)
        col3.metric("Tidak Retak", normal)

        st.bar_chart(df["Prediksi"].value_counts())

        st.subheader("Filter Confidence")

        min_conf = st.slider("Minimal Confidence (%)", 0, 100, 50)

        filtered = df[df["Confidence"] >= min_conf]

        st.dataframe(filtered, use_container_width=True)

# =====================================
# ABOUT
# =====================================

if menu == "ℹ️ About":
    st.title("ℹ️ About Project")

    st.markdown("""
    **ConcreteVision AI**

    Sistem ini dibuat untuk membantu inspeksi visual beton menggunakan AI.

    ### Teknologi:
    - TensorFlow CNN
    - Streamlit
    - Python

    ### Output:
    - Retak
    - Tidak Retak
    """)

    st.info("Project ini dibuat untuk keperluan akademik dan pengembangan AI")
