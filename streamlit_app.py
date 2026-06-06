import os
import streamlit as st
import numpy as np
import pandas as pd
import tempfile

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="ConcreteVision AI",
    page_icon="🧠",
    layout="wide"
)

# =====================================
# SESSION STATE INIT
# =====================================
if "history" not in st.session_state:
    st.session_state.history = []

if "images" not in st.session_state:
    st.session_state.images = []

if "model" not in st.session_state:
    st.session_state.model = None

# Tracker untuk cegah double-append history
if "last_predicted_key" not in st.session_state:
    st.session_state.last_predicted_key = None

# Key dinamis untuk reset file_uploader widget
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# =====================================
# SIDEBAR MENU
# =====================================
menu = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Home", "🔍 Predict", "📊 Analytics", "ℹ️ About"]
)

st.sidebar.markdown("---")

# =====================================
# MODEL UPLOAD
# =====================================
uploaded_model = st.sidebar.file_uploader(
    "📁 Upload Model (.h5)",
    type=["h5"]
)

if uploaded_model is not None:
    try:
        # FIX #5 — Hapus tempfile setelah model dimuat
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(uploaded_model.read())
            tmp_path = tmp.name

        st.session_state.model = load_model(tmp_path, compile=False)
        os.unlink(tmp_path)  # Bersihkan file sementara

        st.sidebar.success("Model loaded ✅")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")

model = st.session_state.model

# =====================================
# MODEL STATUS INFO
# =====================================
if model is None:
    st.sidebar.warning("⚠️ Model belum diupload")
else:
    st.sidebar.success("🧠 Model siap digunakan")

# =====================================
# HOME
# =====================================
if menu == "🏠 Home":
    st.title("🧠 ConcreteVision AI")

    st.markdown("---")

    st.markdown("""
    AI untuk klasifikasi retak beton menggunakan CNN.

    ### Fitur:
    - Upload model AI
    - Analisis gambar
    - Dashboard statistik
    - Data tersimpan antar menu
    """)

# =====================================
# PREDICT
# =====================================
if menu == "🔍 Predict":

    st.title("🔍 AI Prediction System")

    uploaded_images = st.file_uploader(
        "Upload gambar beton",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    # Simpan ke session
    if uploaded_images:
        st.session_state.images = uploaded_images

    images = st.session_state.images

    # RESET BUTTON
    colA, colB = st.columns(2)

    with colA:
        reset = st.button("🔄 Reset Gambar")

    if reset:
        st.session_state.images = []
        st.session_state.history = []
        st.session_state.last_predicted_key = None
        st.session_state.uploader_key += 1  # Reset widget file_uploader
        st.rerun()

    if model is not None and len(images) > 0:

        results = []

        st.subheader("📸 Hasil Prediksi")

        col_count = st.slider("Grid Columns", 2, 4, 3)
        cols = st.columns(col_count)

        for i, img_file in enumerate(images):

            image = Image.open(img_file).convert("RGB")

            with st.spinner("🧠 AI processing..."):
                img = image.resize((150, 150))
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array, verbose=0)

                labels = ["Retak", "Tidak_Retak"]

                idx = np.argmax(prediction[0])
                label = labels[idx]
                conf = float(prediction[0][idx]) * 100

            with cols[i % col_count]:

                if label == "Retak":
                    st.error("⚠️ Retak")
                else:
                    st.success("✔ Tidak Retak")

                st.image(image, use_container_width=True)
                st.progress(conf / 100)
                st.write(f"Confidence: {conf:.2f}%")

            results.append({
                "File": img_file.name,
                "Prediksi": label,
                "Confidence (%)": round(conf, 2)
            })

        df = pd.DataFrame(results)

        # FIX #2 — Cegah double-append history setiap kali halaman re-render
        image_names = sorted([f.name for f in images])
        current_key = str(image_names)

        if current_key != st.session_state.last_predicted_key:
            st.session_state.history.append(df)
            st.session_state.last_predicted_key = current_key

        st.markdown("---")

        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False).encode(),
            "hasil_prediksi.csv",
            "text/csv"
        )

    else:
        st.info("Upload model dan gambar terlebih dahulu")

# =====================================
# ANALYTICS
# =====================================
if menu == "📊 Analytics":

    st.title("📊 Dashboard Analisis")

    if len(st.session_state.history) == 0:
        st.warning("Belum ada data prediksi")
    else:

        df = pd.concat(st.session_state.history, ignore_index=True)

        total = len(df)
        retak = len(df[df["Prediksi"] == "Retak"])
        normal = len(df[df["Prediksi"] == "Tidak_Retak"])

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Gambar", total)
        col2.metric("Retak", retak)
        col3.metric("Tidak Retak", normal)

        st.bar_chart(df["Prediksi"].value_counts())

        st.subheader("🔎 Filter Confidence")

        min_conf = st.slider("Minimum Confidence (%)", 0, 100, 50)

        filtered = df[df["Confidence (%)"] >= min_conf]

        st.dataframe(filtered, use_container_width=True)

# =====================================
# ABOUT
# =====================================
if menu == "ℹ️ About":

    st.title("ℹ️ About Project")

    st.markdown("""
    ConcreteVision AI - Sistem deteksi retak beton berbasis CNN

    ✔ Persistent image upload
    ✔ Session-based history (no duplicate entries)
    ✔ Streamlit deployment ready
    ✔ Reset feature added
    """)

    st.success("Ready 🚀")
