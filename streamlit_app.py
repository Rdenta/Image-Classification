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
# DARK MODE TOGGLE (simple UI control)
# =====================================
theme = st.sidebar.selectbox("🎨 Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        .stApp {
            background-color: #0e1117;
        }
    </style>
    """, unsafe_allow_html=True)

# =====================================
# NAVIGATION
# =====================================
menu = st.sidebar.radio(
    "📌 Menu",
    ["🏠 Home", "🔍 Predict", "📊 Analytics", "ℹ️ About"]
)

st.sidebar.markdown("---")

# =====================================
# LOAD MODEL
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
        st.sidebar.success("Model Loaded ✅")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# =====================================
# SESSION STORAGE (history)
# =====================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================
# HOME PAGE
# =====================================
if menu == "🏠 Home":
    st.title("🧠 ConcreteVision AI")
    st.write("AI untuk klasifikasi retak beton berbasis CNN")

    st.info("Upload model dan gambar di menu Predict untuk mulai analisis")

# =====================================
# PREDICT PAGE
# =====================================
if menu == "🔍 Predict":

    st.title("🔍 AI Prediction System")

    uploaded_images = st.file_uploader(
        "Upload Gambar Beton",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if model is not None and uploaded_images:

        results = []

        col_count = st.slider("Grid Column", 2, 4, 3)
        cols = st.columns(col_count)

        st.subheader("📸 Hasil Analisis")

        for i, img_file in enumerate(uploaded_images):

            image = Image.open(img_file).convert("RGB")

            with st.spinner("🧠 AI sedang menganalisis..."):
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
                "Confidence": conf
            })

        df = pd.DataFrame(results)

        # save session history
        st.session_state.history.append(df)

        st.markdown("---")

        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False).encode(),
            "hasil_prediksi.csv",
            "text/csv"
        )

# =====================================
# ANALYTICS PAGE
# =====================================
if menu == "📊 Analytics":

    st.title("📊 Dashboard Analisis")

    if len(st.session_state.history) == 0:
        st.warning("Belum ada data prediksi")
    else:

        df = pd.concat(st.session_state.history)

        total = len(df)
        retak = len(df[df["Prediksi"] == "Retak"])
        normal = len(df[df["Prediksi"] == "Tidak_Retak"])

        col1, col2, col3 = st.columns(3)

        col1.metric("Total", total)
        col2.metric("Retak", retak)
        col3.metric("Tidak Retak", normal)

        st.bar_chart(df["Prediksi"].value_counts())

        st.subheader("🔎 Filter Confidence")

        min_conf = st.slider("Minimal Confidence (%)", 0, 100, 50)

        filtered = df[df["Confidence"] >= min_conf]

        st.dataframe(filtered, use_container_width=True)

# =====================================
# ABOUT PAGE
# =====================================
if menu == "ℹ️ About":

    st.title("ℹ️ About Project")

    st.markdown("""
    ### ConcreteVision AI

    Sistem AI untuk mendeteksi retakan pada permukaan beton.

    ### Features:
    - CNN Classification
    - Multi-image prediction
    - Dashboard analytics
    - Dark mode UI
    - Session history

    ### Tech Stack:
    Python | TensorFlow | Streamlit
    """)

    st.success("Project siap untuk deployment Streamlit Cloud 🚀")
