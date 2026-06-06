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
# SESSION STATE (history)
# =====================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================
# THEME CONTROL (FIXED)
# =====================================
theme = st.sidebar.selectbox("🎨 Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
    <style>

    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    h1, h2, h3, h4, p, span, label {
        color: #ffffff !important;
    }

    div[data-testid="stMetric"] {
        background-color: #1f2937;
        padding: 10px;
        border-radius: 10px;
    }

    .stDataFrame {
        background-color: #111827;
    }

    button {
        border-radius: 8px;
    }

    * {
        transition: all 0.2s ease-in-out;
    }

    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
    .stApp {
        background-color: white;
        color: black;
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
        st.sidebar.success("Model loaded ✅")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# =====================================
# HOME PAGE
# =====================================
if menu == "🏠 Home":

    st.title("🧠 ConcreteVision AI")
    st.markdown("""
    Sistem AI untuk klasifikasi retak beton menggunakan CNN.

    ### Fitur:
    - Upload model AI
    - Prediksi banyak gambar
    - Dashboard analisis
    - Dark / Light mode
    """)

    st.success("Masuk ke Predict untuk mulai analisis")

# =====================================
# PREDICT PAGE
# =====================================
if menu == "🔍 Predict":

    st.title("🔍 AI Prediction System")

    uploaded_images = st.file_uploader(
        "Upload gambar beton",
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

        st.subheader("Filter Confidence")

        min_conf = st.slider("Minimal Confidence (%)", 0, 100, 50)

        filtered = df[df["Confidence"] >= min_conf]

        st.dataframe(filtered, use_container_width=True)

# =====================================
# ABOUT PAGE
# =====================================
if menu == "ℹ️ About":

    st.title("ℹ️ About Project")

    st.markdown("""
    ConcreteVision AI adalah sistem AI untuk mendeteksi retakan beton.

    Teknologi:
    - TensorFlow
    - CNN
    - Streamlit
    """)

    st.success("Siap deploy ke Streamlit Cloud 🚀")
