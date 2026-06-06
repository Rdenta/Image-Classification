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
    page_title="ConcreteVision AI Dashboard",
    page_icon="🧠",
    layout="wide"
)

# =====================================
# SESSION STATE INIT
# =====================================
if "images" not in st.session_state:
    st.session_state.images = []

if "history" not in st.session_state:
    st.session_state.history = []

if "model" not in st.session_state:
    st.session_state.model = None

# =====================================
# SIDEBAR CONTROL PANEL
# =====================================
st.sidebar.title("🧠 Control Panel")

menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔍 Predict", "📊 Dashboard"]
)

uploaded_model = st.sidebar.file_uploader(
    "Upload Model (.h5)",
    type=["h5"]
)

if uploaded_model:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(uploaded_model.read())
            st.session_state.model = load_model(tmp.name, compile=False)
        st.sidebar.success("Model Loaded")
    except Exception as e:
        st.sidebar.error(e)

model = st.session_state.model

# =====================================
# STEP FLOW INDICATOR
# =====================================
def step_flow(active):
    cols = st.columns(3)

    steps = ["📤 Upload", "⚙️ Process", "📊 Result"]

    for i, step in enumerate(steps):
        if i == active:
            cols[i].success(step)
        else:
            cols[i].info(step)

# =====================================
# HOME
# =====================================
if menu == "🏠 Home":

    st.title("🧠 ConcreteVision AI Dashboard")

    st.markdown("AI untuk deteksi retak beton berbasis CNN")

    st.markdown("---")

    step_flow(0)

    st.info("Masuk ke Predict untuk mulai analisis")

# =====================================
# PREDICT
# =====================================
if menu == "🔍 Predict":

    st.title("🔍 AI Prediction Dashboard")

    step_flow(0)

    uploaded_images = st.file_uploader(
        "Upload Gambar Beton",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_images:
        st.session_state.images = uploaded_images

    images = st.session_state.images

    if model and len(images) > 0:

        step_flow(1)

        results = []

        col_count = st.slider("Grid Layout", 2, 4, 3)
        cols = st.columns(col_count)

        st.subheader("📸 Prediction Results")

        for i, img_file in enumerate(images):

            image = Image.open(img_file).convert("RGB")

            with st.spinner("AI analyzing..."):
                img = image.resize((150, 150))
                arr = img_to_array(img)
                arr = np.expand_dims(arr, axis=0)

                pred = model.predict(arr, verbose=0)

                labels = ["Retak", "Tidak_Retak"]

                idx = np.argmax(pred[0])
                label = labels[idx]
                conf = float(pred[0][idx]) * 100

            with cols[i % col_count]:

                if label == "Retak":
                    st.error("⚠️ RETAK")
                else:
                    st.success("✔ AMAN")

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

        step_flow(2)

        st.markdown("---")

        st.download_button(
            "📥 Download Report CSV",
            df.to_csv(index=False).encode(),
            "report.csv",
            "text/csv"
        )

# =====================================
# DASHBOARD ANALYTICS
# =====================================
if menu == "📊 Dashboard":

    st.title("📊 Analytics Dashboard")

    if len(st.session_state.history) == 0:
        st.warning("Belum ada data")
    else:

        df = pd.concat(st.session_state.history)

        total = len(df)
        retak = len(df[df["Prediksi"] == "Retak"])
        aman = len(df[df["Prediksi"] == "Tidak_Retak"])

        # ==============================
        # CARD UI METRICS
        # ==============================
        c1, c2, c3 = st.columns(3)

        c1.metric("Total Images", total)
        c2.metric("Retak", retak)
        c3.metric("Tidak Retak", aman)

        st.markdown("---")

        # ==============================
        # CHART
        # ==============================
        st.subheader("📊 Distribution")
        st.bar_chart(df["Prediksi"].value_counts())

        # ==============================
        # FILTER
        # ==============================
        st.subheader("🔎 Filter Data")

        min_conf = st.slider("Minimum Confidence (%)", 0, 100, 50)

        filtered = df[df["Confidence"] >= min_conf]

        st.dataframe(filtered, use_container_width=True)
