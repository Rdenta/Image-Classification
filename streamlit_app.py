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
# SESSION STATE
# =====================================
if "history" not in st.session_state:
    st.session_state.history = []

if "images" not in st.session_state:
    st.session_state.images = []

if "model" not in st.session_state:
    st.session_state.model = None

# =====================================
# SIDEBAR
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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(uploaded_model.read())
        st.session_state.model = load_model(tmp.name, compile=False)
    st.sidebar.success("Model loaded ✅")

model = st.session_state.model

# =====================================
# 🔥 LABEL MAPPING (FIX UTAMA)
# =====================================
st.sidebar.subheader("⚙️ Label Mapping (WAJIB CEK)")

class0_label = st.sidebar.text_input("Label untuk Class 0", "Tidak_Retak")
class1_label = st.sidebar.text_input("Label untuk Class 1", "Retak")

labels = [class0_label, class1_label]

# =====================================
# HOME
# =====================================
if menu == "🏠 Home":
    st.title("🧠 ConcreteVision AI")
    st.markdown("---")
    st.write("AI deteksi retak beton (CNN)")

# =====================================
# PREDICT
# =====================================
if menu == "🔍 Predict":

    st.title("🔍 AI Prediction System")

    uploaded_images = st.file_uploader(
        "Upload gambar beton",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_images:
        st.session_state.images = uploaded_images

    images = st.session_state.images

    if st.button("🔄 Reset"):
        st.session_state.images = []
        st.rerun()

    if model is not None and len(images) > 0:

        results = []

        col_count = st.slider("Grid Columns", 2, 4, 3)
        cols = st.columns(col_count)

        for i, img_file in enumerate(images):

            image = Image.open(img_file).convert("RGB")

            img = image.resize((150, 150))
            img_array = img_to_array(img)

            # ❗ IMPORTANT: TIDAK NORMALISASI (karena model kamu sudah punya Rescaling)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)

            # =====================================
            # DEBUG RAW OUTPUT (WAJIB LIHAT)
            # =====================================
            st.write(f"RAW {img_file.name}:", prediction[0])

            idx = np.argmax(prediction[0])
            conf = float(prediction[0][idx]) * 100
            label = labels[idx]

            with cols[i % col_count]:

                if idx == 0:
                    st.info(f"Class 0 → {class0_label}")
                else:
                    st.info(f"Class 1 → {class1_label}")

                st.image(image, use_container_width=True)

                st.progress(conf / 100)
                st.write(f"👉 Prediksi: {label}")
                st.write(f"Confidence: {conf:.2f}%")

            results.append({
                "File": img_file.name,
                "Class Index": int(idx),
                "Prediksi": label,
                "Confidence (%)": round(conf, 2)
            })

        df = pd.DataFrame(results)
        st.session_state.history.append(df)

        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False).encode(),
            "hasil.csv",
            "text/csv"
        )

    else:
        st.info("Upload model + gambar dulu")

# =====================================
# ANALYTICS
# =====================================
if menu == "📊 Analytics":

    st.title("📊 Analytics")

    if len(st.session_state.history) == 0:
        st.warning("Belum ada data")
    else:

        df = pd.concat(st.session_state.history)

        st.metric("Total", len(df))
        st.bar_chart(df["Prediksi"].value_counts())

        st.dataframe(df, use_container_width=True)

# =====================================
# ABOUT
# =====================================
if menu == "ℹ️ About":

    st.title("About")
    st.write("Concrete crack detection CNN app")
