import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import io

from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# =====================================
# CONFIG
# =====================================
st.set_page_config(
    page_title="ConcreteVision AI Pro",
    page_icon="🧠",
    layout="wide"
)

# =====================================
# SESSION STATE
# =====================================
if "images" not in st.session_state:
    st.session_state.images = []

if "history" not in st.session_state:
    st.session_state.history = []

if "model" not in st.session_state:
    st.session_state.model = None

# =====================================
# LOAD MODEL
# =====================================
st.sidebar.title("🧠 AI Control Panel")

menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔍 Predict", "📊 Dashboard"]
)

uploaded_model = st.sidebar.file_uploader("Upload Model (.h5)", type=["h5"])

if uploaded_model:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(uploaded_model.read())
        st.session_state.model = load_model(tmp.name, compile=False)
    st.sidebar.success("Model loaded")

model = st.session_state.model

# =====================================
# GRAD-CAM FUNCTION
# =====================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = tf.reduce_max(predictions)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# =====================================
# OVERLAY HEATMAP
# =====================================
def overlay_heatmap(img, heatmap):

    img = np.array(img)

    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize((img.shape[1], img.shape[0]))

    heatmap = np.array(heatmap)

    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close()

    return buf

# =====================================
# PREDICT PAGE
# =====================================
if menu == "🔍 Predict":

    st.title("🧠 AI Prediction + Explainability")

    uploaded_images = st.file_uploader(
        "Upload Gambar Beton",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_images:
        st.session_state.images = uploaded_images

    images = st.session_state.images

    if model and len(images) > 0:

        results = []

        st.subheader("📸 Hasil + Grad-CAM")

        col1, col2 = st.columns(2)

        for img_file in images:

            image = Image.open(img_file).convert("RGB")

            img = image.resize((150, 150))
            arr = img_to_array(img)
            arr = np.expand_dims(arr, axis=0)

            pred = model.predict(arr, verbose=0)

            labels = ["Retak", "Tidak_Retak"]

            idx = np.argmax(pred[0])
            label = labels[idx]
            conf = float(pred[0][idx]) * 100

            # ==========================
            # GRAD-CAM (ambil layer terakhir CNN)
            # ==========================
            try:
                last_conv = [l.name for l in model.layers if "conv" in l.name][-1]
                heatmap = make_gradcam_heatmap(arr, model, last_conv)
                cam_img = overlay_heatmap(image, heatmap)
            except:
                cam_img = None

            with col1:
                st.image(image, caption=f"Original - {label} ({conf:.2f}%)")

            with col2:
                if cam_img:
                    st.image(cam_img, caption="Grad-CAM Heatmap")

            results.append({
                "File": img_file.name,
                "Prediksi": label,
                "Confidence": conf
            })

        df = pd.DataFrame(results)
        st.session_state.history.append(df)

        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False).encode(),
            "result.csv",
            "text/csv"
        )

# =====================================
# DASHBOARD
# =====================================
if menu == "📊 Dashboard":

    st.title("📊 AI Analytics Dashboard")

    if len(st.session_state.history) == 0:
        st.warning("Belum ada data")
    else:

        df = pd.concat(st.session_state.history)

        st.subheader("Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric("Total", len(df))
        c2.metric("Retak", len(df[df["Prediksi"]=="Retak"]))
        c3.metric("Aman", len(df[df["Prediksi"]=="Tidak_Retak"]))

        st.bar_chart(df["Prediksi"].value_counts())

        # ==========================
        # PDF REPORT GENERATOR
        # ==========================
        def generate_pdf(dataframe):

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer)

            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("ConcreteVision AI Report", styles["Title"]))
            elements.append(Spacer(1, 12))

            table_data = [["File", "Prediksi", "Confidence"]]

            for _, row in dataframe.iterrows():
                table_data.append([row["File"], row["Prediksi"], str(row["Confidence"])])

            table = Table(table_data)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.grey),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("GRID", (0,0), (-1,-1), 0.5, colors.black)
            ]))

            elements.append(table)
            doc.build(elements)

            buffer.seek(0)
            return buffer

        pdf = generate_pdf(df)

        st.download_button(
            "📄 Download PDF Report",
            pdf,
            "report.pdf",
            "application/pdf"
        )
