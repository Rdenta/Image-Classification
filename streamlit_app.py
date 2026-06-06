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
    page_title="ConcreteVision AI — Sistem Deteksi Keretakan Beton",
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

if "labels" not in st.session_state:
    st.session_state.labels = ["Retak", "Tidak_Retak"]  # default

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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(uploaded_model.read())
            tmp_path = tmp.name

        st.session_state.model = load_model(tmp_path, compile=False)
        os.unlink(tmp_path)

        # =============================================
        # AUTO-DETECT LABEL ORDER dari model
        # =============================================
        model_obj = st.session_state.model
        detected_labels = ["Retak", "Tidak_Retak"]  # default fallback

        # Coba baca class names dari output layer model
        try:
            output_layer = model_obj.layers[-1]
            num_classes = output_layer.output_shape[-1]

            # Coba ambil dari names jika tersedia
            if hasattr(output_layer, 'class_names'):
                detected_labels = output_layer.class_names
            else:
                # Jika 2 kelas, gunakan default tapi tampilkan info
                if num_classes == 2:
                    detected_labels = ["Retak", "Tidak_Retak"]
        except Exception:
            pass

        st.session_state.labels = detected_labels

        st.sidebar.success("Model loaded ✅")

    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")

model = st.session_state.model
labels = st.session_state.labels

# =====================================
# MODEL STATUS INFO
# =====================================
if model is None:
    st.sidebar.warning("⚠️ Model belum diupload")
else:
    st.sidebar.success("🧠 Model siap digunakan")

    # Tampilkan info label order di sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🏷️ Urutan Label Model:**")
    for i, lbl in enumerate(labels):
        st.sidebar.markdown(f"- Index `{i}` → `{lbl}`")

    # Izinkan user koreksi manual jika label terdeteksi terbalik
    st.sidebar.markdown("---")
    st.sidebar.markdown("**⚙️ Koreksi Label (jika prediksi terbalik):**")
    swap = st.sidebar.toggle("🔄 Balik urutan label")
    if swap:
        labels = labels[::-1]
        st.session_state.labels = labels

# =====================================
# HOME
# =====================================
if menu == "🏠 Home":
    st.title("🧠 ConcreteVision AI — Sistem Deteksi Keretakan Beton")

    st.markdown("---")

    st.markdown("""
    AI untuk klasifikasi retak beton menggunakan CNN.

    ### Fitur:
    - Upload model AI
    - Analisis gambar
    - Dashboard statistik
    - Data tersimpan antar menu
    - Auto-detect & koreksi urutan label
    """)

    st.markdown("---")

    st.markdown("### 👤 Identitas Mahasiswa")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Nama**")
        st.markdown("**NIM**")
        st.markdown("**Program Studi**")
    with col2:
        st.markdown(": Muhammad Reval Denta")
        st.markdown(": 032400048")
        st.markdown(": Elektro Mekanika")

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
        st.session_state.uploader_key += 1
        st.rerun()

    if model is not None and len(images) > 0:

        results = []

        st.subheader("📸 Hasil Prediksi")

        # Tampilkan label yang sedang dipakai
        st.info(f"🏷️ Label aktif: Index 0 = **{labels[0]}**, Index 1 = **{labels[1]}** — Jika terbalik, aktifkan toggle 'Balik urutan label' di sidebar.")

        col_count = st.slider("Grid Columns", 2, 4, 3)
        cols = st.columns(col_count)

        for i, img_file in enumerate(images):

            image = Image.open(img_file).convert("RGB")

            with st.spinner("🧠 AI processing..."):
                img = image.resize((150, 150))
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array, verbose=0)

                output = prediction[0]

                # Handle sigmoid (1 output) vs softmax (2 output)
                if len(output) == 1:
                    # Sigmoid: nilai mendekati 1 = Tidak_Retak, mendekati 0 = Retak
                    # Threshold 0.5: lebih sensitif mendeteksi retak (lebih aman)
                    sigmoid_val = float(output[0])
                    threshold = 0.5
                    if sigmoid_val >= threshold:
                        label = labels[1]   # Tidak_Retak
                        conf = sigmoid_val * 100
                    else:
                        label = labels[0]   # Retak
                        conf = (1 - sigmoid_val) * 100
                else:
                    # Softmax: ambil index tertinggi
                    idx = np.argmax(output)
                    label = labels[idx]
                    conf = float(output[idx]) * 100

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

        # Cegah double-append history
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
    **ConcreteVision AI — Sistem Deteksi Keretakan Beton**

    Aplikasi berbasis Deep Learning (CNN) untuk mendeteksi keretakan pada permukaan beton secara otomatis melalui analisis citra digital.

    ✔ Persistent image upload
    ✔ Session-based history (no duplicate entries)
    ✔ Streamlit deployment ready
    ✔ Reset feature added
    ✔ Auto-detect label order
    ✔ Manual label swap toggle
    """)

    st.markdown("---")

    st.markdown("### 👤 Identitas Mahasiswa")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Nama**")
        st.markdown("**NIM**")
        st.markdown("**Program Studi**")
    with col2:
        st.markdown(": Muhammad Reval Denta")
        st.markdown(": 032400048")
        st.markdown(": Elektro Mekanika")

    st.markdown("---")
    st.success("Ready 🚀")
