import streamlit as st
import numpy as np
import pandas as pd
import tempfile

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# =====================================
# KONFIGURASI HALAMAN
# =====================================

st.set_page_config(
    page_title="ConcreteVision",
    page_icon="🧠",
    layout="wide"
)

# =====================================
# HEADER
# =====================================

st.title("🧠🏗️ ConcreteVision: Sistem Analisis Retak Beton Berbasis AI")

st.caption(
    "Platform berbasis Artificial Intelligence untuk klasifikasi dan analisis kondisi permukaan beton menggunakan Convolutional Neural Network (CNN)."
)

# =====================================
# IDENTITAS MAHASISWA
# =====================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Nama", "Muhammad Reval Denta")

with col2:
    st.metric("NIM", "032400048")

with col3:
    st.metric("Prodi", "Elektro Mekanika")

st.markdown("---")

# =====================================
# PETUNJUK
# =====================================

st.markdown("""
### Cara Penggunaan
1. Upload model CNN (.h5)
2. Upload satu atau banyak gambar beton
3. Sistem akan melakukan klasifikasi otomatis
4. Lihat hasil analisis pada dashboard
5. Unduh hasil prediksi dalam format CSV
""")

# =====================================
# UPLOAD MODEL
# =====================================

uploaded_model = st.file_uploader(
    "📁 Upload Model CNN (.h5)",
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

# =====================================
# UPLOAD GAMBAR
# =====================================

uploaded_images = st.file_uploader(
    "🖼️ Upload Gambar Beton",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# =====================================
# PREDIKSI
# =====================================

if model is not None and uploaded_images:

    hasil_prediksi = []

    st.header("🔍 Hasil Prediksi")

    for uploaded_image in uploaded_images:

        image = Image.open(uploaded_image).convert("RGB")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(
                image,
                caption=uploaded_image.name,
                use_container_width=True
            )

        # =====================================
        # PREPROCESSING
        # =====================================

        img = image.resize((150, 150))

        img_array = img_to_array(img)

        # Aktifkan jika saat training dilakukan normalisasi
        # img_array = img_array / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        # =====================================
        # PREDIKSI
        # =====================================

        prediction = model.predict(img_array, verbose=0)

        class_names = ["Retak", "Tidak_Retak"]

        predicted_index = np.argmax(prediction[0])
        predicted_class = class_names[predicted_index]

        confidence = float(prediction[0][predicted_index]) * 100

        with col2:

            st.subheader(uploaded_image.name)

            if predicted_class == "Retak":
                st.error(f"⚠️ Hasil: {predicted_class}")
            else:
                st.success(f"✅ Hasil: {predicted_class}")

            st.info(f"Confidence: {confidence:.2f}%")

            st.progress(min(confidence / 100, 1.0))

        hasil_prediksi.append({
            "Nama File": uploaded_image.name,
            "Prediksi": predicted_class,
            "Confidence (%)": round(confidence, 2)
        })

        st.divider()

    # =====================================
    # DATAFRAME HASIL
    # =====================================

    df = pd.DataFrame(hasil_prediksi)

    # =====================================
    # DASHBOARD ANALISIS
    # =====================================

    st.header("📊 Dashboard Analisis")

    total = len(df)

    jumlah_retak = len(
        df[df["Prediksi"] == "Retak"]
    )

    jumlah_normal = len(
        df[df["Prediksi"] == "Tidak_Retak"]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Gambar", total)

    with col2:
        st.metric("Retak", jumlah_retak)

    with col3:
        st.metric("Tidak Retak", jumlah_normal)

    # =====================================
    # PERSENTASE
    # =====================================

    persen_retak = (
        jumlah_retak / total * 100
    ) if total > 0 else 0

    persen_normal = (
        jumlah_normal / total * 100
    ) if total > 0 else 0

    st.subheader("📈 Persentase Hasil")

    st.write(f"🔴 Retak : {persen_retak:.1f}%")
    st.progress(persen_retak / 100)

    st.write(f"🟢 Tidak Retak : {persen_normal:.1f}%")
    st.progress(persen_normal / 100)

    # =====================================
    # GRAFIK DISTRIBUSI
    # =====================================

    st.subheader("📊 Grafik Distribusi")

    summary = df["Prediksi"].value_counts()

    st.bar_chart(summary)

    # =====================================
    # FILTER DATA
    # =====================================

    st.subheader("🔎 Filter Data")

    filter_hasil = st.selectbox(
        "Pilih Kategori",
        ["Semua", "Retak", "Tidak_Retak"]
    )

    if filter_hasil == "Semua":
        filtered_df = df
    else:
        filtered_df = df[
            df["Prediksi"] == filter_hasil
        ]

    # =====================================
    # TABEL HASIL
    # =====================================

    st.subheader("📋 Ringkasan Hasil")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    # =====================================
    # DOWNLOAD CSV
    # =====================================

    csv = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Hasil CSV",
        data=csv,
        file_name="hasil_prediksi_beton.csv",
        mime="text/csv"
    )
