import streamlit as st
import numpy as np
import tempfile

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(
    page_title="Deteksi Retak Beton",
    page_icon="🏗️",
    layout="centered"
)

st.title("🏗️ Deteksi Retak Beton dengan CNN")

st.markdown("""
### Langkah Penggunaan
1. Upload file model `.h5`
2. Upload gambar beton
3. Sistem akan melakukan klasifikasi otomatis
""")

# Upload model
uploaded_model = st.file_uploader(
    "Upload Model CNN (.h5)",
    type=["h5"]
)

model = None

if uploaded_model is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(uploaded_model.read())
        model = load_model(tmp.name, compile=False)

    st.success("✅ Model berhasil dimuat")

# Upload gambar
uploaded_image = st.file_uploader(
    "Upload Gambar Beton",
    type=["jpg", "jpeg", "png"]
)

if model is not None and uploaded_image is not None:

    image = Image.open(uploaded_image).convert("RGB")

    st.image(
        image,
        caption="Gambar yang diupload",
        use_container_width=True
    )

    # preprocessing
    img = image.resize((150, 150))

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # jika saat training memakai normalisasi
    # img_array = img_array / 255.0

    prediction = model.predict(img_array, verbose=0)

    class_names = ["Retak", "Tidak_Retak"]

    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]

    confidence = float(prediction[0][predicted_index]) * 100

    st.subheader("Hasil Prediksi")

    st.success(
        f"Klasifikasi : {predicted_class}"
    )

    st.info(
        f"Confidence : {confidence:.2f}%"
    )
