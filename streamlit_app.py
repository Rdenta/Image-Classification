import os
import io
import streamlit as st
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# =====================================
# FUNGSI GENERATE PDF
# =====================================
def generate_pdf(df, total_img, total_retak, total_aman, persen_retak, persen_aman):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    style_title   = ParagraphStyle("title",   parent=styles["Title"],   fontSize=14, alignment=TA_CENTER, spaceAfter=4)
    style_sub     = ParagraphStyle("sub",     parent=styles["Normal"],  fontSize=10, alignment=TA_CENTER, textColor=colors.grey, spaceAfter=2)
    style_heading = ParagraphStyle("heading", parent=styles["Heading2"],fontSize=11, spaceBefore=12, spaceAfter=4)
    style_normal  = ParagraphStyle("normal",  parent=styles["Normal"],  fontSize=9)

    story = []

    # HEADER
    story.append(Paragraph("LAPORAN HASIL ANALISIS RETAK BETON", style_title))
    story.append(Paragraph("Sistem Analisis Retak Beton Berbasis AI", style_sub))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.3*cm))

    # IDENTITAS MAHASISWA
    story.append(Paragraph("Identitas Mahasiswa", style_heading))
    identitas_data = [
        ["Nama",          "Muhammad Reval Denta"],
        ["NIM",           "032400048"],
        ["Program Studi", "Elektro Mekanika"],
        ["Tanggal",       datetime.now().strftime("%d %B %Y, %H:%M:%S")],
    ]
    tbl_identitas = Table(identitas_data, colWidths=[4*cm, 12*cm])
    tbl_identitas.setStyle(TableStyle([
        ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("FONTNAME",    (0,0), (0,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,0), (0,-1),  colors.HexColor("#444444")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#F5F5F5"), colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#DDDDDD")),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING",(0,0), (-1,-1), 8),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ]))
    story.append(tbl_identitas)
    story.append(Spacer(1, 0.3*cm))

    # RINGKASAN
    story.append(Paragraph("Ringkasan Prediksi", style_heading))
    ringkasan_data = [
        ["Keterangan",       "Jumlah", "Persentase"],
        ["Total Gambar",     str(total_img),   "100%"],
        ["Terdeteksi Retak", str(total_retak), f"{persen_retak:.1f}%"],
        ["Tidak Retak",      str(total_aman),  f"{persen_aman:.1f}%"],
    ]
    tbl_ringkasan = Table(ringkasan_data, colWidths=[8*cm, 4*cm, 4*cm])
    tbl_ringkasan.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#2C3E50")),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ALIGN",       (1,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#DDDDDD")),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING",(0,0), (-1,-1), 8),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ]))
    story.append(tbl_ringkasan)
    story.append(Spacer(1, 0.3*cm))

    # TABEL DETAIL HASIL
    story.append(Paragraph("Detail Hasil Prediksi", style_heading))
    header = [["No", "Nama File", "Resolusi", "Ukuran (KB)", "Prediksi", "Confidence (%)", "Waktu Prediksi"]]
    rows   = [[
                str(i+1),
                row["File"],
                row.get("Resolusi", "-"),
                str(row.get("Ukuran (KB)", "-")),
                row["Prediksi"],
                f"{row['Confidence (%)']:.2f}%",
                row.get("Waktu Prediksi", "-")
              ]
              for i, row in df.iterrows()]
    tbl_detail = Table(header + rows, colWidths=[0.8*cm, 4.5*cm, 2*cm, 2*cm, 2.2*cm, 2.5*cm, 3*cm])
    tbl_detail.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#2C3E50")),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0), (-1,-1), 7),
        ("ALIGN",       (0,0), (0,-1),  "CENTER"),
        ("ALIGN",       (2,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#DDDDDD")),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING",(0,0), (-1,-1), 4),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        # Warna merah untuk baris Retak
        *[("TEXTCOLOR", (4, i+1), (4, i+1), colors.HexColor("#C0392B"))
          for i, row in df.iterrows() if row["Prediksi"] == "Retak"],
        *[("TEXTCOLOR", (4, i+1), (4, i+1), colors.HexColor("#27AE60"))
          for i, row in df.iterrows() if row["Prediksi"] == "Tidak_Retak"],
    ]))
    story.append(tbl_detail)
    story.append(Spacer(1, 0.5*cm))

    # FOOTER
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"Laporan dibuat otomatis oleh Sistem Analisis Retak Beton Berbasis AI "
        f"pada {datetime.now().strftime('%d %B %Y pukul %H:%M:%S')}.",
        style_normal
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


st.set_page_config(
    page_title="Sistem Analisis Retak Beton Berbasis AI",
    page_icon="🔬",
    layout="wide"
)

# =====================================
# CUSTOM CSS & TEMA
# =====================================
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');
:root{--bg:#0D1117;--card:#161B22;--card2:#1C2333;--a:#00C9A7;--a2:#FF6B6B;--a3:#FFD93D;--t:#E6EDF3;--tm:#8B949E;--b:#30363D;--r:12px}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;background:var(--bg)!important;color:var(--t)!important}
.main .block-container{padding:2rem 2.5rem 3rem!important;max-width:1200px!important}
.hero-banner{background:linear-gradient(135deg,#0D1117,#1a2744,#0D1117);border:1px solid var(--b);border-top:3px solid var(--a);border-radius:var(--r);padding:2.5rem 2rem 2rem;margin-bottom:2rem;position:relative;overflow:hidden}
.hero-banner::before,.hero-banner::after{content:'';position:absolute;border-radius:50%}
.hero-banner::before{top:-60px;right:-60px;width:220px;height:220px;background:radial-gradient(circle,rgba(0,201,167,.12),transparent 70%)}
.hero-banner::after{bottom:-40px;left:30%;width:160px;height:160px;background:radial-gradient(circle,rgba(255,107,107,.08),transparent 70%)}
.hero-title{font-family:'Space Mono',monospace!important;font-size:1.9rem!important;font-weight:700!important;color:var(--t)!important;margin:0 0 .4rem!important;letter-spacing:-.5px}
.hero-title span{color:var(--a)}
.hero-sub{font-size:.95rem;color:var(--tm);margin:0}
.hero-badge{display:inline-block;background:rgba(0,201,167,.12);border:1px solid rgba(0,201,167,.3);color:var(--a);font-size:.72rem;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;padding:3px 10px;border-radius:20px;margin-bottom:.8rem}
.section-title{font-family:'Space Mono',monospace!important;font-size:1.1rem!important;font-weight:700!important;color:var(--t)!important;border-left:4px solid var(--a);padding-left:12px;margin:1.8rem 0 1rem}
.info-card{background:var(--card);border:1px solid var(--b);border-radius:var(--r);padding:1.4rem 1.6rem;margin-bottom:1rem;transition:border-color .2s}
.info-card:hover{border-color:var(--a)}
.info-card .label{font-size:.75rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:var(--tm);margin-bottom:4px}
.info-card .value{font-size:1rem;font-weight:600;color:var(--t)}
.pred-card{background:var(--card);border:1px solid var(--b);border-radius:var(--r);padding:1rem;margin-bottom:.8rem;text-align:center}
.pred-retak{border-top:3px solid var(--a2);background:linear-gradient(180deg,rgba(255,107,107,.05),var(--card))}
.pred-aman{border-top:3px solid var(--a);background:linear-gradient(180deg,rgba(0,201,167,.05),var(--card))}
.pred-label-retak,.pred-label-aman{font-family:'Space Mono',monospace;font-size:.95rem;font-weight:700;letter-spacing:1px}
.pred-label-retak{color:var(--a2)}
.pred-label-aman{color:var(--a)}
.pred-conf,.pred-meta{font-size:.78rem;color:var(--tm);margin-top:3px}
.custom-alert{border-radius:var(--r);padding:.9rem 1.2rem;margin:.8rem 0;font-size:.88rem;font-weight:500;border-left:4px solid}
.alert-warning{background:rgba(255,217,61,.08);border-color:var(--a3);color:var(--a3)}
.alert-success{background:rgba(0,201,167,.08);border-color:var(--a);color:var(--a)}
.alert-error{background:rgba(255,107,107,.08);border-color:var(--a2);color:var(--a2)}
.alert-info{background:rgba(88,166,255,.08);border-color:#58A6FF;color:#58A6FF}
.feature-list{list-style:none;padding:0;margin:0}
.feature-list li{padding:.55rem 0;border-bottom:1px solid var(--b);font-size:.9rem;color:var(--t);display:flex;align-items:center;gap:10px}
.feature-list li:last-child{border-bottom:none}
.feature-dot{width:7px;height:7px;background:var(--a);border-radius:50%;flex-shrink:0}
.sidebar-logo{font-family:'Space Mono',monospace;font-size:1rem;font-weight:700;color:var(--a);text-align:center;padding:.5rem 0 1rem;letter-spacing:1px}
.sidebar-logo span{display:block;font-size:.72rem;color:var(--tm);font-weight:400;margin-top:2px}
[data-testid="stSidebar"]{background:var(--card)!important;border-right:1px solid var(--b)!important}
[data-testid="stSidebar"] .block-container{padding:1.5rem 1rem!important}
[data-testid="stFileUploader"]{border:2px dashed var(--b)!important;border-radius:var(--r)!important;background:var(--card2)!important;transition:border-color .2s}
[data-testid="stFileUploader"]:hover{border-color:var(--a)!important}
.stButton>button{background:var(--a)!important;color:#0D1117!important;font-family:'DM Sans',sans-serif!important;font-weight:600!important;font-size:.88rem!important;border:none!important;border-radius:8px!important;padding:.5rem 1.2rem!important;transition:opacity .2s,transform .15s!important}
.stButton>button:hover{opacity:.88!important;transform:translateY(-1px)!important}
.stDownloadButton>button{background:transparent!important;color:var(--a)!important;border:1.5px solid var(--a)!important;font-family:'DM Sans',sans-serif!important;font-weight:600!important;font-size:.88rem!important;border-radius:8px!important;transition:background .2s,color .2s!important}
.stDownloadButton>button:hover{background:var(--a)!important;color:#0D1117!important}
[data-testid="stMetric"]{background:var(--card)!important;border:1px solid var(--b)!important;border-radius:var(--r)!important;padding:1rem 1.2rem!important}
[data-testid="stMetricValue"]{font-family:'Space Mono',monospace!important;color:var(--a)!important;font-size:1.6rem!important}
[data-testid="stMetricLabel"]{color:var(--tm)!important;font-size:.8rem!important;font-weight:500!important}
[data-testid="stAlert"],[data-testid="stDataFrame"]{border-radius:var(--r)!important}
[data-testid="stDataFrame"]{border:1px solid var(--b)!important;overflow:hidden}
.stRadio [data-testid="stWidgetLabel"] p{font-weight:600!important;color:var(--tm)!important;font-size:.75rem!important;letter-spacing:1px!important;text-transform:uppercase!important}
div[data-baseweb="radio"] label{padding:.45rem .8rem!important;border-radius:8px!important;transition:background .15s!important}
div[data-baseweb="radio"] label:hover{background:rgba(0,201,167,.08)!important}
hr{border-color:var(--b)!important}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--b);border-radius:3px}::-webkit-scrollbar-thumb:hover{background:var(--a)}
</style>""", unsafe_allow_html=True)

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
st.sidebar.markdown("""
<div class="sidebar-logo">
    🔬 CRACK<span>Concrete Analysis AI</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "📌 NAVIGATION",
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

    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">AI &nbsp;·&nbsp; CNN &nbsp;·&nbsp; Computer Vision</div>
        <div class="hero-title">Sistem Analisis <span>Retak Beton</span><br>Berbasis AI</div>
        <div class="hero-sub">Deteksi dan klasifikasi keretakan permukaan beton secara otomatis menggunakan
        Convolutional Neural Network (CNN) berbasis analisis citra digital.</div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown('<div class="section-title">Fitur Aplikasi</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <ul class="feature-list">
                <li><span class="feature-dot"></span> Upload &amp; load model CNN (.h5) langsung dari browser</li>
                <li><span class="feature-dot"></span> Analisis multi-gambar beton sekaligus</li>
                <li><span class="feature-dot"></span> Dashboard statistik &amp; visualisasi hasil</li>
                <li><span class="feature-dot"></span> Auto-detect urutan label (sigmoid / softmax)</li>
                <li><span class="feature-dot"></span> Koreksi manual label jika prediksi terbalik</li>
                <li><span class="feature-dot"></span> Export laporan PDF &amp; CSV otomatis</li>
                <li><span class="feature-dot"></span> Data tersimpan antar sesi menu</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-title">Identitas Mahasiswa</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <div class="label">Nama</div>
            <div class="value">Muhammad Reval Denta</div>
        </div>
        <div class="info-card">
            <div class="label">NIM</div>
            <div class="value">032400048</div>
        </div>
        <div class="info-card">
            <div class="label">Program Studi</div>
            <div class="value">Elektro Mekanika</div>
        </div>
        <div class="custom-alert alert-success">
            &#9989; &nbsp; Aplikasi siap digunakan &mdash; upload model .h5 via sidebar untuk memulai.
        </div>
        """, unsafe_allow_html=True)

# =====================================
# PREDICT
# =====================================
if menu == "🔍 Predict":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">ANALISIS CITRA</div>
        <div class="hero-title">&#128269; Prediksi <span>Keretakan Beton</span></div>
        <div class="hero-sub">Upload gambar beton dan model CNN untuk memulai analisis otomatis.</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_images = st.file_uploader(
        "&#128194; Upload Gambar Beton (JPG / PNG)",
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

        st.markdown('<div class="section-title">📸 Hasil Prediksi</div>', unsafe_allow_html=True)

        # Tampilkan label yang sedang dipakai
        st.markdown(f"""
        <div class="custom-alert alert-info">
            &#127991;&#65039; Label aktif: Index 0 = <b>{labels[0]}</b>, Index 1 = <b>{labels[1]}</b>
            &nbsp;&mdash;&nbsp; Jika terbalik, aktifkan toggle <i>Balik urutan label</i> di sidebar.
        </div>
        """, unsafe_allow_html=True)

        col_count = st.slider("Grid Columns", 2, 4, 3)
        cols = st.columns(col_count)

        for i, img_file in enumerate(images):

            image = Image.open(img_file).convert("RGB")
            lebar, tinggi = image.size
            ukuran_file   = img_file.size / 1024  # dalam KB
            waktu_prediksi = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

            with st.spinner("🔬 Memproses gambar..."):
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
                card_class = "pred-retak" if label == "Retak" else "pred-aman"
                label_class = "pred-label-retak" if label == "Retak" else "pred-label-aman"
                icon = "⚠️" if label == "Retak" else "✔"
                st.markdown(f"""
                <div class="pred-card {card_class}">
                    <div class="{label_class}">{icon} {label.replace("_", " ")}</div>
                    <div class="pred-conf">Confidence: {conf:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.image(image, use_container_width=True)
                st.progress(conf / 100)
                st.markdown(f"""
                <div class="pred-meta">
                    📁 {img_file.name}<br>
                    📐 {lebar}x{tinggi} px &nbsp;|&nbsp; 💾 {ukuran_file:.1f} KB<br>
                    🕐 {waktu_prediksi}
                </div>
                """, unsafe_allow_html=True)

            results.append({
                "File"           : img_file.name,
                "Ukuran (KB)"    : round(ukuran_file, 1),
                "Resolusi"       : f"{lebar}x{tinggi}",
                "Prediksi"       : label,
                "Confidence (%)" : round(conf, 2),
                "Waktu Prediksi" : waktu_prediksi
            })

        df = pd.DataFrame(results)

        # Cegah double-append history
        image_names = sorted([f.name for f in images])
        current_key = str(image_names)

        if current_key != st.session_state.last_predicted_key:
            st.session_state.history.append(df)
            st.session_state.last_predicted_key = current_key

        st.markdown("---")

        # =====================================
        # RINGKASAN PREDIKSI OTOMATIS
        # =====================================
        total_img   = len(df)
        total_retak = len(df[df["Prediksi"] == "Retak"])
        total_aman  = len(df[df["Prediksi"] == "Tidak_Retak"])
        persen_retak = (total_retak / total_img * 100) if total_img > 0 else 0
        persen_aman  = (total_aman  / total_img * 100) if total_img > 0 else 0

        st.markdown('<div class="section-title">📋 Ringkasan Hasil Prediksi</div>', unsafe_allow_html=True)

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("🖼️ Total Gambar", total_img)
        col_r2.metric("⚠️ Retak", f"{total_retak} ({persen_retak:.1f}%)")
        col_r3.metric("✅ Tidak Retak", f"{total_aman} ({persen_aman:.1f}%)")

        if total_retak == 0:
            st.markdown(f'''<div class="custom-alert alert-success">&#9989; Dari <b>{total_img}</b> gambar yang dianalisis, <b>tidak ditemukan keretakan</b> pada seluruh sampel beton.</div>''', unsafe_allow_html=True)
        elif total_aman == 0:
            st.markdown(f'''<div class="custom-alert alert-error">&#9888;&#65039; Dari <b>{total_img}</b> gambar yang dianalisis, <b>seluruh sampel terdeteksi retak</b> dan perlu penanganan lebih lanjut.</div>''', unsafe_allow_html=True)
        else:
            st.markdown(f'''<div class="custom-alert alert-warning">&#128270; Dari <b>{total_img}</b> gambar, ditemukan <b>{total_retak} retak ({persen_retak:.1f}%)</b> dan <b>{total_aman} tidak retak ({persen_aman:.1f}%)</b>. Segera periksa sampel yang terdeteksi retak.</div>''', unsafe_allow_html=True)

        st.markdown("---")

        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            st.download_button(
                "📥 Download CSV",
                df.to_csv(index=False).encode(),
                "hasil_prediksi.csv",
                "text/csv",
                use_container_width=True
            )

        with col_dl2:
            pdf_buffer = generate_pdf(
                df, total_img, total_retak, total_aman, persen_retak, persen_aman
            )
            st.download_button(
                "📄 Download Laporan PDF",
                pdf_buffer,
                f"laporan_retak_beton_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                "application/pdf",
                use_container_width=True
            )

    else:
        st.markdown('<div class="custom-alert alert-info">&#8505;&#65039; &nbsp; Upload model (.h5) via sidebar dan pilih gambar beton untuk memulai analisis.</div>', unsafe_allow_html=True)

# =====================================
# ANALYTICS
# =====================================
if menu == "📊 Analytics":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">STATISTIK</div>
        <div class="hero-title">&#128202; Dashboard <span>Analisis</span></div>
        <div class="hero-sub">Ringkasan dan statistik dari seluruh sesi prediksi.</div>
    </div>
    """, unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.markdown('<div class="custom-alert alert-warning">&#9888;&#65039; &nbsp; Belum ada data prediksi. Lakukan prediksi di menu Predict terlebih dahulu.</div>', unsafe_allow_html=True)
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

        st.markdown('<div class="section-title">🔎 Filter Confidence</div>', unsafe_allow_html=True)

        min_conf = st.slider("Minimum Confidence (%)", 0, 100, 50)

        filtered = df[df["Confidence (%)"] >= min_conf]

        st.dataframe(filtered, use_container_width=True)

# =====================================
# ABOUT
# =====================================
if menu == "ℹ️ About":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">ABOUT PROJECT</div>
        <div class="hero-title">&#8505;&#65039; Tentang <span>Aplikasi</span></div>
        <div class="hero-sub">Sistem Analisis Retak Beton Berbasis AI &mdash; CNN · Computer Vision · Streamlit</div>
    </div>
    """, unsafe_allow_html=True)

    col_a1, col_a2 = st.columns([1.2, 1])

    with col_a1:
        st.markdown('<div class="section-title">Deskripsi Proyek</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <div class="label">Tentang</div>
            <div class="value" style="font-size:0.9rem; font-weight:400; line-height:1.6; margin-top:6px;">
                Aplikasi berbasis <b>Convolutional Neural Network (CNN)</b> untuk mendeteksi dan
                mengklasifikasikan keretakan pada permukaan beton secara otomatis melalui analisis
                citra digital. Dibangun menggunakan Streamlit dan TensorFlow.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Fitur Teknis</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <ul class="feature-list">
                <li><span class="feature-dot"></span> Persistent image upload antar menu</li>
                <li><span class="feature-dot"></span> Session-based history (no duplicate entries)</li>
                <li><span class="feature-dot"></span> Streamlit Cloud deployment ready</li>
                <li><span class="feature-dot"></span> Auto-detect label order (sigmoid &amp; softmax)</li>
                <li><span class="feature-dot"></span> Manual label swap toggle</li>
                <li><span class="feature-dot"></span> Export PDF &amp; CSV laporan otomatis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_a2:
        st.markdown('<div class="section-title">Identitas Mahasiswa</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <div class="label">Nama</div>
            <div class="value">Muhammad Reval Denta</div>
        </div>
        <div class="info-card">
            <div class="label">NIM</div>
            <div class="value">032400048</div>
        </div>
        <div class="info-card">
            <div class="label">Program Studi</div>
            <div class="value">Elektro Mekanika</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Teknologi</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <ul class="feature-list">
                <li><span class="feature-dot"></span> Python &amp; Streamlit</li>
                <li><span class="feature-dot"></span> TensorFlow / Keras (CNN)</li>
                <li><span class="feature-dot"></span> NumPy &amp; Pandas</li>
                <li><span class="feature-dot"></span> ReportLab (PDF)</li>
                <li><span class="feature-dot"></span> Pillow (Image Processing)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('''<div class="custom-alert alert-success">&#128640; &nbsp; <b>Ready for Deployment</b> &mdash; Upload ke GitHub dan deploy di Streamlit Cloud.</div>''', unsafe_allow_html=True)
