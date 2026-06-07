import os
import io
import streamlit as st
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

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
# FUNGSI GRAFIK PDF
# =====================================
def _make_pie_chart_img(total_retak, total_aman):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3.2), facecolor="white")
    ax.pie([total_retak, total_aman],
           labels=["Retak", "Tidak Retak"],
           colors=["#C0392B", "#27AE60"],
           autopct="%1.1f%%", startangle=90,
           wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
           textprops={"fontsize": 10})
    ax.set_title("Distribusi Hasil Prediksi", fontsize=11, fontweight="bold", pad=10)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf

def _make_bar_chart_img(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names  = [n[:18] + "..." if len(n) > 18 else n for n in df["File"].tolist()]
    confs  = df["Confidence (%)"].tolist()
    clrs   = ["#C0392B" if p == "Retak" else "#27AE60" for p in df["Prediksi"].tolist()]
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(names) * 0.45)), facecolor="white")
    bars = ax.barh(names, confs, color=clrs, edgecolor="white", linewidth=0.8)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", fontsize=9)
    ax.set_title("Confidence Per Gambar", fontsize=11, fontweight="bold")
    ax.axvline(50, color="#888", linestyle="--", linewidth=0.8, label="Threshold 50%")
    for bar, v in zip(bars, confs):
        ax.text(v + 1, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8)
    ax.legend(fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf

# =====================================
# FUNGSI GENERATE PDF
# =====================================
def generate_pdf(df, total_img, total_retak, total_aman, persen_retak, persen_aman):
    from reportlab.platypus import Image as RLImage
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    style_title   = ParagraphStyle("title",   parent=styles["Title"],   fontSize=14, alignment=TA_CENTER, spaceAfter=4)
    style_sub     = ParagraphStyle("sub",     parent=styles["Normal"],  fontSize=10, alignment=TA_CENTER, textColor=colors.grey, spaceAfter=2)
    style_heading = ParagraphStyle("heading", parent=styles["Heading2"], fontSize=11, spaceBefore=12, spaceAfter=4)
    style_normal  = ParagraphStyle("normal",  parent=styles["Normal"],  fontSize=9)

    story = []

    # HEADER
    story.append(Paragraph("LAPORAN HASIL ANALISIS RETAK BETON", style_title))
    story.append(Paragraph("Sistem Analisis Retak Beton Berbasis AI", style_sub))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.3*cm))

    # IDENTITAS
    story.append(Paragraph("Identitas Mahasiswa", style_heading))
    tbl_identitas = Table([
        ["Nama",          "Muhammad Reval Denta"],
        ["NIM",           "032400048"],
        ["Program Studi", "Elektro Mekanika"],
        ["Tanggal",       datetime.now().strftime("%d %B %Y, %H:%M:%S")],
    ], colWidths=[4*cm, 12*cm])
    tbl_identitas.setStyle(TableStyle([
        ("FONTNAME",       (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",       (0,0), (-1,-1), 9),
        ("FONTNAME",       (0,0), (0,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",      (0,0), (0,-1),  colors.HexColor("#444444")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#F5F5F5"), colors.white]),
        ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#DDDDDD")),
        ("LEFTPADDING",    (0,0), (-1,-1), 8),
        ("RIGHTPADDING",   (0,0), (-1,-1), 8),
        ("TOPPADDING",     (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
    ]))
    story.append(tbl_identitas)
    story.append(Spacer(1, 0.3*cm))

    # RINGKASAN
    story.append(Paragraph("Ringkasan Prediksi", style_heading))
    tbl_ringkasan = Table([
        ["Keterangan",       "Jumlah", "Persentase"],
        ["Total Gambar",     str(total_img),   "100%"],
        ["Terdeteksi Retak", str(total_retak), f"{persen_retak:.1f}%"],
        ["Tidak Retak",      str(total_aman),  f"{persen_aman:.1f}%"],
    ], colWidths=[8*cm, 4*cm, 4*cm])
    tbl_ringkasan.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0),  colors.HexColor("#2C3E50")),
        ("TEXTCOLOR",      (0,0), (-1,0),  colors.white),
        ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTNAME",       (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",       (0,0), (-1,-1), 9),
        ("ALIGN",          (1,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#DDDDDD")),
        ("LEFTPADDING",    (0,0), (-1,-1), 8),
        ("RIGHTPADDING",   (0,0), (-1,-1), 8),
        ("TOPPADDING",     (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
    ]))
    story.append(tbl_ringkasan)
    story.append(Spacer(1, 0.3*cm))

    # GRAFIK
    story.append(Paragraph("Visualisasi Hasil Prediksi", style_heading))
    try:
        pie_buf = _make_pie_chart_img(total_retak, total_aman)
        pie_img = RLImage(pie_buf, width=7*cm, height=5.6*cm)
        bar_buf = _make_bar_chart_img(df)
        bar_h   = max(4*cm, min(len(df) * 1.1*cm, 10*cm))
        bar_img = RLImage(bar_buf, width=9*cm, height=bar_h)
        chart_tbl = Table([[pie_img, bar_img]], colWidths=[8*cm, 9*cm])
        chart_tbl.setStyle(TableStyle([
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("ALIGN",        (0,0), (-1,-1), "CENTER"),
            ("LEFTPADDING",  (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(chart_tbl)
    except Exception:
        story.append(Paragraph("(Grafik tidak tersedia)", style_normal))
    story.append(Spacer(1, 0.4*cm))

    # DETAIL
    story.append(Paragraph("Detail Hasil Prediksi", style_heading))
    header = [["No", "Nama File", "Resolusi", "Ukuran (KB)", "Prediksi", "Confidence (%)", "Waktu Prediksi"]]
    rows   = [[str(i+1), row["File"], row.get("Resolusi", "-"),
               str(row.get("Ukuran (KB)", "-")), row["Prediksi"],
               f"{row['Confidence (%)']:.2f}%", row.get("Waktu Prediksi", "-")]
              for i, row in df.iterrows()]
    tbl_detail = Table(header + rows, colWidths=[0.8*cm, 4.5*cm, 2*cm, 2*cm, 2.2*cm, 2.5*cm, 3*cm])
    tbl_detail.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0),  colors.HexColor("#2C3E50")),
        ("TEXTCOLOR",      (0,0), (-1,0),  colors.white),
        ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTNAME",       (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",       (0,0), (-1,-1), 7),
        ("ALIGN",          (0,0), (0,-1),  "CENTER"),
        ("ALIGN",          (2,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#DDDDDD")),
        ("LEFTPADDING",    (0,0), (-1,-1), 4),
        ("RIGHTPADDING",   (0,0), (-1,-1), 4),
        ("TOPPADDING",     (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 4),
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


# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Sistem Analisis Retak Beton Berbasis AI",
    page_icon="🔬",
    layout="wide"
)

# =====================================
# SESSION STATE
# =====================================
if "history"           not in st.session_state: st.session_state.history           = []
if "images"            not in st.session_state: st.session_state.images            = []
if "model"             not in st.session_state: st.session_state.model             = None
if "labels"            not in st.session_state: st.session_state.labels            = ["Retak", "Tidak_Retak"]
if "last_predicted_key" not in st.session_state: st.session_state.last_predicted_key = None
if "uploader_key"      not in st.session_state: st.session_state.uploader_key      = 0

# =====================================
# SIDEBAR
# =====================================
st.sidebar.title("🔬 Analisis Retak Beton")
st.sidebar.markdown("---")

menu = st.sidebar.radio("📌 Navigation", ["🏠 Home", "🔍 Predict", "📊 Analytics", "ℹ️ About"])

st.sidebar.markdown("---")

# =====================================
# MODEL UPLOAD
# =====================================
uploaded_model = st.sidebar.file_uploader("📁 Upload Model (.h5)", type=["h5"])

if uploaded_model is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(uploaded_model.read())
            tmp_path = tmp.name
        st.session_state.model = load_model(tmp_path, compile=False)
        os.unlink(tmp_path)

        # Auto-detect label order
        try:
            output_layer = st.session_state.model.layers[-1]
            num_classes  = output_layer.output_shape[-1]
            if hasattr(output_layer, "class_names"):
                st.session_state.labels = output_layer.class_names
            elif num_classes == 2:
                st.session_state.labels = ["Retak", "Tidak_Retak"]
        except Exception:
            pass

        st.sidebar.success("Model loaded ✅")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")

model  = st.session_state.model
labels = st.session_state.labels

# Model status
if model is None:
    st.sidebar.warning("⚠️ Model belum diupload")
else:
    st.sidebar.success("🧠 Model siap digunakan")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🏷️ Urutan Label Model:**")
    for i, lbl in enumerate(labels):
        st.sidebar.markdown(f"- Index `{i}` → `{lbl}`")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**⚙️ Koreksi Label (jika prediksi terbalik):**")
    if st.sidebar.toggle("🔄 Balik urutan label"):
        labels = labels[::-1]
        st.session_state.labels = labels

# =====================================
# HOME
# =====================================
if menu == "🏠 Home":
    st.title("🔬 Sistem Analisis Retak Beton Berbasis AI")
    st.markdown("Aplikasi klasifikasi keretakan permukaan beton menggunakan **Convolutional Neural Network (CNN)**.")
    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("⚙️ Fitur Aplikasi")
        st.markdown("""
        - Upload & load model CNN (.h5) langsung dari browser
        - Analisis multi-gambar beton sekaligus
        - Threshold confidence yang bisa diatur
        - Dashboard statistik & visualisasi interaktif
        - Auto-detect urutan label (sigmoid / softmax)
        - Koreksi manual label jika prediksi terbalik
        - Export laporan PDF (dengan grafik) & CSV
        - Data tersimpan antar sesi menu
        """)

    with col_right:
        st.subheader("👤 Identitas Mahasiswa")
        st.markdown("""
        | Keterangan | Detail |
        |---|---|
        | **Nama** | Muhammad Reval Denta |
        | **NIM** | 032400048 |
        | **Program Studi** | Elektro Mekanika |
        """)
        st.success("✅ Aplikasi siap digunakan — upload model .h5 via sidebar untuk memulai.")

# =====================================
# PREDICT
# =====================================
if menu == "🔍 Predict":
    st.title("🔍 Prediksi Keretakan Beton")

    uploaded_images = st.file_uploader(
        "📂 Upload Gambar Beton (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_images:
        st.session_state.images = uploaded_images

    images = st.session_state.images

    colA, colB = st.columns(2)
    with colA:
        if st.button("🔄 Reset Gambar"):
            st.session_state.images            = []
            st.session_state.history           = []
            st.session_state.last_predicted_key = None
            st.session_state.uploader_key      += 1
            st.rerun()

    if model is not None and len(images) > 0:
        results = []

        st.subheader("📸 Hasil Prediksi")
        st.info(f"🏷️ Label aktif: Index 0 = **{labels[0]}**, Index 1 = **{labels[1]}** — Jika terbalik, aktifkan toggle di sidebar.")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            col_count = st.slider("Grid Columns", 2, 4, 3)
        with col_s2:
            threshold = st.slider("⚙️ Threshold Confidence (%)", 30, 90, 50,
                help="Batas confidence untuk klasifikasi. Default: 50%")
        cols = st.columns(col_count)

        for i, img_file in enumerate(images):
            image          = Image.open(img_file).convert("RGB")
            lebar, tinggi  = image.size
            ukuran_file    = img_file.size / 1024
            waktu_prediksi = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

            with st.spinner("🔬 Memproses gambar..."):
                img_arr    = img_to_array(image.resize((150, 150)))
                img_arr    = np.expand_dims(img_arr, axis=0)
                output     = model.predict(img_arr, verbose=0)[0]
                thresh_val = threshold / 100.0

                if len(output) == 1:
                    sigmoid_val = float(output[0])
                    if sigmoid_val >= thresh_val:
                        label = labels[1]
                        conf  = sigmoid_val * 100
                    else:
                        label = labels[0]
                        conf  = (1 - sigmoid_val) * 100
                else:
                    idx   = np.argmax(output)
                    label = labels[idx]
                    conf  = float(output[idx]) * 100

            with cols[i % col_count]:
                if label == "Retak":
                    st.error(f"⚠️ Retak — {conf:.2f}%")
                else:
                    st.success(f"✔ Tidak Retak — {conf:.2f}%")
                st.image(image, use_container_width=True)
                st.progress(conf / 100)
                st.caption(f"📁 {img_file.name}")
                st.caption(f"📐 {lebar}x{tinggi} px  |  💾 {ukuran_file:.1f} KB  |  🕐 {waktu_prediksi}")

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
        current_key = str(sorted([f.name for f in images]))
        if current_key != st.session_state.last_predicted_key:
            st.session_state.history.append(df)
            st.session_state.last_predicted_key = current_key

        st.markdown("---")
        st.subheader("📋 Ringkasan Hasil Prediksi")

        total_img    = len(df)
        total_retak  = len(df[df["Prediksi"] == "Retak"])
        total_aman   = len(df[df["Prediksi"] == "Tidak_Retak"])
        persen_retak = (total_retak / total_img * 100) if total_img > 0 else 0
        persen_aman  = (total_aman  / total_img * 100) if total_img > 0 else 0

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("🖼️ Total Gambar", total_img)
        col_r2.metric("⚠️ Retak",        f"{total_retak} ({persen_retak:.1f}%)")
        col_r3.metric("✅ Tidak Retak",  f"{total_aman} ({persen_aman:.1f}%)")

        if total_retak == 0:
            st.success(f"✅ Dari **{total_img}** gambar, **tidak ditemukan keretakan** pada seluruh sampel.")
        elif total_aman == 0:
            st.error(f"⚠️ Dari **{total_img}** gambar, **seluruh sampel terdeteksi retak** dan perlu penanganan lebih lanjut.")
        else:
            st.warning(f"🔎 Ditemukan **{total_retak} retak ({persen_retak:.1f}%)** dan **{total_aman} tidak retak ({persen_aman:.1f}%)** dari {total_img} gambar.")

        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button("📥 Download CSV", df.to_csv(index=False).encode(),
                               "hasil_prediksi.csv", "text/csv", use_container_width=True)
        with col_dl2:
            pdf_buffer = generate_pdf(df, total_img, total_retak, total_aman, persen_retak, persen_aman)
            st.download_button("📄 Download Laporan PDF", pdf_buffer,
                               f"laporan_retak_beton_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                               "application/pdf", use_container_width=True)
    else:
        st.info("ℹ️ Upload model (.h5) via sidebar dan pilih gambar beton untuk memulai analisis.")

# =====================================
# ANALYTICS
# =====================================
if menu == "📊 Analytics":
    st.title("📊 Dashboard Analisis")

    if len(st.session_state.history) == 0:
        st.warning("⚠️ Belum ada data prediksi. Lakukan prediksi di menu Predict terlebih dahulu.")
    else:
        df     = pd.concat(st.session_state.history, ignore_index=True)
        total  = len(df)
        retak  = len(df[df["Prediksi"] == "Retak"])
        normal = len(df[df["Prediksi"] == "Tidak_Retak"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Gambar", total)
        col2.metric("Retak",        retak)
        col3.metric("Tidak Retak",  normal)

        st.subheader("📊 Visualisasi Distribusi")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            pie_fig = go.Figure(data=[go.Pie(
                labels=["Retak", "Tidak Retak"],
                values=[retak, normal],
                hole=0.45,
                marker=dict(colors=["#FF6B6B", "#00C9A7"], line=dict(color="white", width=2)),
                textinfo="label+percent",
                textfont=dict(size=13),
                hovertemplate="<b>%{label}</b><br>Jumlah: %{value}<br>%{percent}<extra></extra>"
            )])
            pie_fig.update_layout(
                title=dict(text="Distribusi Prediksi", font=dict(size=14), x=0.5),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(t=50, b=20, l=10, r=10), height=320
            )
            st.plotly_chart(pie_fig, use_container_width=True)

        with chart_col2:
            vc = df["Prediksi"].value_counts().reset_index()
            vc.columns = ["Prediksi", "Jumlah"]
            bar_fig = go.Figure(data=[go.Bar(
                x=vc["Prediksi"],
                y=vc["Jumlah"],
                marker_color=["#FF6B6B" if p == "Retak" else "#00C9A7" for p in vc["Prediksi"]],
                text=vc["Jumlah"], textposition="outside",
                hovertemplate="<b>%{x}</b><br>Jumlah: %{y}<extra></extra>"
            )])
            bar_fig.update_layout(
                title=dict(text="Jumlah Per Kelas", font=dict(size=14), x=0.5),
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="#EEEEEE"),
                margin=dict(t=50, b=20, l=10, r=10), height=320
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        st.subheader("📈 Distribusi Confidence")
        hist_fig = px.histogram(
            df, x="Confidence (%)", color="Prediksi", nbins=20,
            color_discrete_map={"Retak": "#FF6B6B", "Tidak_Retak": "#00C9A7"},
            barmode="overlay", opacity=0.8,
            labels={"Confidence (%)": "Confidence (%)", "count": "Jumlah Gambar"}
        )
        hist_fig.update_layout(
            legend=dict(title="Prediksi"),
            margin=dict(t=20, b=20, l=10, r=10), height=280
        )
        st.plotly_chart(hist_fig, use_container_width=True)

        st.subheader("🔎 Filter & Tabel Data")
        fc1, fc2 = st.columns(2)
        with fc1:
            min_conf     = st.slider("Minimum Confidence (%)", 0, 100, 50)
        with fc2:
            filter_label = st.selectbox("Filter Prediksi", ["Semua", "Retak", "Tidak_Retak"])

        filtered = df[df["Confidence (%)"] >= min_conf]
        if filter_label != "Semua":
            filtered = filtered[filtered["Prediksi"] == filter_label]

        st.caption(f"Menampilkan {len(filtered)} dari {len(df)} data")
        st.dataframe(filtered, use_container_width=True)

# =====================================
# ABOUT
# =====================================
if menu == "ℹ️ About":
    st.title("ℹ️ About Project")
    st.markdown("---")

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.subheader("🔬 Deskripsi Proyek")
        st.markdown("""
        Aplikasi berbasis **Convolutional Neural Network (CNN)** untuk mendeteksi dan
        mengklasifikasikan keretakan pada permukaan beton secara otomatis melalui
        analisis citra digital. Dibangun menggunakan Streamlit dan TensorFlow.
        """)

        st.subheader("⚙️ Fitur Teknis")
        st.markdown("""
        - Persistent image upload antar menu
        - Session-based history (no duplicate entries)
        - Streamlit Cloud deployment ready
        - Auto-detect label order (sigmoid & softmax)
        - Manual label swap toggle
        - Threshold confidence yang bisa diatur
        - Export PDF (dengan grafik) & CSV
        """)

    with col_a2:
        st.subheader("👤 Identitas Mahasiswa")
        st.markdown("""
        | Keterangan | Detail |
        |---|---|
        | **Nama** | Muhammad Reval Denta |
        | **NIM** | 032400048 |
        | **Program Studi** | Elektro Mekanika |
        """)

        st.subheader("🛠️ Teknologi")
        st.markdown("""
        | Library | Kegunaan |
        |---|---|
        | Streamlit | Framework UI |
        | TensorFlow / Keras | Model CNN |
        | Plotly | Grafik interaktif |
        | Matplotlib | Grafik PDF |
        | ReportLab | Generate PDF |
        | Pillow | Image processing |
        """)

        st.success("🚀 Ready for Deployment — Upload ke GitHub dan deploy di Streamlit Cloud.")
