import streamlit as st
import zipfile, tempfile, os, re, io
import pydicom
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from math import ceil
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ==================== KONFIGURASI HALAMAN ====================
st.set_page_config(page_title="MRI DICOM Dashboard", page_icon="🩻", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {transition: none !important;}
[data-testid="stImage"] {transition: none !important;}
button[kind="primary"] {
    background-color: #1E90FF !important;
    color: white !important;
    border-radius: 6px !important;
    padding: 0.4rem 1rem !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🩻 MRI DICOM Dashboard (ZIP per TE/TR)")

# ==================== INISIALISASI SESSION STATE ====================
if "dicom_data" not in st.session_state:
    st.session_state["dicom_data"] = {}
if "window_params_by_file" not in st.session_state:
    st.session_state["window_params_by_file"] = {}
if "segment_results" not in st.session_state:
    st.session_state["segment_results"] = {}
if "current_tab" not in st.session_state:
    st.session_state["current_tab"] = "1️⃣ Viewer"

# ==================== HELPER FUNCTIONS ====================
def extract_te_tr_from_filename(filename: str):
    match = re.search(r'TE[_\-]?(\d+\.?\d*)[_\-]?.*TR[_\-]?(\d+\.?\d*)', filename)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

@st.cache_data(show_spinner=False)
def load_dicom_from_zip_bytes(zip_bytes: bytes, filename: str):
    TE, TR = extract_te_tr_from_filename(filename)
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        dicom_files = list(Path(tmpdir).rglob("*.dcm"))
        if not dicom_files:
            return TE, TR, []
        dicoms = []
        for f in dicom_files:
            try:
                ds = pydicom.dcmread(f)
                instance = getattr(ds, "InstanceNumber", 0)
                dicoms.append((instance, ds))
            except Exception:
                continue
        dicoms.sort(key=lambda x: x[0])
    return TE, TR, dicoms

def apply_window(img, WL, WW):
    img_w = np.clip(((img - (WL - 0.5)) / (WW - 1) + 0.5), 0, 1)
    return (img_w * 255).astype(np.uint8)

# ==================== UPLOADER ====================
st.sidebar.header("📦 Upload File ZIP DICOM")
uploaded_zips = st.sidebar.file_uploader(
    "Upload beberapa file ZIP (masing-masing berisi DICOM untuk satu TE/TR)",
    type=["zip"], accept_multiple_files=True
)

if uploaded_zips:
    for up in uploaded_zips:
        if up.name not in st.session_state["dicom_data"]:
            try:
                zip_bytes = up.read()
                TE, TR, dicoms = load_dicom_from_zip_bytes(zip_bytes, up.name)
                if dicoms:
                    st.session_state["dicom_data"][up.name] = {
                        "TE": TE, "TR": TR, "dicoms": dicoms, "n_slices": len(dicoms)
                    }
                    st.success(f"✅ File {up.name} dimuat ({len(dicoms)} slices).")
                else:
                    st.warning(f"⚠️ Tidak ditemukan file DICOM di {up.name}.")
            except Exception as e:
                st.error(f"Gagal memproses {up.name}: {e}")

# ==================== NAVIGASI ====================
tab_labels = ["1️⃣ Viewer", "2️⃣ Segmentasi ROI", "3️⃣ Curve Fitting"]
tab = st.sidebar.radio("Pilih Tab", tab_labels, index=tab_labels.index(st.session_state["current_tab"]))

# ==================== TAB 1: VIEWER ====================
if tab == "1️⃣ Viewer":
    st.session_state["current_tab"] = "1️⃣ Viewer"
    st.subheader("1️⃣ Viewer — Atur & Simpan Window Level / Width")

    if not st.session_state["dicom_data"]:
        st.info("Silakan upload minimal satu file ZIP di sidebar.")
    else:
        filenames = list(st.session_state["dicom_data"].keys())
        selected_file = st.selectbox("Pilih dataset ZIP:", filenames)

        info = st.session_state["dicom_data"][selected_file]
        dicoms = info["dicoms"]
        TE, TR = info["TE"], info["TR"]

        all_images = [ds.pixel_array.astype(np.float32) for _, ds in dicoms]
        WL_default = float(np.mean(all_images[0]))
        WW_default = float(np.max(all_images[0]) - np.min(all_images[0]))

        WL = st.slider("Window Level (WL)", float(np.min(all_images[0])), float(np.max(all_images[0])), WL_default)
        WW = st.slider("Window Width (WW)", 50.0, max(50.0, WW_default), WW_default)
        slice_index = st.slider("Slice", 0, len(all_images) - 1, len(all_images)//2) if len(all_images) > 1 else 0

        img_vis = apply_window(all_images[slice_index], WL, WW)
        st.image(img_vis, caption=f"{selected_file} | TE={TE:.3f} | TR={TR:.0f} | Slice {slice_index+1}", use_container_width=True)

        if st.button("💾 Simpan WL & WW untuk file ini"):
            st.session_state["window_params_by_file"][selected_file] = {"WL": WL, "WW": WW}
            st.success(f"✅ WL/WW disimpan untuk {selected_file} (WL={WL:.1f}, WW={WW:.1f})")

    st.markdown("---")
    if st.button("➡️ Lanjut ke Segmentasi ROI", use_container_width=True):
        st.session_state["current_tab"] = "2️⃣ Segmentasi ROI"
        st.rerun()
# ==================== TAB 2: SEGMENTASI ROI ====================
elif tab == "2️⃣ Segmentasi ROI":
    st.session_state["current_tab"] = "2️⃣ Segmentasi ROI"
    st.subheader("2️⃣ Segmentasi ROI & Hitung Intensitas")

    if not st.session_state["dicom_data"]:
        st.warning("⚠️ Belum ada data. Upload file ZIP di sidebar.")
        st.stop()

    filenames = list(st.session_state["dicom_data"].keys())
    selected_file = st.selectbox("Pilih dataset ZIP:", filenames)
    info = st.session_state["dicom_data"][selected_file]
    dicoms = info["dicoms"]

    ds_ref = dicoms[len(dicoms)//2][1]
    img_raw_ref = ds_ref.pixel_array.astype(np.float32)

    stored = st.session_state["window_params_by_file"].get(selected_file, None)
    if stored:
        WL_init, WW_init = stored["WL"], stored["WW"]
        st.info(f"📥 WL/WW dimuat dari Viewer: WL={WL_init:.1f}, WW={WW_init:.1f}")
    else:
        WL_init = float(np.mean(img_raw_ref))
        WW_init = float(np.max(img_raw_ref) - np.min(img_raw_ref))
        st.warning("⚠️ WL/WW belum disimpan dari Viewer. Gunakan default.")

    WL_opt = st.slider("Window Level (WL)", float(np.min(img_raw_ref)), float(np.max(img_raw_ref)), WL_init)
    WW_opt = st.slider("Window Width (WW)", 50.0, max(50.0, float(np.max(img_raw_ref) - np.min(img_raw_ref))), WW_init)
    img_display_ref = apply_window(img_raw_ref, WL_opt, WW_opt)

    # Layout grid
    layout_grid = [
        [None, "Air", None],
        ["Carbomer 1", "Carbomer 2", "Carbomer 3"],
        ["PEG 1", "PEG 2", "PEG 3"],
        ["Acrylamida 1", "Acrylamida 2", None],
        ["Collagen 1", "Collagen 2", "Collagen 3"],
        ["Xanthan Gum 1", "Xanthan Gum 2", "Xanthan Gum 3"],
        ["Xanthan Gum 4", "Alginate 1", "Alginate 2"],
        ["Alginate 3", "Alginate 4", "Alginate 5"]
    ]

    def generate_static_grid(image_8bit, radius_roi=11):
        img_blur = cv2.medianBlur(image_8bit, 5)
        circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=25,
                                   param1=50, param2=15, minRadius=10, maxRadius=25)
        if circles is None:
            return []
        circles = np.uint16(np.around(circles[0, :]))
        x_coords, y_coords = [c[0] for c in circles], [c[1] for c in circles]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        col_step = (max_x - min_x) / 2
        row_step = (max_y - min_y) / 7
        kolom_x = [int(min_x), int(min_x + col_step), int(max_x)]
        baris_y = [int(min_y + (i * row_step)) for i in range(8)]
        roi_list = []
        for r_idx, baris in enumerate(layout_grid):
            for c_idx, nama in enumerate(baris):
                if nama:
                    roi_list.append((kolom_x[c_idx], baris_y[r_idx], radius_roi, nama))
        return roi_list

    grid_rois = generate_static_grid(img_display_ref)
    if not grid_rois:
        st.error("❌ Gagal deteksi grid.")
        st.stop()

    img_seg_vis = cv2.cvtColor(img_display_ref, cv2.COLOR_GRAY2BGR)
    for (cx, cy, r, label) in grid_rois:
        cv2.circle(img_seg_vis, (cx, cy), r, (0, 255, 0), 1)
        cv2.putText(img_seg_vis, label.split()[0][:4], (cx-12, cy-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    st.image(img_seg_vis, caption="Grid ROI (slice tengah)", use_container_width=True)

    hasil = []
    for inst, ds in dicoms:
        arr = ds.pixel_array.astype(np.float32)
        data = {}
        for (cx, cy, r, label) in grid_rois:
            mask = np.zeros_like(arr, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            data[label] = float(np.mean(arr[mask == 255]))
        hasil.append(data)

    df = pd.DataFrame(hasil)
    df.index = [f"Slice {i+1}" for i in range(len(dicoms))]
    df.loc["Rata-rata"] = df.mean()
    st.dataframe(df.style.format("{:.2f}"))

    if st.button("💾 Simpan hasil ke Excel & Memori", use_container_width=True):
        outname = f"hasil_intensitas_{selected_file.replace('.zip','')}.xlsx"
        df.to_excel(outname)
        if "segment_results" not in st.session_state:
            st.session_state["segment_results"] = {}
        st.session_state["segment_results"][selected_file] = df
        st.success(f"Hasil disimpan & dimuat ke memori: {outname}")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Kembali ke Viewer", use_container_width=True):
            st.session_state["current_tab"] = "1️⃣ Viewer"
            st.rerun()
    with col2:
        if st.button("➡️ Lanjut ke Curve Fitting", use_container_width=True):
            st.session_state["current_tab"] = "3️⃣ Curve Fitting"
            st.rerun()


# ==================== TAB 3: CURVE FITTING ====================
elif tab == "3️⃣ Curve Fitting":
    st.session_state["current_tab"] = "3️⃣ Curve Fitting"
    st.subheader("3️⃣ Curve Fitting (Tri-eksponensial)")

    if not st.session_state.get("segment_results"):
        st.warning("⚠️ Belum ada data segmentasi tersimpan.")
        st.stop()

    results = st.session_state["segment_results"]
    te_tr_pairs = [(extract_te_tr_from_filename(fname)[0], fname) for fname in results.keys()]
    te_tr_pairs = [p for p in te_tr_pairs if p[0] is not None]
    te_tr_pairs.sort(key=lambda x: x[0])
    TE_values = np.array([t[0] for t in te_tr_pairs])

    selected_roi = st.selectbox("Pilih ROI untuk fitting:", results[te_tr_pairs[0][1]].columns)
    intensitas = np.array([results[f].loc["Rata-rata", selected_roi] for _, f in te_tr_pairs])

    # --- Model fungsi ---
    def trie_exp(te, A1, T2_1, A2, T2_2, A3, T2_3):
        return A1*np.exp(-te/T2_1)+A2*np.exp(-te/T2_2)+A3*np.exp(-te/T2_3)
    def trie_exp_offset(te, A1,T2_1,A2,T2_2,A3,T2_3,C):
        return A1*np.exp(-te/T2_1)+A2*np.exp(-te/T2_2)+A3*np.exp(-te/T2_3)+C

    # --- Normalisasi dan fitting ---
    y_raw = intensitas.astype(float)
    s0_guess = y_raw[0]
    y_norm = y_raw / s0_guess

    p0_no_offset = [0.5, 30, 0.3, 100, 0.2, 500]
    bounds_no_offset = ([0, 10, 0, 50, 0, 200], [1, 1000, 1, 1500, 1, 5000])
    p0_offset = [0.3, 15, 0.4, 80, 0.3, 300, 0.01]
    bounds_offset = ([0, 1, 0, 30, 0, 200, -0.2], [1, 30, 1, 200, 1, 1500, 0.5])

    def calc_metrics(y_true, y_pred):
        residuals = y_true - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(ss_res / len(y_true))
        return r2, rmse

    try:
        p_no, _ = curve_fit(trie_exp, TE_values, y_norm, p0=p0_no_offset,
                            bounds=bounds_no_offset, maxfev=50000)
        y_pred_no = trie_exp(TE_values, *p_no)
        r2_no, rmse_no = calc_metrics(y_norm, y_pred_no)
    except Exception as e:
        st.warning(f"Gagal fitting tanpa offset: {e}")
        p_no, y_pred_no, r2_no, rmse_no = [np.nan]*6, np.zeros_like(y_norm), np.nan, np.nan

    try:
        p_opt, _ = curve_fit(trie_exp_offset, TE_values, y_norm, p0=p0_offset,
                             bounds=bounds_offset, maxfev=50000)
        y_pred_off = trie_exp_offset(TE_values, *p_opt)
        r2_off, rmse_off = calc_metrics(y_norm, y_pred_off)
    except Exception as e:
        st.error(f"Gagal fitting dengan offset: {e}")
        st.stop()

    # --- Plotting ---
    te_plot = np.linspace(TE_values.min(), TE_values.max(), 500)
    y_plot_off = trie_exp_offset(te_plot, *p_opt) * s0_guess
    y_plot_no = trie_exp(te_plot, *p_no) * s0_guess

    plt.figure(figsize=(8,6))
    plt.scatter(TE_values, y_raw, color='black', label='Data Eksperimen (SI)')
    plt.plot(te_plot, y_plot_off, 'r-', linewidth=2, label=f'Tri-exponential Fit with Offset (R²={r2_off:.4f})')
    plt.plot(te_plot, y_plot_no, 'b--', linewidth=2, label=f'Tri-exponential Fit without Offset (R²={r2_no:.4f})')

    # Kotak info parameter
    A1, T2_1, A2, T2_2, A3, T2_3, C = p_opt
    info_text = (
        f"Result for {selected_roi} (Offset Model):\n"
        f"--------------------------\n"
        f"T2₁  : {T2_1:.2f} ms\n"
        f"T2₂  : {T2_2:.2f} ms\n"
        f"T2₃  : {T2_3:.2f} ms\n"
        f"Offset : {C*s0_guess:.2f}\n"
        f"--------------------------\n"
        f"R²   : {r2_off:.4f}\n"
        f"RMSE : {rmse_off*s0_guess:.4f}"
    )
    plt.text(0.97, 0.97, info_text, transform=plt.gca().transAxes, fontsize=10,
             va='top', ha='right', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    plt.title(f"T2 Relaxation Decay - {selected_roi}", fontsize=14)
    plt.xlabel("Echo Time (TE) [ms]")
    plt.ylabel("Signal Intensity (SI)")
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(plt)

    # --- Tabel hasil parameter ---
    A1_s, A2_s, A3_s = [p_opt[0]*s0_guess, p_opt[2]*s0_guess, p_opt[4]*s0_guess]
    Offset_s = p_opt[6]*s0_guess
    RMSE_s = rmse_off*s0_guess

    df_result = pd.DataFrame([{
        "Material": selected_roi,
        "A1": A1_s, "T2_1": T2_1,
        "A2": A2_s, "T2_2": T2_2,
        "A3": A3_s, "T2_3": T2_3,
        "Offset": Offset_s, "R²": r2_off, "RMSE": RMSE_s
    }])
    st.markdown("### 📊 Hasil Parameter Fitting (Offset Model)")
    styled_df = df_result.style.format({
        "A1": "{:.2e}", "T2_1": "{:.2f}",
        "A2": "{:.2e}", "T2_2": "{:.2f}",
        "A3": "{:.4f}", "T2_3": "{:.2f}",
        "Offset": "{:.2f}", "R²": "{:.4f}", "RMSE": "{:.4f}"
        })
    st.dataframe(styled_df, use_container_width=True)
    
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Kembali ke Segmentasi ROI", use_container_width=True):
            st.session_state["current_tab"] = "2️⃣ Segmentasi ROI"
            st.rerun()