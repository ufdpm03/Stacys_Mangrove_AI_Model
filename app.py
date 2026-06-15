import subprocess
from pathlib import Path
import shutil

import pandas as pd
import streamlit as st
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

APP_TITLE = "Stacy's Mangrove AI Model"
BASE_DIR = Path(__file__).parent.resolve()

st.set_page_config(page_title=APP_TITLE, page_icon="🌱", layout="wide")

st.title("🌱 Stacy's Mangrove AI Model")
st.caption("Drone-Based Mangrove Detection using the validated U-Net workflow")

uploaded = st.file_uploader(
    "Upload drone orthomosaic",
    type=["tif", "tiff", "png", "jpg", "jpeg"]
)

if uploaded:
    input_path = BASE_DIR / "input_orthomosaic.tif"
    with open(input_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Uploaded {uploaded.name}")

scripts = [
    "make_tiles.py",
    "predict.py",
    "stitch_tiles.py",
    "calculate_percent.py",
]

def clean_previous():
    for folder in ["tiles", "predicted_tiles"]:
        p = BASE_DIR / folder
        if p.exists():
            shutil.rmtree(p)

    old_outputs = [
        "mangrove_overlay_full.png",
        "whole_mangrove_overlay.png",
        "whole_mangrove_prediction.png",
        "prediction.png",
        "prediction_overlay.png",
        "mangrove_results.csv",
    ]

    for f in old_outputs:
        p = BASE_DIR / f
        if p.exists():
            p.unlink()

def run_script(name):
    result = subprocess.run(
        ["python", name],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True
    )
    return result.returncode == 0, result.stdout + result.stderr

def preview(path, max_width=1800):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w > max_width:
        nh = int(h * (max_width / w))
        img = img.resize((max_width, nh))
    return img

if st.button("🌱 Run Mangrove Analysis", type="primary"):

    if not (BASE_DIR / "input_orthomosaic.tif").exists():
        st.error("Upload an orthomosaic first.")
        st.stop()

    clean_previous()

    log = st.empty()
    text = ""

    for script in scripts:
        st.write(f"Running {script}")
        ok, output = run_script(script)

        text += f"\n--- {script} ---\n"
        text += output

        log.text_area("Processing log", text, height=300)

        if not ok:
            st.error(f"{script} failed.")
            st.stop()

    st.success("Analysis complete.")

st.divider()

# Coverage metrics
results_path = BASE_DIR / "mangrove_results.csv"

if results_path.exists():
    st.subheader("Coverage Results")
    df = pd.read_csv(results_path)

    if not df.empty:
        row = df.iloc[0]

        c1, c2, c3 = st.columns(3)

        if "mangrove_pixels" in df.columns:
            c1.metric("Mangrove pixels", f"{int(row['mangrove_pixels']):,}")

        if "total_valid_pixels" in df.columns:
            c2.metric("Total valid pixels", f"{int(row['total_valid_pixels']):,}")

        if "mangrove_cover_percent" in df.columns:
            c3.metric("Mangrove cover", f"{float(row['mangrove_cover_percent']):.3f}%")

    st.dataframe(df, use_container_width=True)

    with open(results_path, "rb") as f:
        st.download_button(
            "Download results CSV",
            f,
            file_name="mangrove_results.csv",
            mime="text/csv"
        )
else:
    st.info("Run analysis to create mangrove_results.csv and display percent cover.")

st.divider()

cols = st.columns(2)

overlay_candidates = [
    BASE_DIR / "mangrove_overlay_full.png",
    BASE_DIR / "whole_mangrove_overlay.png",
    BASE_DIR / "prediction_overlay.png",
]

mask_candidates = [
    BASE_DIR / "mangrove_mask_full.png",
]

overlay = next((p for p in overlay_candidates if p.exists()), None)
mask = next((p for p in mask_candidates if p.exists()), None)

with cols[0]:
    st.subheader("Mangrove Overlay")

    if overlay:
        st.image(preview(overlay), use_container_width=True)
        with open(overlay, "rb") as f:
            st.download_button(
                "Download Overlay",
                f,
                file_name=overlay.name,
                mime="image/png"
            )
    else:
        st.warning("No overlay image found yet.")

with cols[1]:
    st.subheader("Prediction Mask")

    if mask:
        st.image(preview(mask), use_container_width=True)
        with open(mask, "rb") as f:
            st.download_button(
                "Download Mask",
                f,
                file_name=mask.name,
                mime="image/png"
            )
    else:
        st.info(
            "No separate prediction mask image was created by the current workflow. "
            "The overlay is still valid. If you want, we can update stitch_tiles.py "
            "to also build a full-size mask."
        )
