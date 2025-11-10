import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------------------------------------------------------
# APP HEADER
# -------------------------------------------------------------------
st.set_page_config(page_title="Heart Blockage Detection", layout="wide")

st.title("🫀 Heart Blockage Detection using VNet + AI Agent")
st.write("Upload a heart or angiogram image to detect potential blockages and view AI-driven severity analysis.")

st.sidebar.title("⚙️ App Configuration")
st.sidebar.write("**Model:** VNet (50 Epochs)")
st.sidebar.write("**Validation Accuracy:** ~99.2%")
st.sidebar.write("**Developer:** **Chaytanya Krishna**")

# -------------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------------
model = load_model("vnet_best_model.h5", compile=False)

# -------------------------------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload a Heart Scan Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.success("✅ Image loaded successfully. Click **Predict Blockage** below to start AI analysis.")

    # -------------------------------------------------------------------
    # RUN PREDICTION BUTTON
    # -------------------------------------------------------------------
    if st.button("🔍 Predict Blockage"):

        # Preprocessing
        input_img = cv2.resize(img_array, (256, 256))
        input_img = np.expand_dims(input_img, axis=0) / 255.0

        # Predict mask
        pred_mask = model.predict(input_img)[0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        pred_mask = cv2.resize(pred_mask, (img_array.shape[1], img_array.shape[0]))

        # Mask visualization
        mask_vis = np.zeros_like(img_array)
        mask_vis[pred_mask == 255] = [255, 255, 255]

        # Overlay
        overlay = img_array.copy()
        overlay[pred_mask == 255] = [255, 0, 0]
        blended = cv2.addWeighted(img_array, 0.7, overlay, 0.3, 0)

        # Blockage %
        blockage_ratio = float(np.sum(pred_mask == 255) / pred_mask.size * 100)

        # Save variables (critical fix ✅)
        st.session_state["pred_mask"] = pred_mask
        st.session_state["mask_vis"] = mask_vis
        st.session_state["blended"] = blended
        st.session_state["img_array"] = img_array
        st.session_state["blockage_ratio"] = blockage_ratio
        st.session_state["predicted"] = True

# -------------------------------------------------------------------
# SHOW RESULTS AFTER PREDICTION
# -------------------------------------------------------------------
if st.session_state.get("predicted", False):

    img_array = st.session_state["img_array"]
    blended = st.session_state["blended"]
    mask_vis = st.session_state["mask_vis"]
    pred_mask = st.session_state["pred_mask"]
    blockage_ratio = st.session_state["blockage_ratio"]

    # ✅ Status Bar
    if blockage_ratio < 5:
        st.success(f"✅ No Significant Blockage Detected — Confidence: {blockage_ratio:.2f}%")
    else:
        st.error(f"⚠️ Potential Blockage Detected — Severity: {blockage_ratio:.2f}%")

    st.subheader("📊 Visualization Results")

    # 3-column display
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_array, caption="Original Image", use_column_width=True)

    with col2:
        st.image(mask_vis, caption="Predicted Mask", use_column_width=True)

    with col3:
        st.image(blended, caption="Overlay Visualization", use_column_width=True)

    st.markdown("---")

    # ✅ Download button for predicted image
    st.download_button(
        "📥 Download Predicted Image",
        cv2.imencode(".png", blended)[1].tobytes(),
        file_name="predicted_blockage.png",
        mime="image/png"
    )

    # -------------------------------------------------------------------
    # ✅ AI SUMMARY BUTTON
    # -------------------------------------------------------------------
    if st.button("🧠 Generate AI Summary"):

        st.subheader("🧠 AI Summary")

        if blockage_ratio < 5:
            st.info(
                f"✅ **No significant coronary blockage detected.**\n\n"
                f"- Estimated blockage area: **{blockage_ratio:.2f}%**\n"
                "- The scan appears healthy with minimal obstruction.\n"
                "- No immediate medical intervention required.\n"
                "- Routine monitoring is recommended."
            )
        else:
            st.warning(
                f"⚠️ **Potential coronary blockage identified.**\n\n"
                f"- Estimated blockage area: **{blockage_ratio:.2f}%**\n"
                "- Further medical evaluation is strongly recommended.\n"
                "- Consider contacting a cardiologist for follow-up tests."
            )
