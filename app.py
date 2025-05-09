import streamlit as st
import numpy as np
from PIL import Image
import io
import cv2
from enhancer import enhance_image_pipeline

st.set_page_config(page_title="🪄 Product Image Enhancer", layout="centered")
st.title("🪄 Product Image Enhancer")
st.markdown("Enhance product shots by improving contrast, lighting, background, and realism.")

st.info("Enhancement may take up to a few minutes depending on image size.")

uploaded_file = st.file_uploader("Upload your product image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("📸 Original Image")
    st.image(image, use_container_width=True)

    status_text = st.empty()
    status_placeholder = status_text.markdown("Processing...")

    def streamlit_callback(step):
        status_placeholder.markdown(f"**🔄 {step}...**")

    with st.spinner("Enhancing image. Please wait..."):
        try:
            enhanced_np = enhance_image_pipeline(image_np, step_callback=streamlit_callback)
            enhanced_np = enhanced_np.astype(np.uint8)
            enhanced_pil = Image.fromarray(enhanced_np)

            st.subheader("✨ Enhanced Image")
            st.image(enhanced_pil, use_container_width=True)

            buffer = io.BytesIO()
            enhanced_pil.save(buffer, format="PNG")
            st.download_button("⬇️ Download Enhanced Image", buffer.getvalue(), file_name="enhanced.png", mime="image/png")
        except Exception as e:
            st.error(f"An error occurred during enhancement: {str(e)}")
