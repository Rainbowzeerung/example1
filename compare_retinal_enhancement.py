import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure, img_as_float
from skimage.filters import gaussian
from PIL import Image

st.set_page_config(layout="wide")
st.title("🔬 Retinal Image Enhancement: Basic vs Decomposition-Based Method")
st.markdown("This app compares traditional enhancement techniques with a method inspired by image decomposition and visual adaptation.")

def plot_histogram(image, title="Histogram"):
    fig, ax = plt.subplots()
    if len(image.shape) == 3:
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=color)
    else:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(hist, color='k')
    ax.set_title(title)
    ax.set_xlim([0, 256])
    return fig

def basic_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    merged = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def paper_like_method(image):
    # Simulate decomposition into base + detail
    base = gaussian(image, sigma=10, preserve_range=True, channel_axis=-1)
    detail = image.astype(np.float32) - base.astype(np.float32)
    
    # Visual adaptation: enhance base contrast
    base_norm = exposure.equalize_adapthist(img_as_float(base), clip_limit=0.03)
    base_norm = (base_norm * 255).astype(np.uint8)
    
    # Enhance detail
    detail_enhanced = np.clip(detail * 1.5, -255, 255).astype(np.int16)
    
    # Combine base + enhanced detail
    enhanced = np.clip(base_norm.astype(np.int16) + detail_enhanced, 0, 255).astype(np.uint8)
    return enhanced

uploaded_file = st.sidebar.file_uploader("Upload a Retinal Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Apply both methods
    enhanced_basic = basic_clahe(image_np)
    enhanced_paper = paper_like_method(image_np)

    st.subheader("📊 Comparison: CLAHE vs Decomposition + Adaptation")
    col1, col2 = st.columns(2)

    with col1:
        st.image(enhanced_basic, caption="Basic Enhancement (CLAHE)", use_column_width=True)
        st.pyplot(plot_histogram(enhanced_basic, title="Histogram: CLAHE"))

    with col2:
        st.image(enhanced_paper, caption="Paper-Inspired Method", use_column_width=True)
        st.pyplot(plot_histogram(enhanced_paper, title="Histogram: Paper Method"))

    st.markdown("### 🖼️ Original Image")
    st.image(image_np, caption="Original Fundus Image", use_column_width=True)
    st.pyplot(plot_histogram(image_np, title="Histogram: Original"))
else:
    st.info("📤 Please upload a fundus image to begin.")
