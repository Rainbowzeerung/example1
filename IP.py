import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist, equalize_adapthist

st.set_page_config(page_title="Image Processing Toolkit", layout="wide")

# ----------------------------------------
# Helper Functions
# ----------------------------------------

def load_image(uploaded_file, as_gray):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if as_gray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def show_image_and_hist(title, image, is_gray):
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption=f"{title} Image", use_column_width=True, channels="GRAY" if is_gray else "RGB")
    with col2:
        fig, ax = plt.subplots()
        if is_gray:
            ax.hist(image.ravel(), bins=256, range=(0, 256), color='gray')
        else:
            for i, color in enumerate(['r', 'g', 'b']):
                ax.hist(image[:, :, i].ravel(), bins=256, range=(0, 256), color=color, alpha=0.5)
        ax.set_title(f"{title} Histogram")
        st.pyplot(fig)

def linear_negative(image):
    return 255 - image

def contrast_stretching(image, min_val, max_val):
    stretched = np.clip((image - min_val) * (255 / (max_val - min_val)), 0, 255)
    return stretched.astype(np.uint8)

def piecewise_linear(image, r1, s1, r2, s2):
    def transform_pixel(p):
        if p < r1:
            return s1 / r1 * p
        elif p < r2:
            return ((s2 - s1) / (r2 - r1)) * (p - r1) + s1
        else:
            return ((255 - s2) / (255 - r2)) * (p - r2) + s2

    vectorized_transform = np.vectorize(transform_pixel)
    return vectorized_transform(image).astype(np.uint8)

def log_transform(image, c):
    image_float = image.astype(np.float32)
    log_image = c * np.log1p(image_float)
    log_image = np.clip(log_image * 255 / np.max(log_image), 0, 255)
    return log_image.astype(np.uint8)

def gamma_transform(image, gamma):
    norm_image = image / 255.0
    gamma_corrected = np.power(norm_image, gamma)
    return (gamma_corrected * 255).astype(np.uint8)

def histogram_equalization(image):
    if len(image.shape) == 2:
        return equalize_hist(image) * 255
    else:
        img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

def adaptive_histogram_equalization(image, clip_limit):
    if len(image.shape) == 2:
        return equalize_adapthist(image, clip_limit=clip_limit) * 255
    else:
        img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = equalize_adapthist(img_yuv[:, :, 0] / 255.0, clip_limit=clip_limit) * 255
        return cv2.cvtColor(img_yuv.astype(np.uint8), cv2.COLOR_YUV2RGB)

def clahe_transform(image, clip_limit, tile_grid_size):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    if len(image.shape) == 2:
        return clahe.apply(image)
    else:
        img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

# ----------------------------------------
# Streamlit UI
# ----------------------------------------

st.title("🖼️ Image Processing Toolkit")

uploaded_file = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])
mode = st.radio("Select Image Type", ["Grayscale", "Color"], horizontal=True)

if uploaded_file:
    is_gray = (mode == "Grayscale")
    image = load_image(uploaded_file, is_gray)
    processed_image = None

    st.subheader("⚙️ Select Processing Method")
    method = st.selectbox("Choose a transformation method:", [
        "Linear Negative",
        "Contrast Stretching",
        "Piecewise Linear Transformation",
        "Log Transformation",
        "Gamma Transformation",
        "Histogram Equalization",
        "Adaptive Histogram Equalization (AHE)",
        "CLAHE (Contrast Limited AHE)"
    ])

    # Parameter controls per method
    if method == "Linear Negative":
        processed_image = linear_negative(image)

    elif method == "Contrast Stretching":
        min_val = st.slider("Min Value", 0, 255, 50)
        max_val = st.slider("Max Value", 0, 255, 200)
        processed_image = contrast_stretching(image, min_val, max_val)

    elif method == "Piecewise Linear Transformation":
        r1 = st.slider("r1", 0, 255, 70)
        s1 = st.slider("s1", 0, 255, 0)
        r2 = st.slider("r2", 0, 255, 140)
        s2 = st.slider("s2", 0, 255, 255)
        processed_image = piecewise_linear(image, r1, s1, r2, s2)

    elif method == "Log Transformation":
        c = st.slider("Scaling Constant (c)", 1.0, 50.0, 10.0)
        processed_image = log_transform(image, c)

    elif method == "Gamma Transformation":
        gamma = st.slider("Gamma", 0.1, 5.0, 1.0)
        processed_image = gamma_transform(image, gamma)

    elif method == "Histogram Equalization":
        processed_image = histogram_equalization(image)

    elif method == "Adaptive Histogram Equalization (AHE)":
        clip_limit = st.slider("Clip Limit", 0.01, 0.3, 0.03)
        processed_image = adaptive_histogram_equalization(image, clip_limit)

    elif method == "CLAHE (Contrast Limited AHE)":
        clip_limit = st.slider("CLAHE Clip Limit", 1.0, 40.0, 2.0)
        tile_grid_size = st.slider("Tile Grid Size", 1, 16, 8)
        processed_image = clahe_transform(image, clip_limit, tile_grid_size)

    # Show results
    st.markdown("### 📊 Results Comparison")
    show_image_and_hist("Original", image, is_gray)
    show_image_and_hist("Processed", processed_image, is_gray)
