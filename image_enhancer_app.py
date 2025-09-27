import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.color import rgb2gray
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="Image Enhancement Explorer", layout="wide")
st.title("🧠 Retinal Image Enhancement Techniques")
st.markdown("Select an enhancement method and explore the results with real-time parameter tuning.")

# --- Helper Functions ---
def display_images_with_hist(before, after, mode):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(before, use_column_width=True, channels="RGB" if mode=="Color" else "L")
        st.pyplot(plot_histogram(before, title="Original Histogram", mode=mode))
    with col2:
        st.subheader("Processed Image")
        st.image(after, use_column_width=True, channels="RGB" if mode=="Color" else "L")
        st.pyplot(plot_histogram(after, title="Processed Histogram", mode=mode))

def plot_histogram(image, title, mode):
    fig, ax = plt.subplots()
    if mode == "Color" and len(image.shape) == 3:
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=color)
    else:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(hist, color='k')
    ax.set_title(title)
    ax.set_xlim([0, 256])
    return fig

def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

# --- Image Upload ---
mode = st.sidebar.radio("Select Image Mode", ["Grayscale", "Color"])
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

# --- Method Selection ---
method = st.sidebar.selectbox(
    "Enhancement Method",
    [
        "Linear Negative",
        "Contrast Stretching",
        "Piecewise Linear Transformation",
        "Log Transformation",
        "Gamma Transformation",
        "Histogram Equalization",
        "Adaptive Histogram Equalization",
        "CLAHE"
    ]
)

# --- Main Logic ---
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    gray_image = convert_to_gray(image_np)

    processed = gray_image if mode == "Grayscale" else image_np

    if method == "Linear Negative":
        processed = 255 - processed

    elif method == "Contrast Stretching":
        p_min = st.sidebar.slider("Min Percentile", 0, 100, 2)
        p_max = st.sidebar.slider("Max Percentile", 0, 100, 98)
        if mode == "Grayscale":
            p1, p2 = np.percentile(gray_image, (p_min, p_max))
            processed = np.clip((gray_image - p1) * 255.0 / (p2 - p1), 0, 255).astype(np.uint8)
        else:
            channels = cv2.split(image_np)
            processed = cv2.merge([np.clip((ch - np.percentile(ch, p_min)) * 255.0 / (np.percentile(ch, p_max) - np.percentile(ch, p_min)), 0, 255).astype(np.uint8) for ch in channels])

    elif method == "Piecewise Linear Transformation":
        r1 = st.sidebar.slider("r1", 0, 255, 70)
        s1 = st.sidebar.slider("s1", 0, 255, 0)
        r2 = st.sidebar.slider("r2", 0, 255, 140)
        s2 = st.sidebar.slider("s2", 0, 255, 255)
        def piecewise_transform(img):
            return np.piecewise(img, 
                [img < r1, (img >= r1) & (img <= r2), img > r2], 
                [lambda x: x * s1 / r1,
                 lambda x: ((x - r1) * (s2 - s1) / (r2 - r1)) + s1,
                 lambda x: ((x - r2) * (255 - s2) / (255 - r2)) + s2]
            ).astype(np.uint8)
        if mode == "Grayscale":
            processed = piecewise_transform(gray_image)
        else:
            processed = cv2.merge([piecewise_transform(ch) for ch in cv2.split(image_np)])

    elif method == "Log Transformation":
        c = st.sidebar.slider("Constant c", 1, 50, 10)
        log_transform = lambda img: (c * np.log1p(img)).astype(np.uint8)
        if mode == "Grayscale":
            processed = log_transform(gray_image)
        else:
            processed = cv2.merge([log_transform(ch) for ch in cv2.split(image_np)])

    elif method == "Gamma Transformation":
        gamma = st.sidebar.slider("Gamma", 0.1, 5.0, 1.2)
        gamma_transform = lambda img: (255 * (img / 255) ** gamma).astype(np.uint8)
        if mode == "Grayscale":
            processed = gamma_transform(gray_image)
        else:
            processed = cv2.merge([gamma_transform(ch) for ch in cv2.split(image_np)])

    elif method == "Histogram Equalization":
        if mode == "Grayscale":
            processed = cv2.equalizeHist(gray_image)
        else:
            ycrcb = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            processed = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    elif method == "Adaptive Histogram Equalization":
        clip_limit = st.sidebar.slider("Clip Limit", 0.01, 0.1, 0.03)
        processed = exposure.equalize_adapthist(gray_image if mode == "Grayscale" else rgb2gray(image_np), clip_limit=clip_limit)
        processed = (processed * 255).astype(np.uint8)

    elif method == "CLAHE":
        clip_limit = st.sidebar.slider("Clip Limit", 1.0, 10.0, 2.0)
        tile_grid_size = st.sidebar.slider("Tile Grid Size", 1, 16, 8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        if mode == "Grayscale":
            processed = clahe.apply(gray_image)
        else:
            lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # --- Display Results ---
    display_images_with_hist(gray_image if mode == "Grayscale" else image_np, processed, mode)
else:
    st.info("📤 Please upload an image to begin.")
