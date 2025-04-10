import streamlit as st
import numpy as np
import time
from PIL import Image
import cv2
import base64
import os

from compression.dct import dct_compress, apply_block_dct, inverse_block_dct
from compression.huffman import compress_huffman
from utils.metrics import calculate_mse_psnr

st.set_page_config(page_title="Satellite Image Compressor", layout="centered", page_icon="🚀")

# Function to load and encode image
def get_base64_image(image_path):
    if not os.path.exists(image_path):
        st.warning(f"⚠️ Image not found: {image_path}")
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load sidebar images
bg_img_left = get_base64_image("assets/stars1.jpg")
bg_img_right = get_base64_image("assets/stars2.jpg")

# Custom CSS
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&family=Ubuntu+Mono&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Ubuntu Mono', monospace;
            background-color: #0d1117;
            color: white;
        }}

        h1, h2, h3, h4 {{
            font-family: 'Orbitron', sans-serif;
            color: #58a6ff;
            text-shadow: 0 0 8px rgba(88,166,255,0.6);
        }}

        .stButton > button {{
            background-color: #1f6feb;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
            box-shadow: 0 4px 14px rgba(31, 111, 235, 0.4);
            transition: all 0.3s ease;
        }}

        .stButton > button:hover {{
            background-color: #3c8aff;
            box-shadow: 0 0 20px rgba(60,138,255,0.8);
            transform: scale(1.05);
        }}

        img {{
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            position: relative;
            z-index: 2;
        }}

        .vertical-bar {{
            position: fixed;
            top: 0;
            width: 400px;
            height: 100vh;
            z-index: 1;
            background-size: cover;
            background-position: center;
        }}

        .left-bar {{
            left: 0;
            background-image: url("data:image/jpeg;base64,{bg_img_left}");
        }}

        .right-bar {{
            right: 0;
            background-image: url("data:image/jpeg;base64,{bg_img_right}");
        }}

        hr {{
            border: none;
            height: 1px;
            background: linear-gradient(to right, #58a6ff, transparent);
            margin: 2rem 0;
        }}

        .footer {{
            text-align: center;
            font-size: 0.9rem;
            color: #8b949e;
        }}
    </style>
    <div class="vertical-bar left-bar"></div>
    <div class="vertical-bar right-bar"></div>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1 style='text-align: center; color: #58a6ff; text-shadow: 0 0 20px #58a6ff;'>🚀 Satellite Image Compressor</h1>", unsafe_allow_html=True)

# Upload and Process
compression_method = st.selectbox("Choose Compression Method:", ["Huffman Only", "DCT Only", "DCT + Huffman"])
uploaded_file = st.file_uploader("Upload a grayscale satellite image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    img_array = np.array(img)

    start_time = time.time()

    if compression_method == "Huffman Only":
        encoded, codes = compress_huffman(img_array)
        compressed_size_bits = len(encoded)
        compressed_size = compressed_size_bits // 8 + (1 if compressed_size_bits % 8 != 0 else 0)
        decompressed_array = img_array
        compressed_display = np.full_like(img_array, 127)

    elif compression_method == "DCT Only":
        compressed = dct_compress(img_array)
        compressed_data = compressed["dct"].tobytes()
        compressed_size = len(compressed_data)
        decompressed_array = inverse_block_dct(compressed["dct"], img_array.shape)
        dct_vis = np.log(np.abs(compressed["dct"]) + 1e-5)
        dct_vis = (dct_vis / np.max(dct_vis) * 255).astype(np.uint8)
        compressed_display = dct_vis

    else:  # DCT + Huffman
        dct_blocks, shape = apply_block_dct(img_array)
        quantized = np.round(dct_blocks / 10).astype(np.int16)
        encoded, codes = compress_huffman(quantized)
        compressed_size_bits = len(encoded)
        compressed_size = compressed_size_bits // 8 + (1 if compressed_size_bits % 8 != 0 else 0)

        # Decompression simulation
        dequantized = (quantized * 10).astype(np.float32)
        decompressed_array = inverse_block_dct(dequantized, shape)
        compressed_display = np.full_like(img_array, 90)

    time_taken = time.time() - start_time

    st.subheader("🖼️ Image Comparison")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(Image.fromarray(img_array), caption="Original Image", use_column_width=True, channels="GRAY")
    with col2:
        st.image(Image.fromarray(compressed_display), caption="Compressed View", use_column_width=True, channels="GRAY")
    with col3:
        st.image(Image.fromarray(decompressed_array.astype(np.uint8)), caption="Decompressed Image", use_column_width=True, channels="GRAY")

    st.subheader("📊 Compression Statistics")
    original_size = img_array.size  # in bytes
    compression_ratio = round(original_size / compressed_size, 2) if compressed_size else 0
    mse, psnr = calculate_mse_psnr(img_array, decompressed_array.astype(np.uint8))

    st.markdown(f"**Original Size:** {original_size} bytes")
    st.markdown(f"**Compressed Size:** {compressed_size} bytes")
    st.markdown(f"**Compression Ratio:** {compression_ratio}")
    st.markdown(f"**MSE:** {round(mse, 2)}")
    st.markdown(f"**PSNR:** {round(psnr, 2)} dB")
    st.markdown(f"**Time Taken:** {round(time_taken, 3)} seconds")

    st.markdown("<div class='footer'>✨ Made with ❤️ and Python by Sumedh ✨</div>", unsafe_allow_html=True)
