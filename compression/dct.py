import numpy as np
import cv2

def get_quantization_matrix(quality):
    # Basic JPEG quantization matrix scaled by quality
    Q50 = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ], dtype=np.float32)

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - quality * 2

    q_matrix = np.floor((Q50 * scale + 50) / 100)
    q_matrix[q_matrix == 0] = 1
    return q_matrix

# ----------------------
# Compress using DCT + Quantization
# ----------------------
def apply_block_dct(image_array, block_size=8, quality=50):
    h, w = image_array.shape
    h_pad = h + (block_size - h % block_size) % block_size
    w_pad = w + (block_size - w % block_size) % block_size
    padded = np.zeros((h_pad, w_pad), dtype=np.float32)
    padded[:h, :w] = image_array.astype(np.float32)
    
    padded -= 128.0  # center pixel values
    q_matrix = get_quantization_matrix(quality)

    dct_blocks = np.zeros_like(padded, dtype=np.float32)
    for i in range(0, h_pad, block_size):
        for j in range(0, w_pad, block_size):
            block = padded[i:i+block_size, j:j+block_size]
            dct_block = cv2.dct(block)
            quantized = np.round(dct_block / q_matrix)
            dct_blocks[i:i+block_size, j:j+block_size] = quantized

    return dct_blocks, (h, w)

# ----------------------
# Decompress DCT blocks + Dequantization + IDCT
# ----------------------
def inverse_block_dct(dct_blocks, original_shape, block_size=8, quality=50):
    h_pad, w_pad = dct_blocks.shape
    q_matrix = get_quantization_matrix(quality)
    img = np.zeros_like(dct_blocks, dtype=np.float32)

    for i in range(0, h_pad, block_size):
        for j in range(0, w_pad, block_size):
            block = dct_blocks[i:i+block_size, j:j+block_size]
            dequant = block * q_matrix
            idct_block = cv2.idct(dequant)
            img[i:i+block_size, j:j+block_size] = idct_block

    img = img[:original_shape[0], :original_shape[1]]
    img += 128.0  # shift back
    return np.clip(img, 0, 255).astype(np.uint8)
