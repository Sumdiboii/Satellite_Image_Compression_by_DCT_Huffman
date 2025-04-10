import numpy as np
import cv2

def dct_compress(img_array, block_size=8, threshold=10):
    h, w = img_array.shape
    h_pad = h + (block_size - h % block_size) % block_size
    w_pad = w + (block_size - w % block_size) % block_size
    padded = np.zeros((h_pad, w_pad), dtype=np.float32)
    padded[:h, :w] = img_array.astype(np.float32)
    padded /= 255.0

    dct_blocks = np.zeros_like(padded)
    non_zero = 0

    for i in range(0, h_pad, block_size):
        for j in range(0, w_pad, block_size):
            block = padded[i:i+block_size, j:j+block_size]
            block_dct = cv2.dct(block)
            block_dct[np.abs(block_dct) < threshold] = 0
            non_zero += np.count_nonzero(block_dct)
            dct_blocks[i:i+block_size, j:j+block_size] = block_dct

    size = non_zero * 2  # 2 bytes per non-zero value
    return {"dct": dct_blocks, "shape": (h, w), "size": size}

def dct_decompress(compressed, block_size=8):
    dct_blocks = compressed["dct"]
    h, w = compressed["shape"]
    h_pad, w_pad = dct_blocks.shape

    img = np.zeros_like(dct_blocks)
    for i in range(0, h_pad, block_size):
        for j in range(0, w_pad, block_size):
            block = dct_blocks[i:i+block_size, j:j+block_size]
            img[i:i+block_size, j:j+block_size] = cv2.idct(block)

    img = img[:h, :w]
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)

def apply_block_dct(image_array, block_size=8):
    h, w = image_array.shape
    h_pad = h + (block_size - h % block_size) % block_size
    w_pad = w + (block_size - w % block_size) % block_size
    padded = np.zeros((h_pad, w_pad), dtype=np.float32)
    padded[:h, :w] = image_array.astype(np.float32)
    padded /= 255.0

    blocks = np.zeros_like(padded)
    for i in range(0, h_pad, block_size):
        for j in range(0, w_pad, block_size):
            block = padded[i:i+block_size, j:j+block_size]
            blocks[i:i+block_size, j:j+block_size] = cv2.dct(block)

    return blocks, (h, w)

def inverse_block_dct(dct_blocks, original_shape, block_size=8):
    

    h_pad, w_pad = dct_blocks.shape
    img = np.zeros_like(dct_blocks)
    for i in range(0, h_pad, block_size):
        for j in range(0, w_pad, block_size):
            block = dct_blocks[i:i+block_size, j:j+block_size]
            img[i:i+block_size, j:j+block_size] = cv2.idct(block)

    img = img[:original_shape[0], :original_shape[1]]
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)
