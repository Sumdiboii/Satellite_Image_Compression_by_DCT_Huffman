import numpy as np
import math

def calculate_mse_psnr(original, decompressed):
    mse = np.mean((original - decompressed) ** 2)
    if mse == 0:
        return 0, float('inf')
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    return mse, psnr
