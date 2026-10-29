# Experiment 3: Mean vs median filtering (own implementation)
# Name: Soham   Roll No: 23BIT276D

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from common import load_image, show_grid

rng = np.random.default_rng(3)


def salt_and_pepper(image, amount=0.3):
    noisy = image.copy()
    r = rng.random(image.shape)
    noisy[r < amount / 2] = 0
    noisy[r > 1 - amount / 2] = 255
    return noisy


def windows(image, k, stride):
    """All k x k neighbourhoods, taken every `stride` pixels.
    Uses a strided view instead of two nested Python loops."""
    pad = k // 2
    padded = np.pad(image, pad, mode="edge")
    view = sliding_window_view(padded, (k, k))
    return view[::stride, ::stride]


def mean_filter(image, k=3, stride=1):
    return windows(image, k, stride).mean(axis=(-2, -1)).astype(np.uint8)


def median_filter(image, k=3, stride=1):
    return np.median(windows(image, k, stride), axis=(-2, -1)).astype(np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


image = load_image(gray=True)
noisy = salt_and_pepper(image, 0.3)

# 1) Mean vs median at 5x5
mean5 = mean_filter(noisy, 5)
median5 = median_filter(noisy, 5)
print(f"PSNR noisy : {psnr(image, noisy):.2f} dB")
print(f"PSNR mean  : {psnr(image, mean5):.2f} dB")
print(f"PSNR median: {psnr(image, median5):.2f} dB")

show_grid(
    [image, noisy, mean5, median5],
    ["Original", "Noisy (30%)",
     f"Mean 5x5 ({psnr(image, mean5):.1f} dB)",
     f"Median 5x5 ({psnr(image, median5):.1f} dB)"],
    cols=4, out_name="ex3_mean_vs_median.png",
)

# 2) Effect of kernel size on the median filter
sizes = [3, 5, 9, 13]
results = [median_filter(noisy, k) for k in sizes]
show_grid(
    results,
    [f"Median {k}x{k} ({psnr(image, r):.1f} dB)" for k, r in zip(sizes, results)],
    cols=4, out_name="ex3_kernel_sizes.png",
)

# 3) Effect of stride on output size
strides = [1, 3, 5]
results = [median_filter(noisy, 5, s) for s in strides]
for s, r in zip(strides, results):
    print(f"Stride {s}: output shape {r.shape}")
show_grid(
    results,
    [f"Stride {s} -> {r.shape}" for s, r in zip(strides, results)],
    cols=3, out_name="ex3_strides.png",
)
