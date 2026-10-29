# Experiment 4: Gaussian noise and Gaussian smoothing
# Name: Soham   Roll No: 23BIT276D

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import load_image, save_figure, show_grid


def add_gaussian_noise(image, sigma=40, seed=11):
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float64) + rng.normal(0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def gaussian_kernel_2d(size, sigma):
    """Build the full 2D kernel straight from the Gaussian formula."""
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def convolve2d(image, kernel):
    """Plain 2D convolution: multiply each neighbourhood by the kernel and sum.
    Loops over kernel offsets (k*k steps) instead of over every pixel."""
    k = kernel.shape[0]
    pad = k // 2
    padded = np.pad(image.astype(np.float64), pad, mode="reflect")
    h, w = image.shape
    out = np.zeros((h, w))
    for dy in range(k):
        for dx in range(k):
            out += kernel[dy, dx] * padded[dy:dy + h, dx:dx + w]
    return np.clip(out, 0, 255).astype(np.uint8)


image = load_image(gray=True)
noisy = add_gaussian_noise(image, sigma=40)

K, SIGMA = 9, 2.0
kernel = gaussian_kernel_2d(K, SIGMA)
print(f"Kernel {K}x{K}, sigma={SIGMA}, sum={kernel.sum():.4f}")

ours = convolve2d(noisy, kernel)
opencv = cv2.GaussianBlur(noisy, (K, K), SIGMA, borderType=cv2.BORDER_REFLECT_101)

diff = np.abs(ours.astype(int) - opencv.astype(int))
print("Mean abs difference (ours vs OpenCV):", diff.mean())
print("Max abs difference:", diff.max())

show_grid(
    [image, noisy, opencv, ours],
    ["Original", "Gaussian noise (sigma=40)", "OpenCV GaussianBlur", "Our convolution"],
    cols=4, out_name="ex4_gaussian.png",
)

# What the kernel looks like
fig = plt.figure(figsize=(5, 4))
plt.imshow(kernel, cmap="viridis")
plt.colorbar()
plt.title(f"Gaussian kernel {K}x{K}, sigma={SIGMA}")
save_figure(fig, "ex4_kernel.png")

# Effect of sigma
sigmas = [0.8, 2.0, 4.0]
show_grid(
    [convolve2d(noisy, gaussian_kernel_2d(K, s)) for s in sigmas],
    [f"sigma = {s}" for s in sigmas],
    cols=3, out_name="ex4_sigmas.png",
)
