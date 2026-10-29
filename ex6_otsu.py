# Experiment 6: Otsu thresholding (own implementation)
# Name: Soham   Roll No: 23BIT276D

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import load_image, save_figure


def otsu_threshold(image):
    """Try every threshold and keep the one with the largest
    between-class variance, using cumulative sums (no inner loop)."""
    hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    levels = np.arange(256)

    w0 = np.cumsum(p)                 # weight of class 0 (<= t)
    w1 = 1 - w0                       # weight of class 1 (> t)
    mu_cum = np.cumsum(p * levels)
    mu_total = mu_cum[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = mu_cum / w0
        mu1 = (mu_total - mu_cum) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
    between = np.nan_to_num(between)

    return int(np.argmax(between)), hist, between


image = load_image(gray=True)

t, hist, between = otsu_threshold(image)
binary = np.where(image > t, 255, 0).astype(np.uint8)

t_cv, binary_cv = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Our Otsu threshold   :", t)
print("OpenCV Otsu threshold:", int(t_cv))
print("Foreground pixels: {:.1f}%".format(np.mean(binary == 255) * 100))

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

axes[0, 0].imshow(image, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 1].imshow(binary, cmap="gray")
axes[0, 1].set_title(f"Our Otsu (t = {t})")
for ax in axes[0]:
    ax.axis("off")

axes[1, 0].bar(np.arange(256), hist, width=1, color="gray")
axes[1, 0].axvline(t, color="red", linestyle="--", label=f"t = {t}")
axes[1, 0].set_title("Histogram")
axes[1, 0].legend()

axes[1, 1].plot(between, color="teal")
axes[1, 1].axvline(t, color="red", linestyle="--")
axes[1, 1].set_title("Between-class variance vs threshold")
axes[1, 1].set_xlabel("threshold")

fig.tight_layout()
save_figure(fig, "ex6_otsu.png")
