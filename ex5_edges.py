# Experiment 5: Edge detection with Sobel and Prewitt
# Name: Soham   Roll No: 23BIT276D

import cv2
import numpy as np

from common import load_image, show_grid

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)


def filter3x3(image, kernel):
    """Correlate a 3x3 kernel with the image (same as cv2.filter2D)."""
    padded = np.pad(image.astype(np.float64), 1, mode="edge")
    h, w = image.shape
    out = np.zeros((h, w))
    for dy in range(3):
        for dx in range(3):
            out += kernel[dy, dx] * padded[dy:dy + h, dx:dx + w]
    return out


def gradient(image, kx):
    """Y kernel is the transpose of the X kernel."""
    gx = filter3x3(image, kx)
    gy = filter3x3(image, kx.T)
    magnitude = np.hypot(gx, gy)
    direction = np.degrees(np.arctan2(gy, gx))
    return gx, gy, magnitude, direction


def to_u8(a):
    a = np.abs(a)
    return (a / a.max() * 255).astype(np.uint8)


image = load_image(gray=True)
# Light blur first so noise doesn't show up as edges
smooth = cv2.GaussianBlur(image, (3, 3), 0)

gx, gy, mag, direction = gradient(smooth, SOBEL_X)
_, _, prewitt_mag, _ = gradient(smooth, PREWITT_X)

# Check against OpenCV's Sobel
cv_mag = np.hypot(cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE),
                  cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE))
print("Max difference ours vs cv2.Sobel:", np.abs(mag - cv_mag).max())

# Keep only strong edges: top 15% of gradient magnitudes
thresh = np.percentile(mag, 85)
strong = np.where(mag > thresh, 255, 0).astype(np.uint8)
print(f"Edge threshold (85th percentile): {thresh:.1f}")

direction_u8 = ((direction + 180) / 360 * 255).astype(np.uint8)

show_grid(
    [image, to_u8(gx), to_u8(gy), to_u8(mag),
     to_u8(prewitt_mag), strong, direction_u8, cv2.Canny(smooth, 60, 150)],
    ["Original", "Sobel Gx", "Sobel Gy", "Sobel magnitude",
     "Prewitt magnitude", "Strong edges (top 15%)", "Gradient direction",
     "Canny (for comparison)"],
    cols=4, out_name="ex5_edges.png",
)
