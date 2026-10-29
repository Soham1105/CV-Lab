# Experiment 1: Image basics with OpenCV
# Name: Soham   Roll No: 23BIT276D

import cv2
import numpy as np

from common import load_image, show_grid

print("OpenCV :", cv2.__version__)
print("NumPy  :", np.__version__)

img = load_image()
h, w, c = img.shape
print(f"Shape: {img.shape}  (height={h}, width={w}, channels={c})")
print("Dtype:", img.dtype)
print("Total pixels:", h * w)

# Pixel access: read the centre pixel, then paint a small green square there
cy, cx = h // 2, w // 2
print(f"Pixel at centre ({cy},{cx}) BGR =", img[cy, cx])
pixel_edit = img.copy()
pixel_edit[cy - 5:cy + 5, cx - 5:cx + 5] = (0, 255, 0)

# Channel split: show only the red channel
b, g, r = cv2.split(img)

# ROI: centre crop (middle half of the image)
roi = img[h // 4:3 * h // 4, w // 4:3 * w // 4]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Resize to half size, keeping aspect ratio
half = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
print("Half-size shape:", half.shape)

flip_both = cv2.flip(img, -1)
rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# Rotate by an arbitrary angle around the centre
M = cv2.getRotationMatrix2D((cx, cy), 30, 1.0)
rot30 = cv2.warpAffine(img, M, (w, h))

# Drawing: circle, line, rectangle and text on one copy
drawn = img.copy()
cv2.circle(drawn, (cx, cy), min(h, w) // 5, (0, 0, 255), 3)
cv2.line(drawn, (0, 0), (w - 1, h - 1), (255, 255, 0), 2)
cv2.rectangle(drawn, (10, 10), (w // 3, h // 3), (0, 200, 0), 2)
cv2.putText(drawn, "CV Lab - Ex1", (15, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)

# Brightness / contrast: out = alpha * img + beta
bright = cv2.convertScaleAbs(img, alpha=1.3, beta=25)

show_grid(
    [img, pixel_edit, r, roi, gray, hsv[:, :, 0], half, flip_both,
     rot90, rot30, drawn, bright],
    ["Original", "Pixel edit (centre)", "Red channel", "Centre ROI",
     "Grayscale", "HSV hue", "Half size", "Flip both axes",
     "Rotate 90 CW", "Rotate 30 deg", "Shapes + text", "Brighter (a=1.3, b=25)"],
    cols=4, out_name="ex1_basics.png",
)
