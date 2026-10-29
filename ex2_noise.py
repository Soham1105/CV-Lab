# Experiment 2: Adding salt & pepper noise
# Name: Soham   Roll No: 23BIT276D

import numpy as np

from common import load_image, show_grid

rng = np.random.default_rng(7)


def salt_and_pepper(image, amount=0.25, salt_ratio=0.5):
    """amount = fraction of pixels corrupted,
    salt_ratio = share of those that become white (rest become black)."""
    noisy = image.copy()
    r = rng.random(image.shape)
    salt_cut = amount * salt_ratio
    noisy[r < salt_cut] = 255
    noisy[(r >= salt_cut) & (r < amount)] = 0
    return noisy


image = load_image(gray=True)

levels = [0.05, 0.15, 0.25, 0.40]
noisy_images = [salt_and_pepper(image, a) for a in levels]

for a, n in zip(levels, noisy_images):
    changed = np.mean(n != image) * 100
    print(f"amount={a:.2f} -> {changed:.1f}% pixels changed")

# Mostly-salt vs mostly-pepper at the same amount
salty = salt_and_pepper(image, 0.25, salt_ratio=0.9)
peppery = salt_and_pepper(image, 0.25, salt_ratio=0.1)

show_grid(
    [image] + noisy_images + [salty, peppery],
    ["Original"] + [f"Noise {int(a * 100)}%" for a in levels]
    + ["25% (90% salt)", "25% (90% pepper)"],
    cols=4, out_name="ex2_noise.png",
)
