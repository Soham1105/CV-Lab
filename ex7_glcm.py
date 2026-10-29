# Experiment 7: GLCM texture features
# Name: Soham   Roll No: 23BIT276D

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from common import save_figure

rng = np.random.default_rng(21)

# Synthetic test image: smooth "water" on the left, rough "sand" on the right
H, W = 400, 600
image = np.zeros((H, W), dtype=np.uint8)
yy = np.arange(H)[:, None]
water = 120 + 15 * np.sin(yy / 12.0) + rng.normal(0, 3, (H, W // 2))
sand = rng.normal(150, 40, (H, W // 2))
image[:, :W // 2] = np.clip(water, 0, 255)
image[:, W // 2:] = np.clip(sand, 0, 255)

PATCH = 25
water_locs = [(40, 40), (150, 120), (260, 60), (330, 200)]
sand_locs = [(40, 360), (150, 460), (260, 380), (330, 520)]


def glcm(patch, offsets=((0, 3), (3, 0)), levels=16):
    """Symmetric, normalised GLCM averaged over several (dy, dx) offsets."""
    q = (patch.astype(np.int32) * levels) // 256
    m = np.zeros((levels, levels))
    h, w = q.shape
    for dy, dx in offsets:
        a = q[:h - dy, :w - dx].ravel()
        b = q[dy:, dx:].ravel()
        np.add.at(m, (a, b), 1)
        np.add.at(m, (b, a), 1)
    return m / m.sum()


def features(P):
    n = P.shape[0]
    i, j = np.indices((n, n))
    mu_i, mu_j = (i * P).sum(), (j * P).sum()
    sd_i = np.sqrt(((i - mu_i) ** 2 * P).sum())
    sd_j = np.sqrt(((j - mu_j) ** 2 * P).sum())
    nz = P[P > 0]
    return {
        "contrast": (P * (i - j) ** 2).sum(),
        "dissimilarity": (P * np.abs(i - j)).sum(),
        "homogeneity": (P / (1 + (i - j) ** 2)).sum(),
        "energy": np.sqrt((P ** 2).sum()),
        "correlation": ((i - mu_i) * (j - mu_j) * P).sum() / (sd_i * sd_j) if sd_i * sd_j else 1.0,
        "entropy": -(nz * np.log2(nz)).sum(),
    }


def patches(locs):
    return [image[r:r + PATCH, c:c + PATCH] for r, c in locs]


water_f = [features(glcm(p)) for p in patches(water_locs)]
sand_f = [features(glcm(p)) for p in patches(sand_locs)]

names = list(water_f[0])
print(f"{'patch':<10}" + "".join(f"{n:>14}" for n in names))
for label, fl in (("water", water_f), ("sand", sand_f)):
    for k, f in enumerate(fl, 1):
        print(f"{label + str(k):<10}" + "".join(f"{f[n]:>14.4f}" for n in names))

# Image with the chosen patches marked
fig, ax = plt.subplots(figsize=(9, 6))
ax.imshow(image, cmap="gray", vmin=0, vmax=255)
for locs, colour in ((water_locs, "cyan"), (sand_locs, "orange")):
    for r, c in locs:
        ax.add_patch(Rectangle((c, r), PATCH, PATCH, edgecolor=colour,
                               facecolor="none", linewidth=2))
ax.set_title("Water patches (cyan) vs sand patches (orange)")
ax.axis("off")
save_figure(fig, "ex7_patches.png")

# 2D feature space: homogeneity vs entropy
fig = plt.figure(figsize=(7, 5))
for label, fl, m in (("Water", water_f, "o"), ("Sand", sand_f, "^")):
    plt.scatter([f["homogeneity"] for f in fl], [f["entropy"] for f in fl],
                s=140, marker=m, label=label)
plt.xlabel("Homogeneity")
plt.ylabel("Entropy")
plt.title("GLCM features: homogeneity vs entropy")
plt.legend()
plt.grid(alpha=0.3)
save_figure(fig, "ex7_feature_space.png")

# 3D feature space
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")
for label, fl, m in (("Water", water_f, "o"), ("Sand", sand_f, "^")):
    ax.scatter([f["contrast"] for f in fl], [f["energy"] for f in fl],
               [f["correlation"] for f in fl], s=140, marker=m, label=label)
ax.set_xlabel("Contrast")
ax.set_ylabel("Energy")
ax.set_zlabel("Correlation")
ax.set_title("3D GLCM feature space")
ax.legend()
save_figure(fig, "ex7_feature_space_3d.png")

# One GLCM matrix from each class
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (label, p) in zip(axes, (("Water 1", patches(water_locs)[0]),
                                 ("Sand 1", patches(sand_locs)[0]))):
    im = ax.imshow(glcm(p), cmap="magma")
    ax.set_title(f"GLCM of {label}")
    fig.colorbar(im, ax=ax)
fig.tight_layout()
save_figure(fig, "ex7_glcm_matrices.png")
