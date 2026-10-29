"""Helper functions shared by every experiment."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "images"
OUTPUTS = ROOT / "outputs"


def load_image(name="input.jpg", gray=False):
    path = IMAGES / name
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Put an image at {path} and run again.")
    return img


def save_figure(fig, name):
    OUTPUTS.mkdir(exist_ok=True)
    out = OUTPUTS / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("Saved:", out.relative_to(ROOT))
    plt.show()


def show_grid(images, titles, cols, out_name, figsize=None):
    """Show several images in one figure and save it to outputs/."""
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (4 * cols, 4 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    fig.tight_layout()
    save_figure(fig, out_name)
