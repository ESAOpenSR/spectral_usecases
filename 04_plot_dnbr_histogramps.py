import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt


def load_valid_pixels(path):
    """Load raster values and return only valid (non-nodata) pixels."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nod = src.nodata if src.nodata is not None else -9999.0

    return arr[arr != nod]


def otsu_threshold(values, nbins=256):
    """
    Compute Otsu threshold for a 1D array of values.
    Returns the threshold value in the same units as 'values'.
    """
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No valid values provided for thresholding.")

    hist, bin_edges = np.histogram(values, bins=nbins)
    hist = hist.astype("float64")
    prob = hist / hist.sum()

    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    w0 = np.cumsum(prob)
    w1 = 1.0 - w0

    # avoid division by zero
    valid = (w0 > 0) & (w1 > 0)

    mu0 = np.cumsum(prob * bin_mids)
    muT = mu0[-1]
    mu0 = mu0 / np.where(w0 == 0, 1, w0)  # mean of class 0
    mu1 = (muT - mu0 * w0) / np.where(w1 == 0, 1, w1)  # mean of class 1

    sigma_b2 = w0 * w1 * (mu0 - mu1) ** 2
    sigma_b2[~valid] = -np.inf

    idx = np.argmax(sigma_b2)
    thr = bin_mids[idx]
    return float(thr)


def plot_dnbr_histograms_with_threshold(lr_path, sr_path, bins=200, title="dNBR distribution"):
    """
    Plot LR and SR dNBR histograms and compute a global Otsu threshold.
    Returns the threshold value.
    """
    lr_vals = load_valid_pixels(lr_path)
    sr_vals = load_valid_pixels(sr_path)

    all_vals = np.concatenate([lr_vals, sr_vals])
    thr = otsu_threshold(all_vals, nbins=bins)

    plt.figure(figsize=(10, 5))
    plt.hist(lr_vals, bins=bins, alpha=0.5, label="LR dNBR", density=True)
    plt.hist(sr_vals, bins=bins, alpha=0.5, label="SR dNBR", density=True)

    plt.axvline(thr, linestyle="--", linewidth=2, color="red", label=f"Threshold = {thr:.4f}")

    plt.xlabel("dNBR value")
    plt.ylabel("Density")
    plt.title(f"{title}\nOtsu threshold = {thr:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Otsu threshold (LR+SR combined): {thr:.6f}")
    return thr


if __name__ == "__main__":
    # Adjust base folder as needed
    base = "data/products/"

    lr_dnbr = os.path.join(base, "lr_dnbr_masked.tif")
    sr_dnbr = os.path.join(base, "sr_dnbr_masked.tif")

    threshold = plot_dnbr_histograms_with_threshold(lr_dnbr, sr_dnbr)

    # Example: later you can use `threshold` for binary detection, e.g.:
    #   detection = (dnbr_array >= threshold).astype("uint8")
