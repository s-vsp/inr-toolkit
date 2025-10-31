#!/usr/bin/env python3
"""
compute_image_metrics.py

Compute PSNR, SSIM, MSE, and LPIPS between a ground-truth image and a reconstructed image.
Print results with up to 3 digits (values < 0.001 shown in scientific notation with 3 significant digits).

Dependencies (install if needed):
    pip install numpy pillow scikit-image torch lpips

Usage:
    python compute_image_metrics.py GT.png REC.png
    # Optional flags:
    #   --resize         : resize REC to GT's size (bicubic) if shapes differ
    #   --lpips-net alex : backbone for LPIPS (alex|vgg|squeeze). Default: alex
    #   --require-lpips  : exit with error if LPIPS deps are missing (default: compute what is available)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# SSIM (from scikit-image)
try:
    from skimage.metrics import structural_similarity as ssim_fn
    _HAVE_SKIMAGE = True
except Exception as e:
    _HAVE_SKIMAGE = False
    _SKIMAGE_ERR = e

# LPIPS (requires torch + lpips); handled as optional
try:
    import torch
    import lpips
    _HAVE_LPIPS = True
except Exception as e:
    _HAVE_LPIPS = False
    _LPIPS_ERR = e


def fmt3(x: float) -> str:
    """Format with up to 3 digits: use 3 decimals normally, scientific if very small, strip trailing zeros."""
    if x is None:
        return "n/a"
    if np.isinf(x) or np.isnan(x):
        return str(x)
    ax = abs(float(x))
    if 0 < ax < 1e-3:
        return f"{x:.3e}"
    s = f"{x:.3f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def load_rgb(path: Path) -> np.ndarray:
    """Load image as float32 RGB in [0,1]."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    # Clamp for safety
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def ensure_same_shape(gt: np.ndarray, rec: np.ndarray, resize: bool) -> np.ndarray:
    """Ensure shapes match; optionally resize reconstructed to match GT."""
    if gt.shape == rec.shape:
        return rec
    if not resize:
        raise ValueError(
            f"Shape mismatch: GT {gt.shape} vs REC {rec.shape}. "
            f"Use --resize to match REC to GT."
        )
    # resize REC to GT (width, height)
    h, w = gt.shape[:2]
    rec_img = Image.fromarray((rec * 255.0).astype(np.uint8))
    rec_img = rec_img.resize((w, h), resample=Image.BICUBIC)
    rec_resized = np.asarray(rec_img).astype(np.float32) / 255.0
    return rec_resized


def compute_mse(gt: np.ndarray, rec: np.ndarray) -> float:
    return float(np.mean((gt - rec) ** 2))


def compute_psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def compute_ssim(gt: np.ndarray, rec: np.ndarray) -> float:
    if not _HAVE_SKIMAGE:
        raise RuntimeError(
            f"scikit-image not available: {type(_SKIMAGE_ERR).__name__}: {_SKIMAGE_ERR}\n"
            f"Install with: pip install scikit-image"
        )
    # skimage >= 0.19 uses channel_axis; older versions use multichannel
    try:
        val = ssim_fn(gt, rec, data_range=1.0, channel_axis=-1)
    except TypeError:
        val = ssim_fn(gt, rec, data_range=1.0, multichannel=True)
    return float(val)


def compute_lpips(gt: np.ndarray, rec: np.ndarray, net: str = "alex") -> float:
    if not _HAVE_LPIPS:
        raise RuntimeError(
            f"LPIPS not available (needs torch + lpips): {type(_LPIPS_ERR).__name__}: {_LPIPS_ERR}\n"
            f"Install with: pip install torch lpips"
        )
    # to torch in [-1,1], NCHW
    def to_torch(x: np.ndarray) -> "torch.Tensor":
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
        t = t * 2.0 - 1.0
        return t

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ta = to_torch(gt).to(dev)
    tb = to_torch(rec).to(dev)
    metric = lpips.LPIPS(net=net).to(dev)
    with torch.no_grad():
        d = metric(ta, tb)
    return float(d.item())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute PSNR, SSIM, MSE, and LPIPS between two images.")
    p.add_argument("gt", help="Ground-truth image path")
    p.add_argument("rec", help="Reconstructed image path")
    p.add_argument("--resize", action="store_true",
                   help="Resize REC to match GT size (bicubic) if shapes differ")
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"],
                   help="Backbone net for LPIPS (default: alex)")
    p.add_argument("--require-lpips", action="store_true",
                   help="Exit with error if LPIPS deps are missing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = Path(args.gt)
    rec_path = Path(args.rec)

    if not gt_path.exists():
        print(f"Error: ground-truth image not found: {gt_path}", file=sys.stderr)
        sys.exit(1)
    if not rec_path.exists():
        print(f"Error: reconstructed image not found: {rec_path}", file=sys.stderr)
        sys.exit(1)

    gt = load_rgb(gt_path)
    rec = load_rgb(rec_path)
    rec = ensure_same_shape(gt, rec, resize=args.resize)

    # Compute basic metrics
    mse = compute_mse(gt, rec)
    psnr = compute_psnr_from_mse(mse, data_range=1.0)

    # SSIM
    try:
        ssim_val = compute_ssim(gt, rec)
    except Exception as e:
        print(f"Warning: SSIM unavailable: {e}", file=sys.stderr)
        ssim_val = None

    # LPIPS
    lpips_val = None
    if _HAVE_LPIPS:
        try:
            lpips_val = compute_lpips(gt, rec, net=args.lpips_net)
        except Exception as e:
            print(f"Warning: LPIPS failed: {e}", file=sys.stderr)
            if args.require_lpips:
                sys.exit(2)
    else:
        if args.require_lpips:
            print(f"Error: LPIPS not available: {type(_LPIPS_ERR).__name__}: {_LPIPS_ERR}", file=sys.stderr)
            sys.exit(2)

    # Print results with up to 3 digits
    print(f"PSNR (dB): {fmt3(psnr)}")
    print(f"SSIM    : {fmt3(ssim_val)}")
    print(f"MSE     : {fmt3(mse)}")
    print(f"LPIPS   : {fmt3(lpips_val)}")


if __name__ == "__main__":
    main()
