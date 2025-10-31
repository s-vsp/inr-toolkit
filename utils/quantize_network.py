#!/usr/bin/env python3
"""
quantize_inr.py

Create NF4, Linear8bitLt (LLM.int8), and INT4 (PTQ) quantized variants of a SIREN/BACON INR
that encodes a single image.

Requirements:
    python -m pip install torch pillow safetensors bitsandbytes scikit-image

Usage:
    python quantize_inr.py \
        --arch siren \
        --depth 12 \
        --width 512 \
        --ckpt /path/to/model.safetensors \
        --image /path/to/target_image.png \
        --outdir ./quantized_out \
        --device cuda \
        --int4-keep-first-last

Outputs (in --outdir):
    - *_nf4.pt        : model state_dict + meta; nn.Linear -> bnb.nn.Linear4bit (NF4)
    - *_int8lt.pt     : model state_dict + meta; nn.Linear -> bnb.nn.Linear8bitLt
    - *_int4_ptq.pt   : model state_dict + meta; nn.Linear -> Int4LinearDequant (scales + packed int4)
    - *_int4_meta.pth : extra metadata for INT4 (per-layer scales, config)

Notes:
    * NF4 and Linear8bitLt quantize when the model is moved to CUDA; these paths require a GPU.
    * INT4 PTQ is weight-only and works on CPU or GPU. By default it keeps first/last linear in FP.
    * Assumes 2D input coords in [-1,1]^2 and 3-channel RGB output.
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional deps
try:
    from safetensors.torch import load_file as safe_load_file, save_file as safe_save_file
    _HAVE_SAFE = True
except Exception:
    _HAVE_SAFE = False


# ---- Model definitions --------------------------------------------------------

class Sine(nn.Module):
    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, w0: float, is_first: bool = False, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.act = Sine(w0=w0)
        self.is_first = is_first
        self.w0 = w0
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = np.sqrt(6 / self.linear.in_features) / self.w0
                self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.fill_(0.0)

    def forward(self, x):
        return self.act(self.linear(x))


class Siren(nn.Module):
    def __init__(self, in_dim=2, out_dim=3, depth=5, width=256, first_w0=30.0, hidden_w0=30.0):
        super().__init__()
        layers: List[nn.Module] = []
        layers.append(SirenLayer(in_dim, width, w0=first_w0, is_first=True))
        for _ in range(depth - 2):
            layers.append(SirenLayer(width, width, w0=hidden_w0))
        self.hidden = nn.Sequential(*layers)
        self.final = nn.Linear(width, out_dim)

    def forward(self, x):
        x = self.hidden(x)
        return self.final(x)


class Bacon(nn.Module):
    """
    Simplified 'BACON-like' MLP: sine activations with a mild frequency ramp across layers.
    (Not a paper-accurate impl; good enough for quantization experiments.)
    """
    def __init__(self, in_dim=2, out_dim=3, depth=5, width=256, first_w0=15.0, hidden_w0=15.0, ramp: float = 1.15):
        super().__init__()
        layers: List[nn.Module] = []
        layers.append(SirenLayer(in_dim, width, w0=first_w0, is_first=True))
        w0 = hidden_w0
        for _ in range(depth - 2):
            layers.append(SirenLayer(width, width, w0=w0))
            w0 *= ramp
        self.hidden = nn.Sequential(*layers)
        self.final = nn.Linear(width, out_dim)

    def forward(self, x):
        x = self.hidden(x)
        return self.final(x)


# ---- Bitsandbytes replacements ------------------------------------------------

def clone_with_bnb_linear(model: nn.Module, kind: str = "nf4", device: torch.device = torch.device("cuda")) -> nn.Module:
    """
    Copy the model; replace nn.Linear with bitsandbytes quantized linears.

    kind:
        - "nf4"  -> bnb.nn.Linear4bit(quant_type='nf4')
        - "int8" -> bnb.nn.Linear8bitLt (LLM.int8)
    """
    import copy
    import bitsandbytes as bnb

    m2 = copy.deepcopy(model)

    def replace(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                in_f, out_f = child.in_features, child.out_features
                bias = child.bias is not None
                if kind == "nf4":
                    q = bnb.nn.Linear4bit(in_f, out_f, bias=bias, compute_dtype=torch.bfloat16, quant_type="nf4")
                elif kind == "int8":
                    q = bnb.nn.Linear8bitLt(in_f, out_f, bias=bias, has_fp16_weights=False)
                else:
                    raise ValueError(f"Unknown kind: {kind}")
                q.load_state_dict(child.state_dict())  # load FP weights; quantization occurs on .to(device)
                setattr(module, name, q)
            else:
                replace(child)

    replace(m2)
    m2 = m2.to(device)  # trigger quantization for bnb modules
    return m2


# ---- INT4 PTQ (weight-only, per-channel symmetric) ---------------------------

class Int4LinearDequant(nn.Module):
    """
    Stores per-output-channel int4 weights (packed into uint8) + scales, dequantizes on-the-fly.
    Weight shape: (out, in) quantized to values in [-8, 7], symmetric per-channel scale.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        # Quantized params (buffers)
        self.register_buffer("qweight", torch.empty((out_features, (in_features + 1) // 2), dtype=torch.uint8))
        self.register_buffer("scale", torch.ones(out_features, dtype=torch.float32))

    @staticmethod
    def quantize_per_channel(W: torch.Tensor, scale: torch.Tensor):
        """Quantize to int4 per output channel and pack two nibbles into one byte (uint8)."""
        q = torch.round(W / scale[:, None]).clamp(-8, 7).to(torch.int16)  # [-8,7]
        q_u = (q + 8).to(torch.uint8)  # [0,15]
        low = q_u[:, 0::2]
        high = q_u[:, 1::2]
        if high.shape[1] < low.shape[1]:
            pad = torch.zeros((q.shape[0], 1), dtype=torch.uint8, device=q.device)
            high = torch.cat([high, pad], dim=1)
        packed = (low | (high << 4)).contiguous()
        return packed

    @staticmethod
    def dequantize_per_channel(packed: torch.Tensor, scale: torch.Tensor, in_features: int) -> torch.Tensor:
        """Unpack uint8 bytes into signed int4 values in [-8,7] and dequantize."""
        u = packed.to(torch.uint8)
        low_u = (u & 0x0F).to(torch.int16)        # [0,15]
        high_u = ((u >> 4) & 0x0F).to(torch.int16)
        low = (low_u - 8).to(torch.int8)          # [-8,7]
        high = (high_u - 8).to(torch.int8)
        q = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.int8, device=packed.device)
        q[:, 0::2] = low
        q[:, 1::2] = high
        q = q[:, :in_features]
        W = q.to(torch.float32) * scale[:, None]
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.dequantize_per_channel(self.qweight, self.scale, self.in_features)
        out = F.linear(x, W, self.bias)
        return out


def calibrate_linear_int4(W: torch.Tensor, X: torch.Tensor, b: torch.Tensor = None,
                          n_scales: int = 5, widen: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Layerwise per-channel INT4 calibration: choose scales that minimize MSE on outputs for inputs X.
    Returns: (packed_qweight (uint8), best_scale (float))
    """
    device = W.device
    with torch.no_grad():
        Y_ref = X @ W.t()
        if b is not None:
            Y_ref = Y_ref + b

        s0 = W.abs().amax(dim=1) / 7.0
        s0 = torch.clamp(s0, min=1e-8)

        alphas = torch.linspace(1 - widen, 1 + widen, steps=n_scales, device=device)
        best_scale = s0.clone()
        best_err = torch.full_like(s0, float("inf"))

        for a in alphas:
            s = torch.clamp(s0 * a, min=1e-8)
            q = torch.round(W / s[:, None]).clamp(-8, 7)
            Yq = X @ (q * s[:, None]).t()
            if b is not None:
                Yq = Yq + b
            err = ((Y_ref - Yq) ** 2).mean(dim=0)  # per-output channel
            improved = err < best_err
            best_err[improved] = err[improved]
            best_scale[improved] = s[improved]

        qweight_packed = Int4LinearDequant.quantize_per_channel(W, best_scale)
        return qweight_packed, best_scale


def collect_layer_inputs(model: nn.Module, layers: List[nn.Linear], X: torch.Tensor) -> List[torch.Tensor]:
    """
    Forward once and collect the input activations to each linear layer.
    Returns list of inputs aligned with `layers` order.
    """
    inputs: List[torch.Tensor] = []
    hooks = []

    def make_hook():
        def hook(mod, inp):
            inputs.append(inp[0].detach())
        return hook

    for layer in layers:
        hooks.append(layer.register_forward_pre_hook(make_hook()))

    with torch.no_grad():
        _ = model(X)

    for h in hooks:
        h.remove()
    return inputs


def find_parent_module(root: nn.Module, target: nn.Module) -> nn.Module:
    for name, module in root.named_modules():
        for child_name, child in module.named_children():
            if child is target:
                return module
    raise RuntimeError("Parent module not found")


def replace_child(parent: nn.Module, child_old: nn.Module, child_new: nn.Module):
    for name, module in parent.named_children():
        if module is child_old:
            setattr(parent, name, child_new)
            return
    raise RuntimeError("replace_child failed: child not found")


def ptq_int4_model(model: nn.Module, image: Image.Image, n_samples: int = 100_000,
                   keep_first_last: bool = True, device: torch.device = torch.device("cuda"),
                   n_scales: int = 5, widen: float = 0.4) -> Tuple[nn.Module, Dict]:
    """
    Post-training INT4 quantization (weight-only) with per-channel scales.
    Uses coordinates sampled from the provided image as calibration inputs.
    """
    import copy
    m2 = copy.deepcopy(model).to(device)
    m2.eval()

    # Build coordinate samples in [-1,1]^2
    Wimg, Himg = image.size
    xs = np.random.randint(0, Wimg, size=(n_samples,))
    ys = np.random.randint(0, Himg, size=(n_samples,))
    cx = (xs / (Wimg - 1)) * 2.0 - 1.0
    cy = (ys / (Himg - 1)) * 2.0 - 1.0
    X = torch.tensor(np.stack([cx, cy], axis=1), dtype=torch.float32, device=device)

    # Collect linear layers and their inputs
    linear_layers: List[nn.Linear] = [m for m in m2.modules() if isinstance(m, nn.Linear)]
    layer_inputs = collect_layer_inputs(m2, linear_layers, X)

    meta = {"layers": []}
    for idx, (layer, Xin) in enumerate(zip(linear_layers, layer_inputs)):
        quantize_this = not (keep_first_last and (idx == 0 or idx == len(linear_layers) - 1))
        layer_meta = {"index": idx, "in": layer.in_features, "out": layer.out_features, "quantized": bool(quantize_this)}
        if not quantize_this:
            meta["layers"].append(layer_meta)
            continue

        W = layer.weight.data.to(device)
        b = layer.bias.data.to(device) if layer.bias is not None else None

        if Xin.shape[0] > n_samples:
            Xin = Xin[:n_samples]
        qpacked, scale = calibrate_linear_int4(W, Xin, b=b, n_scales=n_scales, widen=widen)

        # Replace with dequant layer
        qlayer = Int4LinearDequant(layer.in_features, layer.out_features, bias=(layer.bias is not None)).to(device)
        qlayer.qweight.copy_(qpacked)
        qlayer.scale.copy_(scale)
        if layer.bias is not None:
            qlayer.bias.data.copy_(b)

        parent = find_parent_module(m2, layer)
        replace_child(parent, layer, qlayer)

        layer_meta["scale_mean"] = float(scale.mean().item())
        meta["layers"].append(layer_meta)

    return m2, meta


# ---- Utilities ----------------------------------------------------------------

def build_model(arch: str, depth: int, width: int, in_dim: int = 2, out_dim: int = 3,
                first_w0: float = 30.0, hidden_w0: float = 30.0) -> nn.Module:
    arch = arch.lower()
    if arch == "siren":
        return Siren(in_dim=in_dim, out_dim=out_dim, depth=depth, width=width,
                     first_w0=first_w0, hidden_w0=hidden_w0)
    elif arch == "bacon":
        return Bacon(in_dim=in_dim, out_dim=out_dim, depth=depth, width=width,
                     first_w0=first_w0, hidden_w0=hidden_w0)
    else:
        raise ValueError(f"Unknown arch: {arch}")


def load_weights_into_model(model: nn.Module, ckpt_path: Path):
    if ckpt_path.suffix == ".safetensors" and _HAVE_SAFE:
        sd = safe_load_file(str(ckpt_path))
    else:
        obj = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj:
            sd = obj["state_dict"]
        else:
            sd = obj
    try:
        model.load_state_dict(sd, strict=True)
        return
    except Exception:
        prefixes = ["model.", "module.", "net."]
        for p in prefixes:
            sd2 = {k[len(p):] if k.startswith(p) else k: v for k, v in sd.items()}
            try:
                model.load_state_dict(sd2, strict=True)
                return
            except Exception:
                pass
        raise


def read_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def save_checkpoint_state(out_path: Path, model: nn.Module, meta: Dict):
    payload = {"state_dict": model.state_dict(), "meta": meta}
    torch.save(payload, str(out_path))


# ---- CLI ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Quantize an INR (SIREN/BACON) into NF4, INT8 (LLM.int8), and INT4 PTQ variants.")
    p.add_argument("--arch", choices=["siren", "bacon"], default="siren")
    p.add_argument("--depth", type=int, required=True, help="Total layers incl. first and last (>=2)")
    p.add_argument("--width", type=int, required=True, help="Hidden layer width")
    p.add_argument("--in-channels", type=int, default=2)
    p.add_argument("--out-channels", type=int, default=3)

    p.add_argument("--ckpt", type=str, required=True, help=".pt or .safetensors checkpoint with float weights")
    p.add_argument("--image", type=str, required=True, help="Target image for INT4 calibration (path)")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu). bnb paths need CUDA.")

    # SIREN/BACON params
    p.add_argument("--first-w0", type=float, default=30.0)
    p.add_argument("--hidden-w0", type=float, default=30.0)

    # INT4 PTQ params
    p.add_argument("--int4-samples", type=int, default=100000, help="Calibration sample count")
    p.add_argument("--int4-keep-first-last", action="store_true", help="Keep first and last linear in FP")
    p.add_argument("--int4-n-scales", type=int, default=5, help="Grid size for scale search")
    p.add_argument("--int4-widen", type=float, default=0.4, help="Scale search +/- range")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build and load base FP model
    base = build_model(args.arch, args.depth, args.width, args.in_channels, args.out_channels,
                       first_w0=args.first_w0, hidden_w0=args.hidden_w0)
    load_weights_into_model(base, Path(args.ckpt))
    base.eval()

    # 1) NF4 (bnb Linear4bit with quant_type='nf4')
    try:
        nf4_model = clone_with_bnb_linear(base, kind="nf4", device=device)
        nf4_meta = {"arch": args.arch, "quant": "nf4", "depth": args.depth, "width": args.width}
        save_checkpoint_state(outdir / "model_nf4.pt", nf4_model, nf4_meta)
        print(f"✅ Saved NF4 quantized model: {outdir / 'model_nf4.pt'}")
    except Exception as e:
        print(f"⚠️ NF4 quantization failed (bitsandbytes?): {e}")

    # 2) Linear8bitLt (LLM.int8())
    try:
        int8_model = clone_with_bnb_linear(base, kind="int8", device=device)
        int8_meta = {"arch": args.arch, "quant": "int8_lt", "depth": args.depth, "width": args.width}
        save_checkpoint_state(outdir / "model_int8lt.pt", int8_model, int8_meta)
        print(f"✅ Saved Linear8bitLt quantized model: {outdir / 'model_int8lt.pt'}")
    except Exception as e:
        print(f"⚠️ INT8 (Linear8bitLt) quantization failed (bitsandbytes?): {e}")

    # 3) INT4 PTQ with calibration from the image
    try:
        img = read_image(Path(args.image))
        int4_model, int4_meta = ptq_int4_model(
            base, img, n_samples=args.int4_samples, keep_first_last=args.int4_keep_first_last,
            device=device, n_scales=args.int4_n_scales, widen=args.int4_widen
        )
        int4_meta.update({"arch": args.arch, "quant": "int4_ptq", "depth": args.depth, "width": args.width})
        save_checkpoint_state(outdir / "model_int4_ptq.pt", int4_model, int4_meta)
        torch.save(int4_meta, outdir / "model_int4_meta.pth")
        print(f"✅ Saved INT4 PTQ model: {outdir / 'model_int4_ptq.pt'}")
    except Exception as e:
        print(f"⚠️ INT4 PTQ failed: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
