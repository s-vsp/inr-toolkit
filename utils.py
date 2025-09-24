import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from typing import Tuple
from numpy.typing import NDArray


def image_to_coords_and_pixels(img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Converts image to coordinates (X, Y) and color values (RGB)

    Args:
        * img: Input PIL image
    Returns:
        * coords: Torch tensor of X,Y coordinates - (N,2)
        * pixels: Torch tensor of RGB values - (N,3)
        * (H, W): Tuple of integers representing image height and width
    """
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.
    H, W, C = arr.shape

    xs = np.linspace(-1., 1., W)
    ys = np.linspace(-1., 1., H)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    pixels = arr.reshape(-1, 3)

    return torch.from_numpy(coords).float(), torch.from_numpy(pixels).float(), (H, W)


def infer_in_chunks(net: nn.Module, coords: torch.Tensor, device: torch.device, chunk_size: int = 65536) -> NDArray:
    """
    Run net on coords in chunks to avoid OOM.
    
    Args:
        * net: NN (TODO: Adjust for other networks) working on SIREN
        * coords: Torch tensor of coordinates (N,2) on CPU
        * device: Torch device to perform computations on - CUDA
        * chunk_size: Amount of coords which can be fed to the net in a single pass
    Returns: 
        * out: Numpy array (N, 3) on CPU in range [0,1] - predicted image 
    """
    net = net.eval()
    N = coords.shape[0]
    out_list = []
    with torch.no_grad():
        for i in range(0, N, chunk_size):
            c = coords[i:i+chunk_size].to(device)
            out_chunk = net(c)
            out_chunk = out_chunk.clamp(0.0, 1.0).cpu()
            out_list.append(out_chunk)
    out = torch.cat(out_list, dim=0).numpy()
    return out