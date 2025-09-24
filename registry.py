import torch.nn as nn

from functools import partial
from siren_pytorch import SirenNet

MODEL_REGISTRY = {
    "SIREN": partial(SirenNet, final_activation=nn.Identity())
}