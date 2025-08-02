import os
import base64
import torch
from io import BytesIO
from .clip_transform import _transform

def load(device="cpu"):
    path = os.path.join(os.path.dirname(__file__), "model_encoded.txt")
    with open(path, "r", encoding="ascii") as f:
        encoded = f.read()

    model_bytes = base64.b64decode(encoded)
    buffer = BytesIO(model_bytes)
    model = torch.jit.load(buffer, map_location=device).eval()
    return model, _transform()
