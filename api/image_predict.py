"""EfficientNet-B0 benign/malignant image classifier.

PyTorch and the weights are loaded lazily on the first image request, so the
tabular API starts quickly and stays within a small instance's memory.
"""

import io
import threading

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from pathlib import Path

import json

MODEL_PATH = Path(__file__).resolve().parents[1] / "image_models" / "cancer_efficientnet.pth"
# Optional sidecar written by src/train_image_model.py: preprocessing + evaluation metrics.
CARD_PATH = MODEL_PATH.with_suffix(".json")
LABELS = {0: "benign", 1: "malignant"}


def model_card() -> dict:
    """Preprocessing and metrics for the deployed weights (empty for the original model)."""
    try:
        return json.loads(CARD_PATH.read_text())
    except (OSError, ValueError):
        return {}

_model = None
_transform = None
_torch = None
_lock = threading.Lock()


def _load():
    global _model, _transform, _torch
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        import torch
        import torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as transforms

        torch.set_num_threads(1)
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()

        card = model_card()
        size = int(card.get("image_size", 224))
        steps = [transforms.Resize((size, size)), transforms.ToTensor()]
        if card.get("normalize"):
            steps.append(transforms.Normalize(card["normalize"]["mean"], card["normalize"]["std"]))
        _transform = transforms.Compose(steps)
        _torch = torch
        _model = model


def image_model_loaded() -> bool:
    return _model is not None


def _gradcam_png(img, class_idx: int, original: Image.Image) -> str:
    """Grad-CAM over the last EfficientNet feature block, overlaid on the image.

    Returns a base64 PNG. Red areas are the regions that pushed the model
    towards the predicted class.
    """
    import base64

    import numpy as np

    torch = _torch
    acts = {}

    def keep(_module, _inp, out):
        acts["a"] = out

    handle = _model.features[-1].register_forward_hook(keep)
    try:
        with torch.enable_grad():
            logits = _model(img.clone().requires_grad_(True))
            (grad,) = torch.autograd.grad(logits[0, class_idx], acts["a"])
    finally:
        handle.remove()

    a = acts["a"][0].detach()              # C x h x w
    weights = grad[0].mean(dim=(1, 2))      # C
    cam = torch.relu((weights[:, None, None] * a).sum(0)).numpy()
    if cam.max() > 0:
        cam = cam / cam.max()

    base = original.convert("RGB")
    base.thumbnail((448, 448))
    heat = Image.fromarray(np.uint8(cam * 255)).resize(base.size, Image.BILINEAR)
    h = np.asarray(heat, dtype=np.float32) / 255.0
    colour = np.stack([h, np.zeros_like(h), 1.0 - h], axis=-1) * 255  # blue (low) -> red (high)
    alpha = (0.15 + 0.45 * h)[..., None]
    blend = np.asarray(base, dtype=np.float32) * (1 - alpha) + colour * alpha
    out = Image.fromarray(np.uint8(blend.clip(0, 255)))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def predict_image(file: UploadFile, explain: bool = False):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image larger than 10 MB")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Not a valid image file")

    _load()
    img = _transform(image).unsqueeze(0)
    with _torch.no_grad():
        probs = _torch.softmax(_model(img), dim=1)[0]

    pred = int(_torch.argmax(probs).item())
    result = {
        "prediction": LABELS[pred],
        "confidence": round(float(probs[pred]), 4),
        "probabilities": {LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "model": "EfficientNet-B0 (transfer learning)",
    }
    if explain:
        result["gradcam_png_base64"] = _gradcam_png(img, pred, image)
        result["gradcam_note"] = ("Grad-CAM on the last feature block: red marks the regions that "
                                  "most increased the predicted class score. A sanity check, not a diagnosis.")
    return result
