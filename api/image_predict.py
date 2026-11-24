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


async def predict_image(file: UploadFile):
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
    return {
        "prediction": LABELS[pred],
        "confidence": round(float(probs[pred]), 4),
        "probabilities": {LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "model": "EfficientNet-B0 (transfer learning)",
    }
