"""Optional, experimental image classifier (EfficientNet-B0 transfer learning).

Not part of the default build: install ``requirements-image.txt`` and set
``IMAGE_MODEL_PATH`` to enable the ``/v1/predict-image`` endpoint. The training
images are not included in this repository; see scripts/train_image_model.py.
"""

from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

DEFAULT_WEIGHTS = Path("image_models/cancer_efficientnet.pth")
LABELS = ("benign", "malignant")


def _weights_path() -> Path:
    return Path(os.getenv("IMAGE_MODEL_PATH", str(DEFAULT_WEIGHTS)))


def available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        return False
    return _weights_path().exists()


@lru_cache(maxsize=1)
def _load():
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms

    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(LABELS))
    model.load_state_dict(torch.load(_weights_path(), map_location="cpu", weights_only=True))
    model.eval()
    # Must match the transform used in training (scripts/train_image_model.py).
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return model, transform


def predict(image_bytes: bytes) -> dict:
    import torch
    from PIL import Image

    model, transform = _load()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    with torch.no_grad():
        probs = torch.softmax(model(transform(image).unsqueeze(0)), dim=1)[0]
    idx = int(torch.argmax(probs))
    return {"prediction": LABELS[idx], "confidence": round(float(probs[idx]), 4), "experimental": True}
