"""Score a trained image model on the independent TN3K test set.

TN3K comes from a different hospital and study than TN5000 (the training
data), so this measures how well the model generalises to new sources.

    python src/evaluate_external.py --model outputs/cancer_efficientnet.pth --tn3k /content/tn3k

Writes external_tn3k_metrics.json, external_tn3k_roc.png and
external_tn3k_confusion_matrix.png next to the model, and adds the result
to the model card (cancer_efficientnet.json) under metrics.external_tn3k.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tn3k import load_test_items  # noqa: E402
from train_image_model import build_model, metrics, predict, save_plots  # noqa: E402


class Images(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="cancer_efficientnet.pth")
    ap.add_argument("--tn3k", required=True, help="folder containing the extracted TN3K data")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    model_path = Path(args.model)
    card_path = model_path.with_suffix(".json")
    card = json.loads(card_path.read_text()) if card_path.exists() else {}
    size = int(card.get("image_size", 224))
    steps = [transforms.Resize((size, size)), transforms.ToTensor()]
    if card.get("normalize"):
        steps.append(transforms.Normalize(card["normalize"]["mean"], card["normalize"]["std"]))

    items = load_test_items(args.tn3k)
    y = np.array([l for _, l in items])
    print(f"TN3K test: {len(items)} images ({int(y.sum())} malignant, {int(len(y) - y.sum())} benign)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    loader = DataLoader(Images(items, transforms.Compose(steps)), batch_size=args.batch_size)

    yt, pt = predict(model, loader, device)
    m = metrics(yt, pt)
    m["dataset"] = "TN3K test set (Gong et al.), external: different hospital and study from training data"
    print("EXTERNAL (TN3K):", json.dumps(m, indent=1))

    out = model_path.parent
    (out / "external_tn3k_metrics.json").write_text(json.dumps(m, indent=2))
    save_plots(yt, pt, m, out, dataset="TN3K external test set", prefix="external_tn3k_")
    if card:
        card.setdefault("metrics", {})["external_tn3k"] = m
        card_path.write_text(json.dumps(card, indent=2))
        print(f"Added external result to {card_path}")


if __name__ == "__main__":
    main()
