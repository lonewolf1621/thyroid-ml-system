"""Train the EfficientNet-B0 benign/malignant classifier on TN5000.

Uses the dataset's official splits so the reported test score is honest:
  train (3,500) -> fit, val (500) -> pick the best epoch, test (1,000) -> scored once.

    python src/train_image_model.py --data /path/to/TN5000_forReview.zip --out outputs/

Writes to --out:
  cancer_efficientnet.pth   state_dict for api/image_predict.py
  cancer_efficientnet.json  preprocessing + metrics (read by the API)
  metrics.json, roc.png, confusion_matrix.png, history.csv
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tn5000 import LABELS, TN5000Index  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


class TN5000Dataset(Dataset):
    def __init__(self, index: TN5000Index, split: str, transform):
        self.index, self.items, self.transform = index, index.split(split), transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        image_id, label = self.items[i]
        img = Image.open(io.BytesIO(self.index.image_bytes(image_id))).convert("RGB")
        return self.transform(img), label


def build_transforms():
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        norm,
    ])
    evaluate = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        norm,
    ])
    return train, evaluate


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        ps.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def metrics(y, p, threshold=0.5):
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                                 roc_auc_score)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "f1": float(f1_score(y, pred)),
        "auc": float(roc_auc_score(y, p)),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "majority_class_baseline": float(max(y.mean(), 1 - y.mean())),
    }


def save_plots(y, p, m, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except ImportError:
        return
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {m['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC, TN5000 test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc.png", dpi=150)
    plt.close(fig)

    cm = m["confusion_matrix"]
    grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(grid, cmap="Blues")
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                color="white" if v > grid.max() / 2 else "black")
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix, TN5000 test set")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="TN5000 zip or extracted folder")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    index = TN5000Index(args.data)
    t_train, t_eval = build_transforms()
    ds = {s: TN5000Dataset(index, s, t_train if s == "train" else t_eval) for s in ("train", "val", "test")}
    for s, d in ds.items():
        y = np.array([l for _, l in d.items])
        print(f"{s}: {len(d)} images ({int(y.sum())} malignant, {int(len(y) - y.sum())} benign)")

    loaders = {
        s: DataLoader(d, batch_size=args.batch_size, shuffle=(s == "train"),
                      num_workers=args.workers, pin_memory=device.type == "cuda")
        for s, d in ds.items()
    }

    # Class weights: the training split is ~70% malignant.
    y_train = np.array([l for _, l in ds["train"].items])
    counts = np.bincount(y_train, minlength=2)
    class_weights = torch.tensor(len(y_train) / (2.0 * counts), dtype=torch.float32, device=device)
    print("class weights:", class_weights.tolist())

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_auc, best_state, history = -1.0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
        scheduler.step()

        yv, pv = predict(model, loaders["val"], device)
        mv = metrics(yv, pv)
        row = {"epoch": epoch, "train_loss": running / len(ds["train"]), "val_auc": mv["auc"],
               "val_accuracy": mv["accuracy"], "val_balanced_accuracy": mv["balanced_accuracy"],
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(row)
        if mv["auc"] > best_auc:
            best_auc = mv["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    best_epoch = max(history, key=lambda r: r["val_auc"])["epoch"]

    # Score the untouched test split once.
    yt, pt = predict(model, loaders["test"], device)
    mt = metrics(yt, pt)
    yv, pv = predict(model, loaders["val"], device)
    mv = metrics(yv, pv)
    print("TEST:", json.dumps(mt, indent=1))

    torch.save(best_state, out / "cancer_efficientnet.pth")
    card = {
        "architecture": "efficientnet_b0",
        "labels": LABELS,
        "image_size": IMAGE_SIZE,
        "normalize": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "training": {
            "dataset": "TN5000 (Zhang et al., Scientific Data 2025), official train/val/test split",
            "epochs": args.epochs, "best_epoch": best_epoch, "batch_size": args.batch_size,
            "lr": args.lr, "optimizer": "AdamW + cosine", "class_weights": class_weights.tolist(),
            "pretrained": "ImageNet", "seed": args.seed,
        },
        "metrics": {"val": mv, "test": mt},
    }
    (out / "cancer_efficientnet.json").write_text(json.dumps(card, indent=2))
    (out / "metrics.json").write_text(json.dumps({"val": mv, "test": mt}, indent=2))
    with open(out / "history.csv", "w") as f:
        f.write(",".join(history[0]) + "\n")
        for r in history:
            f.write(",".join(str(v) for v in r.values()) + "\n")
    save_plots(yt, pt, mt, out)
    print(f"Saved model and reports to {out.resolve()}")


if __name__ == "__main__":
    main()
