"""Read the TN3K test set (Gong et al.) for external validation.

TN3K / TNCD as released with "Less is More: Adaptive Curriculum Learning for
Thyroid Nodule Diagnosis" (MICCAI 2022) contains, somewhere under its root:
    test-image/<name>.jpg
    label4test.csv          lines of "<name>,<label>"  (0 benign, 1 malignant)

The archive layout has varied between releases, so the files are located by
searching under the given root. No PyTorch dependency.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def find_test_set(root: str | Path) -> Tuple[Path, Path]:
    """Return (test_image_dir, label_csv) found under root."""
    root = Path(root)
    labels = sorted(root.rglob("label4test.csv"))
    if not labels:
        raise FileNotFoundError(f"label4test.csv not found under {root}")
    for label_csv in labels:
        for candidate in (label_csv.parent / "test-image", *label_csv.parent.rglob("test-image")):
            if candidate.is_dir():
                return candidate, label_csv
    raise FileNotFoundError(f"test-image/ folder not found next to {labels[0]}")


def load_test_items(root: str | Path) -> List[Tuple[Path, int]]:
    """List (image_path, label) for every labelled TN3K test image."""
    image_dir, label_csv = find_test_set(root)
    items = []
    for line in label_csv.read_text().splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        name, label = [p.strip() for p in line.split(",", 1)]
        if not label.isdigit():  # header row
            continue
        path = image_dir / name
        if path.exists():
            items.append((path, int(label)))
    if not items:
        raise ValueError(f"No labelled images matched between {label_csv} and {image_dir}")
    return items
