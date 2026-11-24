"""Read the TN5000 thyroid ultrasound dataset (PASCAL VOC layout).

TN5000 (Zhang et al., Scientific Data 2025) ships as:
    TN5000*/JPEGImages/<id>.jpg
    TN5000*/Annotations/<id>.xml       <object><name>0|1</name> (0 benign, 1 malignant)
    TN5000*/ImageSets/Main/{train,val,test}.txt

This module has no PyTorch dependency so it can be tested anywhere.
"""
from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

LABELS = ["benign", "malignant"]
_NAME = re.compile(r"<name>\s*(\d)\s*</name>")


def _label_from_xml(text: str) -> int:
    names = {int(n) for n in _NAME.findall(text)}
    if not names:
        raise ValueError("annotation has no <name> label")
    # An image with any malignant nodule is labelled malignant.
    return 1 if 1 in names else 0


class TN5000Index:
    """Lists (image_id, label) per official split from a folder or the original zip."""

    def __init__(self, source: str | Path):
        self.source = Path(source)
        self.is_zip = self.source.suffix == ".zip"
        self._zip_handle = None
        self._zip_pid = None
        self.root = self._find_root()

    @property
    def _zip(self):
        """Zip handle opened per process: DataLoader workers must not share one."""
        if not self.is_zip:
            return None
        if self._zip_handle is None or self._zip_pid != os.getpid():
            self._zip_handle = zipfile.ZipFile(self.source)
            self._zip_pid = os.getpid()
        return self._zip_handle

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zip_handle"] = None
        state["_zip_pid"] = None
        return state

    def _find_root(self) -> str:
        if self.is_zip:
            for name in self._zip.namelist():
                if name.endswith("ImageSets/Main/test.txt"):
                    return name[: -len("ImageSets/Main/test.txt")]
        else:
            for p in [self.source, *self.source.iterdir()]:
                if (p / "ImageSets" / "Main" / "test.txt").exists():
                    return str(p) + "/"
        raise FileNotFoundError("Could not find ImageSets/Main/test.txt in " + str(self.source))

    def read(self, rel: str) -> bytes:
        if self.is_zip:
            return self._zip.read(self.root + rel)
        return (Path(self.root) / rel).read_bytes()

    def split(self, name: str) -> List[Tuple[str, int]]:
        ids = self.read(f"ImageSets/Main/{name}.txt").decode().split()
        return [(i, _label_from_xml(self.read(f"Annotations/{i}.xml").decode())) for i in ids]

    def image_bytes(self, image_id: str) -> bytes:
        return self.read(f"JPEGImages/{image_id}.jpg")

    def extract_split(self, name: str, out_dir: str | Path) -> Dict[str, int]:
        """Write images into out_dir/<split>/<benign|malignant>/ (ImageFolder layout)."""
        counts = {l: 0 for l in LABELS}
        for image_id, label in self.split(name):
            dest = Path(out_dir) / name / LABELS[label]
            dest.mkdir(parents=True, exist_ok=True)
            (dest / f"{image_id}.jpg").write_bytes(self.image_bytes(image_id))
            counts[LABELS[label]] += 1
        return counts
