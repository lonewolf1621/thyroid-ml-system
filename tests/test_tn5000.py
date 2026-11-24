import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tn5000 import _label_from_xml  # noqa: E402


def test_labels_from_voc_xml():
    assert _label_from_xml("<object><name>0</name></object>") == 0
    assert _label_from_xml("<object><name>1</name></object>") == 1
    # any malignant nodule makes the image malignant
    assert _label_from_xml("<object><name>0</name></object><object><name>1</name></object>") == 1
