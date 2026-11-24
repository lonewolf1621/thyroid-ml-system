"""Live smoke test for the deployed Thyroid ML API (run by GitHub Actions).

Waits out a Render free-tier cold start, then checks /health, a normal and a
hypothyroid lab profile on every tabular model, batch CSV scoring, and one
image prediction with a Grad-CAM heatmap. Standard library only.

    python scripts/smoke_test.py https://thyroid-ml-system.onrender.com
"""
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import zlib

NORMAL = {"age": 45, "sex": "F", "TSH": 1.3, "T3": 2.0, "TT4": 110, "T4U": 1.0, "FTI": 110}
HYPO = {"age": 55, "sex": "F", "TSH": 60, "T3": 0.5, "TT4": 40, "T4U": 1.0, "FTI": 40}


def grey_png(w=224, h=224, v=90):
    raw = b"".join(b"\x00" + bytes([v]) * w for _ in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def call(url, json_body=None, file=None, timeout=120):
    headers = {"User-Agent": "thyroid-smoke-test"}
    data = None
    if json_body is not None:
        data, headers["Content-Type"] = json.dumps(json_body).encode(), "application/json"
    elif file is not None:
        name, content, ctype = file
        b = uuid.uuid4().hex
        data = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
                f"Content-Type: {ctype}\r\n\r\n").encode() + content + f"\r\n--{b}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={b}"
    req = urllib.request.Request(url, data=data, headers=headers)
    start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - start


def main():
    base = (sys.argv[1] if len(sys.argv) > 1 else "https://thyroid-ml-system.onrender.com").rstrip("/")
    start, last = time.time(), None
    while True:
        try:
            health, _ = call(f"{base}/health")
            if health.get("status") == "ok":
                break
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            last = e
        if time.time() - start > 300:
            raise SystemExit(f"FAIL /health not ok after 300s (last error: {last})")
        time.sleep(10)
    print(f"OK   /health after {time.time() - start:.0f}s: {health}")

    problems = []
    for model in health.get("tabular_models", []):
        n, t1 = call(f"{base}/predict?model_name={model}", NORMAL)
        h, t2 = call(f"{base}/predict?model_name={model}", HYPO)
        ok = n["prediction"] == "negative" and h["prediction"] == "hypothyroid"
        print(f"{'OK  ' if ok else 'FAIL'} {model}: normal={n['probability_hypothyroid']} "
              f"hypo={h['probability_hypothyroid']} ({max(t1, t2):.2f}s)")
        if not ok:
            problems.append(f"{model} gave wrong labels")
    if len(health.get("tabular_models", [])) < 3:
        problems.append(f"expected 3 tabular models, got {health.get('tabular_models')}")

    csv = ("age,sex,TSH,T3,TT4,T4U,FTI\n45,F,1.3,2.0,110,1.0,110\n55,F,60,0.5,40,1.0,40\n").encode()
    b, secs = call(f"{base}/predict/batch", file=("p.csv", csv, "text/csv"))
    labels = [r["prediction"] for r in b.get("results", [])]
    print(f"{'OK  ' if labels == ['negative', 'hypothyroid'] else 'FAIL'} /predict/batch: {labels} ({secs:.2f}s)")
    if labels != ["negative", "hypothyroid"]:
        problems.append(f"batch labels {labels}")

    img, secs = call(f"{base}/predict-image?explain=true", file=("scan.png", grey_png(), "image/png"), timeout=180)
    ok = img.get("prediction") in {"benign", "malignant"} and len(img.get("gradcam_png_base64", "")) > 100
    print(f"{'OK  ' if ok else 'FAIL'} /predict-image: {img.get('prediction')} {img.get('confidence')} "
          f"gradcam={'yes' if img.get('gradcam_png_base64') else 'no'} ({secs:.1f}s)")
    if not ok:
        problems.append("image prediction or Grad-CAM missing")

    if problems:
        raise SystemExit("FAIL " + "; ".join(problems))


if __name__ == "__main__":
    main()
