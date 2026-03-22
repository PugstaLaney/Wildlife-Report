"""
Wildlife Camera Trap Detection Pipeline
========================================
Uses Microsoft MegaDetector v5 (via PytorchWildlife) to detect animals,
people, and vehicles in trail camera images.

Outputs:
  - detections.csv  : one row per detection with metadata
  - annotated/      : copies of images with bounding boxes drawn
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ExifTags

from PytorchWildlife.models import detection as pw_detection


# Configuration 

IMAGE_DIR  = Path(r"G:\My Drive\Wildlife pictures")
OUTPUT_DIR = Path(r"C:\Users\palla\OneDrive\Documents\Coding Projects\Wildlife report\Annotated_Megadetector_Output")
CSV_PATH   = OUTPUT_DIR / "detections.csv"

# Detection confidence threshold (0.0–1.0).
# Lower = catches more animals but more false positives.
# 0.2 is a good starting point for trail cameras.
CONFIDENCE = 0.2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# MegaDetector v5 class IDs (0-indexed in supervision library)
LABEL_MAP = {0: "animal", 1: "person", 2: "vehicle"}
BOX_COLORS = {"animal": "lime", "person": "red", "vehicle": "yellow"}


# Path Setup 

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")
print("Loading MegaDetector v5 (downloads ~600 MB on first run)...")

model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True)
print("Model ready.\n")


#Helpers

def get_timestamp(image_path: Path) -> str:
    """Return EXIF capture time, or file modification time as fallback."""
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if exif:
            for tag_id, value in exif.items():
                if ExifTags.TAGS.get(tag_id) == "DateTimeOriginal":
                    return str(value)  # "YYYY:MM:DD HH:MM:SS"
    except Exception:
        pass
    mtime = os.path.getmtime(image_path)
    return datetime.fromtimestamp(mtime).strftime("%Y:%m:%d %H:%M:%S")


def draw_boxes(img: Image.Image, detections, output_path: Path):
    """Draw bounding boxes + labels on image and save to output_path."""
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", size=max(16, img.width // 60))
    except Exception:
        font = ImageFont.load_default()

    for box, conf, cls_id in zip(
        detections.xyxy, detections.confidence, detections.class_id
    ):
        label = LABEL_MAP.get(int(cls_id), "unknown")
        color = BOX_COLORS.get(label, "white")
        x1, y1, x2, y2 = map(int, box)

        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label background + text
        text = f"{label}  {conf:.2f}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1 - text_h - 4, x1 + (text_bbox[2] - text_bbox[0]) + 4, y1],
                       fill=color)
        draw.text((x1 + 2, y1 - text_h - 2), text, fill="black", font=font)

    img.save(output_path)


#Main pipeline

def run():
    # Collect all images, skipping the annotated output folder
    image_files = sorted([
        p for p in IMAGE_DIR.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
        and OUTPUT_DIR not in p.parents
        and p.parent != OUTPUT_DIR
    ])

    if not image_files:
        print(f"No images found in: {IMAGE_DIR}")
        return

    print(f"Found {len(image_files)} image(s). Starting detection...\n")

    rows = []

    for i, img_path in enumerate(image_files, 1):
        print(f"[{i:>4}/{len(image_files)}] {img_path.name}", end="  ")

        try:
            pil_img   = Image.open(img_path).convert("RGB")
            img_array = np.array(pil_img)
            timestamp = get_timestamp(img_path)

            result     = model.single_image_detection(img_array, img_path=str(img_path),
                                                      det_conf_thres=CONFIDENCE)
            detections = result["detections"]
            n          = len(detections)
            print(f"→ {n} detection(s)")

            if n == 0:
                rows.append({
                    "filename"        : img_path.name,
                    "filepath"        : str(img_path),
                    "timestamp"       : timestamp,
                    "detection_count" : 0,
                    "label"           : "none",
                    "confidence"      : None,
                    "x1": None, "y1": None, "x2": None, "y2": None,
                })
            else:
                # Save annotated image
                ann_path = OUTPUT_DIR / img_path.name
                draw_boxes(pil_img.copy(), detections, ann_path)

                for box, conf, cls_id in zip(
                    detections.xyxy, detections.confidence, detections.class_id
                ):
                    label = LABEL_MAP.get(int(cls_id), "unknown")
                    x1, y1, x2, y2 = map(int, box)
                    rows.append({
                        "filename"        : img_path.name,
                        "filepath"        : str(img_path),
                        "timestamp"       : timestamp,
                        "detection_count" : n,
                        "label"           : label,
                        "confidence"      : round(float(conf), 4),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    })

        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append({
                "filename"        : img_path.name,
                "filepath"        : str(img_path),
                "timestamp"       : "",
                "detection_count" : -1,
                "label"           : f"ERROR: {e}",
                "confidence"      : None,
                "x1": None, "y1": None, "x2": None, "y2": None,
            })

    # Save CSV
    df = pd.DataFrame(rows, columns=[
        "filename", "filepath", "timestamp",
        "detection_count", "label", "confidence",
        "x1", "y1", "x2", "y2"
    ])
    df.to_csv(CSV_PATH, index=False)

    # Summary
    print("\n" + "=" * 50)
    print(f"Done!")
    print(f"  CSV saved   : {CSV_PATH}")
    print(f"  Annotated   : {OUTPUT_DIR}")
    print(f"\nDetection summary:")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    run()
