"""
iNaturalist Species Classification Pipeline
============================================
Takes detections from MegaDetector (detections.csv), crops each animal
bounding box, and sends it to the iNaturalist Computer Vision API for
species identification.

Inputs:
  - detections.csv       : output from megadetector_pipeline.py
  - Original images      : sourced from filepath column in detections.csv

Outputs:
  - species_results.csv        : one row per detection with top species guesses
  - Annotated_iNaturalist_Output/ : images with bounding boxes labeled by species

Install dependencies first:
    pip install requests pillow pandas
"""

import time
import requests
import pandas as pd
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# ── Configuration 

WILDLIFE_REPORT_DIR = Path(r"C:\Users\palla\OneDrive\Documents\Coding Projects\Wildlife report")
DETECTIONS_CSV      = WILDLIFE_REPORT_DIR / "Annotated_Megadetector_Output" / "detections.csv"
OUTPUT_DIR          = WILDLIFE_REPORT_DIR / "Annotated_iNaturalist_Output"
OUTPUT_CSV          = OUTPUT_DIR / "species_results.csv"

# Your iNaturalist API token.
# Get it by logging into inaturalist.org, then visiting:
#   https://www.inaturalist.org/users/api_token
INAT_API_TOKEN = "eyJhbGciOiJIUzUxMiJ9.eyJ1c2VyX2lkIjoxMDIxODE0NCwiZXhwIjoxNzc0MjQ3OTM4fQ.KD2-VpUPNMvUarzwxBjDbfOIDbqr75tTNA_TBpp3N_U12ZZlTJ_PA_XCODl7E-F_JPu6ZesPnNi_wWpX8KcrfA"

# Optional: GPS coordinates of your camera trap location.
# Providing these improves accuracy by filtering out implausible species.
# Set to None to skip.
CAMERA_LAT = 31.880564254097866   # Grosvenor, Texas
CAMERA_LNG = -99.12752705480614   # Grosvenor, Texas

# Only classify detections at or above this MegaDetector confidence.
MIN_CONFIDENCE = 0.3

# Number of top species guesses to save per detection.
TOP_N = 3

# Seconds to wait between API calls (be respectful of rate limits).
REQUEST_DELAY = 1.5

INAT_CV_URL = "https://api.inaturalist.org/v1/computervision/score_image"


# ── Helpers 

def crop_detection(image_path: str, x1: int, y1: int, x2: int, y2: int) -> BytesIO:
    """Open image, crop to bounding box, return as in-memory JPEG bytes."""
    img = Image.open(image_path).convert("RGB")

    # Add small padding around the crop (5% of box size) so the animal
    # isn't cut right to the edge, which helps the classifier.
    pad_x = int((x2 - x1) * 0.05)
    pad_y = int((y2 - y1) * 0.05)
    w, h  = img.size

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop   = img.crop((x1, y1, x2, y2))
    buffer = BytesIO()
    crop.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    return buffer


def score_image(image_bytes: BytesIO, lat=None, lng=None) -> list[dict]:
    """
    Send image crop to iNaturalist CV API.
    Returns list of top species guesses: [{"name": ..., "score": ...}, ...]
    """
    headers = {"Authorization": f"Bearer {INAT_API_TOKEN}"}
    files   = {"image": ("crop.jpg", image_bytes, "image/jpeg")}
    data    = {}

    if lat is not None and lng is not None:
        data["lat"] = lat
        data["lng"] = lng

    response = requests.post(INAT_CV_URL, headers=headers, files=files, data=data, timeout=30)
    response.raise_for_status()

    results = response.json().get("results", [])
    guesses = []
    for r in results:
        taxon = r.get("taxon", {})
        guesses.append({
            "name"         : taxon.get("name", "unknown"),
            "common_name"  : taxon.get("preferred_common_name", ""),
            "rank"         : taxon.get("rank", ""),
            "score"        : round(r.get("combined_score", 0), 4),
        })
    return guesses


# ── Annotation 

def annotate_image(image_path: str, detections: list[dict], output_path: Path):
    """
    Draw bounding boxes on the full image labeled with top species guess.
    detections: list of dicts with keys x1, y1, x2, y2, label, score
    """
    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", size=max(16, img.width // 60))
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        label = det["label"]
        score = det["score"]

        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)

        text      = f"{label}  {score:.2f}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_h    = text_bbox[3] - text_bbox[1]
        text_w    = text_bbox[2] - text_bbox[0]
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill="lime")
        draw.text((x1 + 2, y1 - text_h - 2), text, fill="black", font=font)

    img.save(output_path)


# ── Main pipeline 

def run():
    if INAT_API_TOKEN == "PASTE_YOUR_TOKEN_HERE":
        print("ERROR: Set your iNaturalist API token in INAT_API_TOKEN before running.")
        return

    df = pd.read_csv(DETECTIONS_CSV)

    # Only process animal detections above confidence threshold
    animals = df[
        (df["label"] == "animal") &
        (df["confidence"] >= MIN_CONFIDENCE)
    ].copy()

    if animals.empty:
        print("No qualifying animal detections found in CSV.")
        return

    print(f"Found {len(animals)} animal detection(s) to classify.\n")

    rows            = []
    # Accumulate per-image detections for annotation: {filepath: [det, ...]}
    image_detections = {}

    for i, (_, row) in enumerate(animals.iterrows(), 1):
        print(f"[{i:>4}/{len(animals)}] {row['filename']}", end="  ")

        try:
            crop = crop_detection(
                row["filepath"],
                int(row["x1"]), int(row["y1"]),
                int(row["x2"]), int(row["y2"])
            )

            guesses = score_image(crop, lat=CAMERA_LAT, lng=CAMERA_LNG)
            top     = guesses[:TOP_N]

            top_label = top[0]["common_name"] or top[0]["name"] if top else "unknown"
            top_score = top[0]["score"] if top else 0.0
            print(f"→ {top_label}  ({top_score:.2f})" if top else "→ no results")

            base_row = {
                "filename"           : row["filename"],
                "filepath"           : row["filepath"],
                "timestamp"          : row["timestamp"],
                "megadetector_conf"  : row["confidence"],
                "x1": row["x1"], "y1": row["y1"],
                "x2": row["x2"], "y2": row["y2"],
            }

            for rank_idx, guess in enumerate(top, 1):
                base_row[f"species_{rank_idx}_name"]   = guess["name"]
                base_row[f"species_{rank_idx}_common"] = guess["common_name"]
                base_row[f"species_{rank_idx}_rank"]   = guess["rank"]
                base_row[f"species_{rank_idx}_score"]  = guess["score"]

            rows.append(base_row)

            # Queue this detection for image annotation
            fp = row["filepath"]
            if fp not in image_detections:
                image_detections[fp] = []
            image_detections[fp].append({
                "x1"   : int(row["x1"]), "y1": int(row["y1"]),
                "x2"   : int(row["x2"]), "y2": int(row["y2"]),
                "label": top_label,
                "score": top_score,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append({
                "filename"          : row["filename"],
                "filepath"          : row["filepath"],
                "timestamp"         : row["timestamp"],
                "megadetector_conf" : row["confidence"],
                "x1": row["x1"], "y1": row["y1"],
                "x2": row["x2"], "y2": row["y2"],
                "species_1_name"    : f"ERROR: {e}",
            })

        time.sleep(REQUEST_DELAY)

    # Save annotated images
    print(f"\nSaving annotated images...")
    for filepath, detections in image_detections.items():
        out_path = OUTPUT_DIR / Path(filepath).name
        try:
            annotate_image(filepath, detections, out_path)
        except Exception as e:
            print(f"  Could not annotate {Path(filepath).name}: {e}")

    # Save CSV
    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 50)
    print(f"Done!")
    print(f"  CSV saved     : {OUTPUT_CSV}")
    print(f"  Annotated     : {OUTPUT_DIR}")
    print(f"\nTop species identified:")
    if "species_1_common" in results_df.columns:
        print(results_df["species_1_common"].value_counts().head(10).to_string())


if __name__ == "__main__":
    run()
