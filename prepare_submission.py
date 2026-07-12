#!/usr/bin/env python3
"""
FREUID Challenge 2026 — reference inference entrypoint (template).

Organizers mount:
  /data/           read-only   test images only (flat directory, no CSV)
  /submissions/    read-write  must contain submission.csv after exit

Image filenames define row ids: ``{id}.jpeg`` (``.jpg`` / ``.png`` / ``.webp`` also
accepted). The document id is the filename stem.

Output schema: ``id,label`` where ``label`` is a real-valued fraud score
(higher = more confident the document is fraudulent) — same semantics as on the Kaggle leaderboard.

This script must be run from the repository root so that package imports
(e.g. `from Model...`) resolve correctly.

Example:

    python -m prepare_submission

or, if using uv:

    uv run python -m prepare_submission

Do not execute this file from inside another directory unless the repository
root is on PYTHONPATH.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from Model.Config import config, data_config
from Model.model import Model

import torch
from torchvision import transforms
from torchvision.io import decode_image

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf=data_config()

model=Model(config)
state_dict=torch.load(
    "Model/best_model_optuna_15trials.pt",
    map_location=device,
    weights_only=False
)
model.load_state_dict(state_dict,strict=False)
model.to(device)
model.eval()

transform=transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((conf.image_dim,conf.image_dim)),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

DATA_DIR = Path(os.environ.get("FREUID_DATA_DIR", "/data"))
OUTPUT_DIR = Path(os.environ.get("FREUID_OUTPUT_DIR", "/submissions"))
SUBMISSION_PATH = Path(os.environ.get("FREUID_SUBMISSION_PATH", OUTPUT_DIR / "submission.csv"))

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate FREUID submission.csv from images in /data (template)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory of test images (default: $FREUID_DATA_DIR or /data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SUBMISSION_PATH,
        help="Output CSV path (default: $FREUID_SUBMISSION_PATH).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for deterministic placeholder scores (template only).",
    )
    return parser.parse_args()


def discover_images(data_dir: Path) -> list[tuple[str, Path]]:
    """Return (id, path) pairs for every image file directly under ``data_dir``."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pairs: list[tuple[str, Path]] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        row_id = path.stem
        if not row_id:
            raise ValueError(f"Cannot derive id from filename: {path.name}")
        pairs.append((row_id, path))

    if not pairs:
        raise FileNotFoundError(
            f"No images found in {data_dir}. Expected flat files like '{{id}}.jpeg'."
        )
    return pairs

def predict_labels(image_rows:list[tuple[str,Path]],seed:int)->pd.DataFrame:
    ids=[]
    labels=[]

    with torch.inference_mode():
        for row_id,image_path in image_rows:
            image=decode_image(str(image_path))
            image=transform(image).unsqueeze(0).to(device)

            logit=model(image).squeeze()
            score=torch.sigmoid(logit).item()

            ids.append(row_id)
            labels.append(score)

    return pd.DataFrame({
        "id":ids,
        "label":labels
    })


def validate_submission(submission: pd.DataFrame, expected_ids: set[str]) -> None:
    if list(submission.columns) != ["id", "label"]:
        raise ValueError(
            f"submission.csv must have columns ['id', 'label']; got {list(submission.columns)}"
        )

    got = set(submission["id"].astype(str))
    missing = expected_ids - got
    extra = got - expected_ids
    if missing:
        raise ValueError(f"submission.csv missing {len(missing)} id(s), e.g. {sorted(missing)[:3]}")
    if extra:
        raise ValueError(f"submission.csv has {len(extra)} unexpected id(s), e.g. {sorted(extra)[:3]}")


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_path = args.output.resolve()

    image_rows = discover_images(data_dir)
    expected_ids = {row_id for row_id, _ in image_rows}
    submission = predict_labels(image_rows, seed=args.seed)
    validate_submission(submission, expected_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} rows to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
