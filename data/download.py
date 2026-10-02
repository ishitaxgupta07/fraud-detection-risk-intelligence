#!/usr/bin/env python3
"""
Data Download Helper

Downloads datasets from Kaggle using the Kaggle API.
Requires: pip install kaggle
Setup: Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY env vars.

Usage:
    python data/download.py --dataset creditcard
    python data/download.py --dataset paysim
    python data/download.py --dataset all
"""

import argparse
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path("data")

DATASETS = {
    "creditcard": {
        "slug": "mlg-ulb/creditcardfraud",
        "url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "expected_file": "creditcard.csv",
        "description": "Credit Card Fraud Detection (284K transactions)",
    },
    "paysim": {
        "slug": "ealaxi/paysim1",
        "url": "https://www.kaggle.com/datasets/ealaxi/paysim1",
        "expected_file": "PS_20174392719_1491204439457_log.csv",
        "description": "PaySim Mobile Money Fraud (6.3M transactions)",
    },
}


def check_kaggle():
    try:
        import kaggle  # noqa: F401

        return True
    except ImportError:
        print("[ERROR] kaggle package not found. Install with: pip install kaggle")
        print("[INFO]  Then set up credentials: https://www.kaggle.com/docs/api")
        return False


def download_dataset(name: str):
    info = DATASETS[name]
    expected = DATA_DIR / info["expected_file"]

    if expected.exists():
        print(f"[SKIP] {name}: {expected} already exists.")
        return True

    print(f"[INFO] Downloading {info['description']}...")
    print(f"[INFO] Source: {info['url']}")

    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        info["slug"],
        "-p",
        str(DATA_DIR),
        "--unzip",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[OK] {name} downloaded successfully.")
        return True
    else:
        print(f"[ERROR] Failed to download {name}:")
        print(result.stderr)
        print(f"[INFO] Manual download: {info['url']}")
        print(f"[INFO] Place file at: {expected}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download fraud detection datasets")
    parser.add_argument("--dataset", choices=["creditcard", "paysim", "all"], default="all", help="Dataset to download")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    if not check_kaggle():
        print("\n[INFO] Synthetic data will be used automatically during training.")
        return

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for name in targets:
        download_dataset(name)

    print("\n[DONE] Dataset setup complete.")
    print("[INFO] Run training: python train.py --dataset creditcard")


if __name__ == "__main__":
    main()
