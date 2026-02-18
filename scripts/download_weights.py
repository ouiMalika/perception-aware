#!/usr/bin/env python3
"""
Download pre-trained YOLOv7 weights for traffic sign detection.

Supports two sources:

  1. Direct URL (fastest – any public .pt file):

       python scripts/download_weights.py \\
         --source url \\
         --url https://example.com/yolov7_lisa.pt \\
         --output models/yolov7_lisa.pt

  2. Roboflow Universe (requires a free API key from roboflow.com):

       python scripts/download_weights.py \\
         --source roboflow \\
         --api-key YOUR_KEY \\
         --workspace dakota-smith \\
         --project lisa-road-signs \\
         --version 2 \\
         --output models/

After downloading, point the worker at the weights:

  export YOLOV7_WEIGHTS_PATH=/absolute/path/to/weights.pt

Or set YOLOV7_WEIGHTS_URL in your .env so the worker downloads on startup.

Suggested Roboflow LISA models (free, no registration gate):
  - workspace=dakota-smith  project=lisa-road-signs  version=2   (4 classes)
"""

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download_url(url: str, dest: Path) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:.1f}%  ({downloaded:,} / {total:,} bytes)", end="", flush=True)

    print(f"\nSaved → {dest}  ({downloaded:,} bytes)")


def _download_roboflow(api_key: str, workspace: str, project: str, version: int, output: Path) -> Path:
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit(
            "roboflow package not installed.\n"
            "Run:  pip install roboflow\n"
            "Then retry."
        )

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov7", location=str(output))
    print(f"\nDataset downloaded to: {dataset.location}")

    # Locate the weights file if the version has a trained model
    pt_files = list(Path(dataset.location).rglob("*.pt"))
    if pt_files:
        pt = pt_files[0]
        print(f"Weights file: {pt}")
        return pt

    print(
        "\nNote: this Roboflow version may not have exported weights.\n"
        "You can train on the downloaded dataset, or use --source url with\n"
        "a direct link to a .pt file instead."
    )
    return Path(dataset.location)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download YOLOv7 traffic sign weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        choices=["url", "roboflow"],
        required=True,
        help="Where to get the weights from",
    )

    # URL source
    parser.add_argument("--url", help="Direct URL to a .pt file (--source url)")

    # Roboflow source
    parser.add_argument("--api-key", help="Roboflow API key (--source roboflow)")
    parser.add_argument("--workspace", default="dakota-smith", help="Roboflow workspace slug")
    parser.add_argument("--project", default="lisa-road-signs", help="Roboflow project slug")
    parser.add_argument("--version", type=int, default=2, help="Roboflow dataset version")

    # Common
    parser.add_argument(
        "--output",
        default="models/",
        help="Output path (.pt file or directory, default: models/)",
    )

    args = parser.parse_args()
    output = Path(args.output)

    if args.source == "url":
        if not args.url:
            parser.error("--url is required when --source url")

        # If output is a directory, derive filename from URL
        if output.suffix != ".pt":
            raw = args.url.split("/")[-1].split("?")[0]
            filename = raw if raw.endswith(".pt") else (raw or "weights") + ".pt"
            output = output / filename

        _download_url(args.url, output)
        pt = output

    else:  # roboflow
        if not args.api_key:
            parser.error("--api-key is required when --source roboflow")
        pt = _download_roboflow(args.api_key, args.workspace, args.project, args.version, output)

    print(
        f"\nDone. To use these weights, add to your .env:\n\n"
        f"  YOLOV7_WEIGHTS_PATH={pt.resolve()}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
