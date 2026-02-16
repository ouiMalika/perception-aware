#!/usr/bin/env python3
"""
CLI tool for testing traffic sign detection locally.

Usage:
    # Detect signs in a local image
    python scripts/detect_cli.py --image path/to/image.jpg

    # Detect signs from a URL
    python scripts/detect_cli.py --url https://example.com/road.jpg

    # Save annotated output
    python scripts/detect_cli.py --image photo.jpg --output result.jpg

    # Adjust confidence thresholds
    python scripts/detect_cli.py --image photo.jpg --yolo-conf 0.3 --clip-conf 0.2

    # List all supported sign categories
    python scripts/detect_cli.py --list-categories
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add worker directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "worker"))


def list_categories():
    from sign_taxonomy import SIGN_TAXONOMY
    print(f"\nSupported Sign Categories ({len(SIGN_TAXONOMY)} total)")
    print("=" * 70)

    by_priority = {}
    for cat, info in SIGN_TAXONOMY.items():
        pri = info["priority"]
        by_priority.setdefault(pri, []).append((cat, info))

    priority_labels = {
        1: "CRITICAL (Priority 1)",
        2: "WARNING (Priority 2)",
        3: "GUIDE (Priority 3)",
        4: "INFO (Priority 4)",
    }

    for pri in sorted(by_priority.keys()):
        print(f"\n  {priority_labels[pri]}")
        print(f"  {'-' * 40}")
        for cat, info in sorted(by_priority[pri]):
            print(f"    {cat:<25} {info['description']}")
            print(f"    {'':25} Prompts: {len(info['prompts'])}")


def detect_image(args):
    from detector import process_image_path, process_image_url

    print("\nLoading models (this may take a minute on first run)...")
    start = time.time()

    if args.url:
        print(f"Processing URL: {args.url}")
        result = process_image_url(
            args.url,
            yolo_conf=args.yolo_conf,
            clip_conf=args.clip_conf,
        )
    else:
        print(f"Processing file: {args.image}")
        result = process_image_path(
            args.image,
            yolo_conf=args.yolo_conf,
            clip_conf=args.clip_conf,
        )

    elapsed = time.time() - start

    if result.error:
        print(f"\nERROR: {result.error}")
        return 1

    print(f"\nImage: {result.image_width}x{result.image_height}")
    print(f"Processing time: {elapsed:.1f}s")
    print(f"Detections: {len(result.detections)}")

    if result.detections:
        print(f"\n{'Category':<25} {'Confidence':>10} {'CLIP':>8} {'YOLO':>8} {'Location'}")
        print("-" * 80)
        for det in result.detections:
            loc = f"({det.bbox.x1:.0f},{det.bbox.y1:.0f})-({det.bbox.x2:.0f},{det.bbox.y2:.0f})"
            print(
                f"  {det.category:<23} {det.confidence:>9.1%} "
                f"{det.clip_confidence:>7.1%} {det.yolo_confidence:>7.1%} {loc}"
            )
            if args.verbose:
                for cat, score in det.clip_scores.items():
                    print(f"    {'':23} -> {cat}: {score:.3f}")
    else:
        print("\nNo traffic signs detected.")

    # Save annotated image
    if args.output:
        from PIL import Image

        from visualizer import save_annotated

        if args.url:
            from detector import load_image_from_url
            image = load_image_from_url(args.url)
        else:
            image = Image.open(args.image).convert("RGB")

        out_path = save_annotated(image, result.detections, args.output)
        print(f"\nAnnotated image saved to: {out_path}")

    # Save JSON results
    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"JSON results saved to: {args.json_output}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Sign Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--image", "-i", help="Path to a local image file")
    parser.add_argument("--url", "-u", help="URL of an image to process")
    parser.add_argument("--output", "-o", help="Path to save annotated image")
    parser.add_argument("--json-output", "-j", help="Path to save JSON results")
    parser.add_argument(
        "--yolo-conf", type=float, default=0.25,
        help="YOLO detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--clip-conf", type=float, default=0.15,
        help="CLIP classification confidence threshold (default: 0.15)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed scores")
    parser.add_argument("--list-categories", action="store_true", help="List all sign categories")

    args = parser.parse_args()

    if args.list_categories:
        list_categories()
        return 0

    if not args.image and not args.url:
        parser.error("Provide --image or --url (or use --list-categories)")

    return detect_image(args)


if __name__ == "__main__":
    sys.exit(main())
