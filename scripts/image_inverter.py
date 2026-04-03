#!/usr/bin/env python3
"""
Watch an input folder for new images and save inverted-color copies
to an output folder.

Requires:
    pip install pillow

Usage:
    python invert_watcher.py /path/to/input /path/to/output

Optional:
    python invert_watcher.py /path/to/input /path/to/output --interval 5
    python invert_watcher.py /path/to/input /path/to/output --recursive
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Set

from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def invert_image(input_path: Path, output_path: Path) -> None:
    """
    Open an image, invert its colors, and save it to output_path.
    Handles RGB, RGBA, grayscale, and palette images.
    """
    with Image.open(input_path) as img:
        img.load()

        if img.mode == "RGBA":
            r, g, b, a = img.split()
            rgb = Image.merge("RGB", (r, g, b))
            inverted_rgb = ImageOps.invert(rgb)
            result = Image.merge("RGBA", (*inverted_rgb.split(), a))

        elif img.mode == "LA":
            l, a = img.split()
            inverted_l = ImageOps.invert(l)
            result = Image.merge("LA", (inverted_l, a))

        else:
            # Convert palette and other uncommon modes to RGB if needed
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            result = ImageOps.invert(img)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)


def build_output_path(input_file: Path, input_dir: Path, output_dir: Path) -> Path:
    """
    Mirror the input folder structure inside the output folder.
    Example:
        input_dir/sub/a.png -> output_dir/sub/a.png
    """
    relative = input_file.relative_to(input_dir)
    return output_dir / relative


def scan_files(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        return [p for p in input_dir.rglob("*") if is_image_file(p)]
    return [p for p in input_dir.iterdir() if is_image_file(p)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuously watch a folder for new images and save inverted copies."
    )
    parser.add_argument("input_dir", type=Path, help="Folder to watch for new images")
    parser.add_argument("output_dir", type=Path, help="Folder where inverted images are saved")
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Watch subfolders recursively",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logging.info("Watching input folder: %s", input_dir)
    logging.info("Saving inverted images to: %s", output_dir)
    logging.info("Polling every %.2f seconds", args.interval)

    # Track files already seen so only newly added files are processed.
    seen_files: Set[Path] = set(scan_files(input_dir, args.recursive))
    logging.info("Found %d existing image(s); only new files will be processed.", len(seen_files))

    try:
        while True:
            current_files = set(scan_files(input_dir, args.recursive))
            new_files = sorted(current_files - seen_files)

            for input_file in new_files:
                output_file = build_output_path(input_file, input_dir, output_dir)

                try:
                    invert_image(input_file, output_file)
                    logging.info("Processed: %s -> %s", input_file.name, output_file)
                except Exception as exc:
                    logging.error("Failed to process %s: %s", input_file, exc)

            seen_files = current_files
            time.sleep(args.interval)

    except KeyboardInterrupt:
        logging.info("Stopped by user.")


if __name__ == "__main__":
    main()