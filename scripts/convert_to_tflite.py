"""
Convert SSD MobileNet V2 FPNLite SavedModel to TFLite.

Usage (from repo root):
    python scripts/convert_to_tflite.py

The script:
  1. Extracts the archive into a temp dir
  2. Converts with DEFAULT + SELECT_TF_OPS (required for tensor-list ops)
  3. Writes models/ssd_mobilenet_v2_fpnlite_320x320.tflite
"""

import os
import sys
import tarfile
import tempfile
import pathlib

import tensorflow as tf

REPO_ROOT   = pathlib.Path(__file__).resolve().parent.parent
ARCHIVE     = REPO_ROOT / "models" / "ssd-mobilenet-v2-tensorflow2-fpnlite-320x320-v1.tar.gz"
OUTPUT_PATH = REPO_ROOT / "models" / "ssd_mobilenet_v2_fpnlite_320x320.tflite"


def extract_saved_model(archive_path: pathlib.Path, dest_dir: pathlib.Path) -> pathlib.Path:
    print(f"Extracting {archive_path.name} …")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest_dir)

    # The archive root is the SavedModel itself — confirm required files exist.
    if not (dest_dir / "saved_model.pb").exists():
        raise FileNotFoundError(
            f"saved_model.pb not found in {dest_dir}. "
            "Archive may have an unexpected directory structure."
        )
    return dest_dir


def convert(saved_model_dir: pathlib.Path, output_path: pathlib.Path) -> None:
    print(f"Loading SavedModel from {saved_model_dir} …")
    converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # SELECT_TF_OPS
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    # Disable tensor list lowering
    converter._experimental_lower_tensor_list_ops = False

    print("Converting …")
    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    print(f"Saved: {output_path}")


def main() -> None:
    if not ARCHIVE.exists():
        sys.exit(f"Archive not found: {ARCHIVE}")

    # Always convert

    # Extract into a stable temp dir next to the archive so we can inspect it
    # if something goes wrong, instead of using TemporaryDirectory.
    extract_target = REPO_ROOT / "models" / "_saved_model_tmp"
    extract_target.mkdir(exist_ok=True)

    try:
        saved_model_dir = extract_saved_model(ARCHIVE, extract_target)
        convert(saved_model_dir, OUTPUT_PATH)
    finally:
        # Clean up the extraction dir to avoid leaving large files around.
        import shutil
        if extract_target.exists():
            shutil.rmtree(extract_target, ignore_errors=True)
            print("Cleaned up temporary extraction directory.")


if __name__ == "__main__":
    main()
