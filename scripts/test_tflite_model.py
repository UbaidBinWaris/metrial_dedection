"""
Validate the converted SSD MobileNet V2 FPNLite TFLite model.

Usage (from repo root):
    python scripts/test_tflite_model.py

Prints input/output tensor shapes and runs a dummy inference to confirm
the model is loadable and produces the expected output structure.
"""

import pathlib
import sys

import numpy as np
import tensorflow as tf

REPO_ROOT   = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH  = REPO_ROOT / "models" / "ssd_mobilenet_v2_fpnlite_320x320.tflite"


def load_and_inspect(model_path: pathlib.Path) -> tf.lite.Interpreter:
    if not model_path.exists():
        sys.exit(
            f"Model not found: {model_path}\n"
            "Run  python scripts/convert_to_tflite.py  first."
        )

    print(f"Loading {model_path.name}  ({model_path.stat().st_size / (1024**2):.1f} MB)\n")

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def print_tensors(interpreter: tf.lite.Interpreter) -> dict:
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("─── INPUT TENSORS ───────────────────────────────")
    for t in input_details:
        print(f"  [{t['index']}] {t['name']}")
        print(f"       shape : {t['shape'].tolist()}")
        print(f"       dtype : {t['dtype'].__name__}")

    print("\n─── OUTPUT TENSORS ──────────────────────────────")
    label_hints = {
        0: "detection_boxes   (y1,x1,y2,x2 normalized)",
        1: "detection_classes (0-indexed class IDs)",
        2: "detection_scores  (confidence 0–1)",
        3: "num_detections    (valid detection count)",
    }
    for i, t in enumerate(output_details):
        hint = label_hints.get(i, "")
        print(f"  [{t['index']}] {t['name']}")
        print(f"       shape : {t['shape'].tolist()}  {hint}")
        print(f"       dtype : {t['dtype'].__name__}")

    return {"inputs": input_details, "outputs": output_details}


def run_dummy_inference(interpreter: tf.lite.Interpreter, details: dict) -> None:
    inp = details["inputs"][0]
    shape = inp["shape"]          # [1, 320, 320, 3]
    dtype = inp["dtype"]

    # Pixel values in [0, 255] for uint8 models or [-1, 1] / [0, 1] for float32.
    if dtype == np.float32:
        dummy = np.random.rand(*shape).astype(np.float32)
    elif dtype == np.uint8:
        dummy = np.random.randint(0, 256, size=shape, dtype=np.uint8)
    else:
        dummy = np.zeros(shape, dtype=dtype)

    interpreter.set_tensor(inp["index"], dummy)
    interpreter.invoke()

    print("\n─── DUMMY INFERENCE RESULTS ─────────────────────")
    output_names = ["boxes", "classes", "scores", "num_detections"]
    for i, t in enumerate(details["outputs"]):
        result = interpreter.get_tensor(t["index"])
        name = output_names[i] if i < len(output_names) else t["name"]
        print(f"  {name}:")
        print(f"    shape   : {result.shape}")
        print(f"    dtype   : {result.dtype}")
        if result.size > 0:
            print(f"    min/max : {result.min():.4f} / {result.max():.4f}")

    print("\nInference completed successfully.")


def main() -> None:
    interpreter = load_and_inspect(MODEL_PATH)
    details     = print_tensors(interpreter)

    input_shape = details["inputs"][0]["shape"].tolist()
    if input_shape != [1, 320, 320, 3]:
        print(
            f"\nWARNING: Expected input shape [1, 320, 320, 3], got {input_shape}.\n"
            "The model may still work but preprocessing must match this shape."
        )
    else:
        print(f"\nInput shape confirmed: {input_shape}")

    run_dummy_inference(interpreter, details)


if __name__ == "__main__":
    main()
