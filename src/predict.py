"""
Loads the trained model and runs a prediction on a single MRI image.
Generates a simple AI preliminary report.
"""

import sys
from pathlib import Path

import joblib

from load_data import preprocess_single_image, CLASS_MAP
from report_generator import generate_report


def main(image_path: str):
    # Loading the trained model
    model_path = Path("models/brain_tumor_rf.joblib")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Train the model first by running: python train_model.py"
        )

    clf = joblib.load(model_path)

    # Preprocess the image (same pipeline as training)
    X = preprocess_single_image(image_path)

    # Run prediction
    pred_label = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]

    inv_class_map = {v: k for k, v in CLASS_MAP.items()}
    predicted_class = inv_class_map[pred_label]
    confidence = float(proba[pred_label])

    print(f"[INFO] Image: {image_path}")
    print(f"[INFO] Predicted class: {predicted_class}")
    print(f"[INFO] Confidence: {confidence:.2f}")

    # 4) Generate text report
    generate_report(image_path, predicted_class, confidence)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.png")
        sys.exit(1)

    img_path = sys.argv[1]
    main(img_path)
