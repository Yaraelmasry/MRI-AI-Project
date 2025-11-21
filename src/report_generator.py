"""
Creates a simple text report for one MRI prediction.
"""

from pathlib import Path
from datetime import datetime


def generate_report(image_path: str, predicted_class: str, confidence: float):

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    image_name = Path(image_path).name
    stem = Path(image_path).stem

    report_text = f"""AI-ASSISTED MRI PRELIMINARY REPORT
=====================================

Timestamp: {datetime.now().isoformat(timespec='seconds')}

Input image: {image_name}

Predicted tumor type: {predicted_class}
Model confidence: {confidence:.2f}

Important:
- This is an AI-generated PRELIMINARY report.
- It is NOT a final diagnosis.
- A radiologist must review the images, confirm, edit, or reject these findings.

"""

    out_path = reports_dir / f"report_{stem}.txt"
    with open(out_path, "w") as f:
        f.write(report_text)

    print(f"[INFO] Report saved to: {out_path}")

    return str(out_path)
