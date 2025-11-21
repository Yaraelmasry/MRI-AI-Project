import streamlit as st
from src.load_data import preprocess_single_image
from src.report_generator import generate_report
from src.load_data import CLASS_MAP
import joblib
from pathlib import Path

st.title("MRI Brain Tumor AI Assistant")

# Load model
model_path = Path("models/brain_tumor_rf.joblib")
clf = joblib.load(model_path)
inv_class_map = {v: k for k, v in CLASS_MAP.items()}

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)
    
    # Save to temp file
    temp_path = "temp_uploaded_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess
    X = preprocess_single_image(temp_path)

    # Predict
    pred_label = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0][pred_label]
    predicted_class = inv_class_map[pred_label]

    st.subheader("AI Prediction")
    st.write(f"**Tumor type:** {predicted_class}")
    st.write(f"**Confidence:** {proba:.2f}")

    # Generate report
    report_path = generate_report(temp_path, predicted_class, proba)

# radiologist review 
st.subheader("Radiologist Review")

# Radiologist Notes
notes = st.text_area(
    "Radiologist Notes (optional):",
    placeholder="Add additional observations or clarifications here..."
)

# Reject Reasons dropdown (only relevant if Reject Report is chosen)
reject_reason = None

choice = st.radio(
    "Select Radiologist Action:",
    ("Approve Report", "Reject Report")
)

if choice == "Reject Report":
    reject_reason = st.selectbox(
        "Reason for Rejection:",
        [
            "Poor image quality",
            "Wrong modality/body part",
            "Model misclassification",
            "Insufficient clinical value",
            "Other"
        ]
    )

if st.button("Submit Decision"):
    validated_dir = Path("reports/validated")
    rejected_dir = Path("reports/rejected")
    validated_dir.mkdir(exist_ok=True)
    rejected_dir.mkdir(exist_ok=True)

    original_report = Path(report_path)

    # If radiologist approves
    if choice == "Approve Report":
        validated_report = validated_dir / f"validated_{Path(temp_path).stem}.txt"

        with open(original_report, "r") as f:
            content = f.read()

        validated_content = (
            "=== RADIOLOGIST-APPROVED REPORT ===\n\n"
            + content
            + "\nDecision: APPROVED\n"
            + f"Radiologist Notes: {notes if notes else 'None'}\n"
        )

        with open(validated_report, "w") as f:
            f.write(validated_content)

        st.success(f"Report approved and saved to: {validated_report}")

    # If radiologists rejects
    else:
        rejected_report = rejected_dir / f"rejected_{Path(temp_path).stem}.txt"

        with open(original_report, "r") as f:
            content = f.read()

        rejected_content = (
            "=== RADIOLOGIST-REJECTED REPORT ===\n\n"
            + content
            + f"\nDecision: REJECTED\nReason: {reject_reason}\n"
            + f"Radiologist Notes: {notes if notes else 'None'}\n"
        )

        with open(rejected_report, "w") as f:
            f.write(rejected_content)

        st.error(f"Report rejected and saved to: {rejected_report}")
