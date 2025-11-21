"""
Trains a simple Random Forest classifier on the Kaggle brain MRI dataset.
"""

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

from load_data import load_dataset, CLASS_MAP


def main():
    # Load training and testing data
    X_train, y_train = load_dataset("data/Training")
    X_test, y_test = load_dataset("data/Testing")

    # Define the model
    # RandomForest is easy to train and works well on tabular features
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    #  Train the model
    print("[INFO] Training RandomForest model...")
    clf.fit(X_train, y_train)

    # Evaluate on test set
    print("[INFO] Evaluating on test set...")
    y_pred = clf.predict(X_test)

    # Build a list of class names in numeric order for the report
    inv_class_map = {v: k for k, v in CLASS_MAP.items()}
    target_names = [inv_class_map[i] for i in sorted(inv_class_map)]

    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Save the trained model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "brain_tumor_rf.joblib"
    joblib.dump(clf, model_path)
    print(f"[INFO] Model saved to: {model_path}")


if __name__ == "__main__":
    main()
