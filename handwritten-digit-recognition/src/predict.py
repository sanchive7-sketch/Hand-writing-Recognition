"""Load trained models and predict a preprocessed handwritten digit."""

from collections import Counter
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
def model_files(task="digit"):
    if task not in ("digit", "letter"):
        raise ValueError("task must be 'digit' or 'letter'")
    prefix = "" if task == "digit" else "letter_"
    return {
        "MLP": ROOT / "models" / f"{prefix}mlp_model.keras",
        "SVM": ROOT / "models" / f"{prefix}svm_model.pkl",
        "CNN": ROOT / "models" / f"{prefix}cnn_model.keras",
    }


def load_models(task="digit"):
    """Return (available models, missing names); call once and reuse."""
    models = {}
    missing = []
    for name, path in model_files(task).items():
        if not path.is_file():
            missing.append(name)
        elif name == "SVM":
            # Load only files produced by this project; pickle is not safe for
            # untrusted model files.
            models[name] = joblib.load(path)
        else:
            import tensorflow as tf
            models[name] = tf.keras.models.load_model(path)
    return models, missing


def predict_all(processed, models, task="digit"):
    """Use arrays returned by preprocess_custom_image."""
    predictions = {}
    for name, model in models.items():
        key = name.lower()
        if name == "SVM":
            probabilities = model.predict_proba(processed[key])[0]
            digit = int(model.classes_[int(np.argmax(probabilities))])
        else:
            probabilities = model.predict(processed[key], verbose=0)[0]
            digit = int(np.argmax(probabilities))
        predictions[name] = {
            "prediction": chr(ord("A") + digit) if task == "letter" else digit,
            "confidence": float(np.max(probabilities)),
        }
    return predictions


def majority_vote(predictions):
    """Break a tie with the highest summed model confidence."""
    if not predictions:
        return None
    votes = Counter(result["prediction"] for result in predictions.values())
    most_votes = max(votes.values())
    tied = [digit for digit, count in votes.items() if count == most_votes]
    return max(tied, key=lambda digit: sum(
        result["confidence"] for result in predictions.values()
        if result["prediction"] == digit
    ))
