"""Streamlit interface for comparing handwritten digits and letters."""

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.preprocessing import preprocess_custom_image
from src.predict import load_models, majority_vote, predict_all


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Handwriting Recognition", page_icon="🔤")
st.title("Handwriting Recognition")
st.write(
    "Compare a Backpropagation Neural Network, Support Vector Machine, and "
    "Convolutional Neural Network for single handwritten digits or letters."
)
task = st.radio("Recognize", ["Digit (0–9)", "Letter (A–Z)"], horizontal=True)
task_key = "digit" if task.startswith("Digit") else "letter"


@st.cache_resource
def cached_models(selected_task):
    return load_models(selected_task)


st.header("1. Digit Input")
st.write(f"Upload a clear image containing one {task_key}. Dark ink on a light "
         "background or light ink on a dark background both work.")
uploaded = st.file_uploader("Choose a digit image", type=["png", "jpg", "jpeg", "bmp"])

processed = None
if uploaded is not None:
    try:
        original = Image.open(uploaded)
        original.load()
        processed = preprocess_custom_image(original)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.warning(f"Could not process this image: {error}")

st.header("2. Image Preview")
if processed is not None:
    left, right = st.columns(2)
    with left:
        st.image(original, caption="Original input", width=200)
    with right:
        st.image(processed["image"], caption="Processed 28 × 28 image",
                 width=200, clamp=True)
else:
    st.info("Upload an image to see the original and processed versions.")

st.header("3. Prediction")
if st.button(f"Predict {task_key.title()}", disabled=processed is None):
    try:
        models, missing = cached_models(task_key)
        if missing:
            st.warning("Missing trained models: " + ", ".join(missing) +
                       ". Run their training commands to include them.")
        if models:
            predictions = predict_all(processed, models, task_key)
            columns = st.columns(len(predictions))
            for column, (name, result) in zip(columns, predictions.items()):
                column.metric(f"{name} Prediction", result["prediction"],
                              help=f"Confidence: {result['confidence']:.1%}")
                column.write(f"Confidence: {result['confidence']:.1%}")
            st.success(f"Final Prediction: {majority_vote(predictions)}")
        else:
            st.error("No trained models were found. Train a model before predicting.")
    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.header("4. Model Comparison")
metrics_path = ROOT / "results" / ("metrics.csv" if task_key == "digit" else "metrics_letters.csv")
if metrics_path.exists():
    try:
        metrics = pd.read_csv(metrics_path)
        columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
        st.dataframe(metrics[columns].style.format({key: "{:.2%}" for key in columns[1:]}),
                     hide_index=True, use_container_width=True)
        dataset_name = "MNIST" if task_key == "digit" else "EMNIST Letters"
        st.caption(f"Scores use the {dataset_name} test set; precision, recall, and F1 are macro averages.")
    except (OSError, ValueError, KeyError) as error:
        st.warning(f"The metrics file could not be read: {error}")
else:
    st.info("Run evaluation to create the model comparison table.")

st.header("5. Visual Analysis")
plots = [
    ("Confusion matrices", ROOT / "results" / "confusion_matrices"),
    ("Training graphs", ROOT / "results" / "training_plots"),
]
shown = False
for title, directory in plots:
    images = sorted(directory.glob("*.png")) if directory.exists() else []
    images = [path for path in images if path.name.startswith("letter_") == (task_key == "letter")]
    if images:
        shown = True
        st.subheader(title)
        for path in images:
            st.image(str(path), caption=path.stem.replace("_", " ").title())
if not shown:
    st.info("Train and evaluate the models to generate graphs and confusion matrices.")
