"""Evaluate trained models on the same task-specific test examples."""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

from .data_loader import load_dataset, set_seeds
from .predict import load_models
from .preprocessing import for_cnn, for_svm, normalize_images


ROOT = Path(__file__).resolve().parents[1]


def save_confusion_matrix(actual, predicted, name, directory, task):
    labels = list(range(26 if task == "letter" else 10))
    tick_labels = [chr(ord("A") + i) for i in labels] if task == "letter" else labels
    matrix = confusion_matrix(actual, predicted, labels=labels)
    fig, ax = plt.subplots(figsize=(11, 10) if task == "letter" else (7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=labels, yticks=labels, xticklabels=tick_labels,
           yticklabels=tick_labels, xlabel=f"Predicted {task}",
           ylabel=f"Actual {task}", title=f"{task.title()} {name} confusion matrix")
    if task == "digit":
        for row in labels:
            for col in labels:
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                        color="white" if matrix[row, col] > matrix.max() / 2 else "black",
                        fontsize=7)
    fig.tight_layout()
    prefix = "" if task == "digit" else "letter_"
    fig.savefig(directory / f"{prefix}{name.lower()}_confusion_matrix.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["digit", "letter"], default="digit")
    args = parser.parse_args()
    set_seeds()
    models, missing = load_models(args.task)
    if not models:
        print("No trained models found. Train at least one model before evaluation.")
        return
    if missing:
        print("Skipping missing models: " + ", ".join(missing))
    (_, _), (x_test, y_test) = load_dataset(args.task)
    inputs = {"MLP": normalize_images(x_test), "SVM": for_svm(x_test),
              "CNN": for_cnn(x_test)}
    results = ROOT / "results"
    matrices = results / "confusion_matrices"
    matrices.mkdir(parents=True, exist_ok=True)
    prefix = "" if args.task == "digit" else "letter_"
    rows = []
    for name, model in models.items():
        print(f"Evaluating {name}...")
        start = time.perf_counter()
        if name == "SVM":
            predicted = model.predict(inputs[name])
        else:
            probabilities = model.predict(inputs[name], batch_size=256, verbose=0)
            predicted = np.argmax(probabilities, axis=1)
        inference_seconds = time.perf_counter() - start
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predicted, average="macro", zero_division=0)
        metadata_file = results / f"{prefix}{name.lower()}_training.json"
        metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predicted),
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Training Time (s)": metadata.get("training_seconds", float("nan")),
            "Inference Time (s)": inference_seconds,
            "Training Samples": metadata.get("train_size", float("nan")),
        })
        labels = list(range(26 if args.task == "letter" else 10))
        names = [chr(ord("A") + i) for i in labels] if args.task == "letter" else None
        report = classification_report(y_test, predicted, labels=labels,
                                       target_names=names, zero_division=0)
        (results / f"{prefix}{name.lower()}_classification_report.txt").write_text(report)
        save_confusion_matrix(y_test, predicted, name, matrices, args.task)
        print(f"{name} Accuracy: {rows[-1]['Accuracy']:.4f}")
    output = results / ("metrics_letters.csv" if args.task == "letter" else "metrics.csv")
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
