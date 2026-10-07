"""Train an RBF support vector machine on a configurable MNIST sample."""

import argparse
import json
import time
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from .data_loader import SEED, load_dataset, set_seeds
from .preprocessing import for_svm


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--task", choices=["digit", "letter"], default="digit")
    args = parser.parse_args()
    set_seeds()
    (x_train, y_train), (x_test, y_test) = load_dataset(args.task)
    if not 10 <= args.train_size <= len(x_train):
        parser.error(f"--train-size must be between 10 and {len(x_train)}")
    # Choose a repeatable random sample instead of the first N records.
    import numpy as np
    indices = np.random.default_rng(SEED).choice(len(x_train), args.train_size, replace=False)
    model = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=SEED)
    print(f"Training {args.task} SVM on {args.train_size} images...")
    start = time.perf_counter()
    model.fit(for_svm(x_train[indices]), y_train[indices])
    training_seconds = time.perf_counter() - start
    predictions = model.predict(for_svm(x_test))
    print(f"{args.task.title()} SVM Test Accuracy: {accuracy_score(y_test, predictions):.4f}")
    (ROOT / "models").mkdir(exist_ok=True)
    print("Saving model...")
    prefix = "" if args.task == "digit" else "letter_"
    joblib.dump(model, ROOT / "models" / f"{prefix}svm_model.pkl")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{prefix}svm_training.json").write_text(
        json.dumps({"training_seconds": training_seconds, "train_size": args.train_size}, indent=2)
    )


if __name__ == "__main__":
    main()
