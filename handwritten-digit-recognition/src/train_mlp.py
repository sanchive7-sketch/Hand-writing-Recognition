"""Train the backpropagation (fully connected) neural network."""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from .data_loader import load_dataset, set_seeds
from .preprocessing import normalize_images


ROOT = Path(__file__).resolve().parents[1]


def build_model(num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_plots(history, task):
    plot_dir = ROOT / "results" / "training_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric, label in (("accuracy", "Accuracy"), ("loss", "Loss")):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history.history[metric], label="Training")
        ax.plot(history.history["val_" + metric], label="Validation")
        ax.set(title=f"{task.title()} MLP {label}", xlabel="Epoch", ylabel=label)
        ax.legend()
        fig.tight_layout()
        prefix = "" if task == "digit" else "letter_"
        fig.savefig(plot_dir / f"{prefix}mlp_{metric}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--task", choices=["digit", "letter"], default="digit")
    args = parser.parse_args()
    set_seeds()
    (x_train, y_train), (x_test, y_test) = load_dataset(args.task)
    train_size = len(x_train) if args.train_size is None else args.train_size
    if not 10 <= train_size <= len(x_train):
        parser.error(f"--train-size must be between 10 and {len(x_train)}")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    x_train = normalize_images(x_train[:train_size])
    x_test = normalize_images(x_test)
    model = build_model(26 if args.task == "letter" else 10)
    print(f"Training {args.task} MLP...")
    start = time.perf_counter()
    history = model.fit(x_train, y_train[:train_size], epochs=args.epochs,
                        batch_size=128, validation_split=0.1, verbose=2)
    training_seconds = time.perf_counter() - start
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"{args.task.title()} MLP Test Accuracy: {accuracy:.4f}")
    (ROOT / "models").mkdir(exist_ok=True)
    print("Saving model...")
    prefix = "" if args.task == "digit" else "letter_"
    model.save(ROOT / "models" / f"{prefix}mlp_model.keras")
    save_plots(history, args.task)
    (ROOT / "results" / f"{prefix}mlp_training.json").write_text(
        json.dumps({"training_seconds": training_seconds, "train_size": train_size}, indent=2)
    )


if __name__ == "__main__":
    main()
