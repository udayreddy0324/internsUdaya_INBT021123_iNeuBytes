"""Reproducible CIFAR-10 CNN baseline and controlled experiments."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers, regularizers

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_data(seed: int, validation_size: int):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train, y_test = y_train.ravel(), y_test.ravel()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x_train))
    val_idx, train_idx = order[:validation_size], order[validation_size:]
    normalize = lambda x: x.astype("float32") / 255.0
    return (normalize(x_train[train_idx]), y_train[train_idx]), (normalize(x_train[val_idx]), y_train[val_idx]), (normalize(x_test), y_test)


def augmentation(level: str | None):
    if not level:
        return None
    settings = {
        "light": (0.0, 0.0, 0.0),
        "moderate": (0.08, 0.08, 0.08),
        "aggressive": (0.20, 0.20, 0.20),
    }
    rotation, translation, contrast = settings[level]
    stack = [layers.RandomFlip("horizontal", seed=1)]
    if rotation:
        stack += [layers.RandomRotation(rotation, seed=2), layers.RandomTranslation(translation, translation, seed=3), layers.RandomContrast(contrast, seed=4)]
    return keras.Sequential(stack, name=f"augmentation_{level}")


def build_model(cfg: dict) -> keras.Model:
    reg = regularizers.l2(float(cfg.get("l2", 0))) if cfg.get("l2") else None
    use_bn, drop = bool(cfg.get("batch_norm")), float(cfg.get("dropout", 0))
    kernel = int(cfg.get("kernel_size", 3))
    inputs = keras.Input((32, 32, 3), name="image")
    x = inputs
    aug = augmentation(cfg.get("augmentation"))
    if aug:
        x = aug(x)
    for filters in ([64, 128, 256, 384] if cfg.get("deeper") else [64, 128, 256]):
        x = layers.Conv2D(filters, kernel, padding="same", kernel_regularizer=reg)(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        if drop:
            x = layers.Dropout(drop)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    if drop:
        x = layers.Dropout(drop)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name=cfg["name"])
    lr = float(cfg["learning_rate"])
    opts = {"adam": keras.optimizers.Adam, "sgd": lambda learning_rate: keras.optimizers.SGD(learning_rate, momentum=0.9), "rmsprop": keras.optimizers.RMSprop}
    model.compile(optimizer=opts[cfg["optimizer"]](learning_rate=lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_curves(history, out: Path, name: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("accuracy", "loss")):
        ax.plot(history.history[metric], label="train")
        ax.plot(history.history[f"val_{metric}"], label="validation")
        ax.set(title=metric.title(), xlabel="Epoch", ylabel=metric.title())
        ax.legend()
    fig.tight_layout(); fig.savefig(out / f"{name}_curves.png", dpi=160); plt.close(fig)


def evaluate(model, x_test, y_test, out: Path, name: str):
    probs = model.predict(x_test, batch_size=256, verbose=0)
    pred = probs.argmax(axis=1)
    report = classification_report(y_test, pred, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(9, 7)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True", title=f"{name} confusion matrix"); fig.tight_layout(); fig.savefig(out / f"{name}_confusion_matrix.png", dpi=160); plt.close(fig)
    pairs = []
    for i in range(10):
        for j in range(i + 1, 10):
            pairs.append((int(cm[i, j] + cm[j, i]), CLASSES[i], CLASSES[j]))
    pairs.sort(reverse=True)
    return report, [{"class_a": a, "class_b": b, "mutual_errors": n} for n, a, b in pairs[:3]]


def run_one(cfg, common, data, out: Path):
    set_seed(int(common["seed"]))
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    model = build_model(cfg)
    start = time.perf_counter()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=int(common["epochs"]), batch_size=int(common["batch_size"]), verbose=2)
    elapsed = time.perf_counter() - start
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    report, pairs = evaluate(model, x_test, y_test, out, cfg["name"])
    save_curves(history, out, cfg["name"])
    model.save(out / f"{cfg['name']}.keras")
    with (out / f"{cfg['name']}_architecture.txt").open("w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))
    result = {
        "experiment": cfg["name"], "test_accuracy": test_acc, "test_loss": test_loss,
        "precision_macro": report["macro avg"]["precision"], "recall_macro": report["macro avg"]["recall"], "f1_macro": report["macro avg"]["f1-score"],
        "train_accuracy": history.history["accuracy"][-1], "validation_accuracy": history.history["val_accuracy"][-1],
        "train_val_gap": history.history["accuracy"][-1] - history.history["val_accuracy"][-1],
        "parameters": model.count_params(), "training_seconds": elapsed, "epochs": len(history.history["loss"]),
        "most_confused_pairs": json.dumps(pairs),
    }
    pd.DataFrame(history.history).to_csv(out / f"{cfg['name']}_history.csv", index_label="epoch")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--only", nargs="*", help="Experiment names; default runs all")
    parser.add_argument("--quick", action="store_true", help="Smoke test: 1 epoch and 5000 training images")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    selected = [x for x in spec["experiments"] if not args.only or x["name"] in args.only]
    if not selected:
        raise SystemExit("No matching experiments")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    set_seed(spec["seed"]); data = load_data(spec["seed"], spec["validation_size"])
    if args.quick:
        spec["epochs"] = 1
        data = ((data[0][0][:5000], data[0][1][:5000]), data[1], data[2])
    results_file = out / "master_experiment_table.csv"
    existing = pd.read_csv(results_file).to_dict("records") if results_file.exists() else []
    for cfg in selected:
        row = run_one(cfg, spec, data, out)
        existing = [r for r in existing if r["experiment"] != cfg["name"]] + [row]
        pd.DataFrame(existing).sort_values("experiment").to_csv(results_file, index=False)
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()

