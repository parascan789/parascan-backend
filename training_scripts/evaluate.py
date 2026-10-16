"""Evaluate trained models on the held-out test split.

Per model: accuracy, macro-F1, per-class report and a confusion matrix, written to
models/<model>/eval_test.json and models/<model>/confusion_test.png.

With --pipeline: runs the full two-stage pipeline on every test image from
dataset/splits.csv and reports stage-1 accuracy and end-to-end accuracy, written to
models/pipeline_eval_test.json. Thin-smear filariasis images have no thin-model class,
so they count towards stage 1 only.

Usage:
    uv run python -m training_scripts.evaluate --model thin_thick_pbs_ic
    uv run python -m training_scripts.evaluate --all --pipeline
"""

import argparse
import csv
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from parascan.config import (
    MODEL_CLASSES,
    MODELS_DIR,
    RAW_DIR,
    SMEAR_MODEL,
    SPECIES_MODEL_FOR_SMEAR,
    SPLITS_CSV,
    checkpoint_path,
    split_dir,
)
from parascan.modeling import pick_device
from parascan.pipeline import Classifier, Pipeline
from training_scripts.train import SplitDataset


def plot_confusion(cm, classes: list[str], title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(1.2 * len(classes) + 2, 1.2 * len(classes) + 1.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    peak = cm.max() if cm.size else 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > peak / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_model(name: str, device: torch.device) -> dict:
    clf = Classifier(name, device)
    ds = SplitDataset(split_dir(name, "test"), clf.classes, lambda im: im)
    if not len(ds):
        raise SystemExit(f"{name}: test split is empty; run prepare_dataset first.")
    y_true, y_pred = [], []
    for i in range(len(ds)):
        image, label = ds[i]
        pred = clf.predict(image)
        y_true.append(label)
        y_pred.append(clf.classes.index(pred["label"]))

    labels = list(range(len(clf.classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    present = sorted(set(y_true))
    report = classification_report(y_true, y_pred, labels=labels, target_names=clf.classes,
                                   output_dict=True, zero_division=0)
    result = {
        "model": name,
        "n_test": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        "classes": clf.classes,
        "per_class": {c: report[c] for c in clf.classes},
        "confusion_matrix": cm.tolist(),
        "missing_test_classes": [clf.classes[i] for i in labels if i not in present],
    }

    out = MODELS_DIR / name
    (out / "eval_test.json").write_text(json.dumps(result, indent=2))
    plot_confusion(cm, clf.classes, f"{name} (test, n={len(y_true)})", out / "confusion_test.png")

    print(f"\n=== {name}: {len(y_true)} test images ===")
    print(classification_report(y_true, y_pred, labels=labels, target_names=clf.classes, zero_division=0))
    if result["missing_test_classes"]:
        print(f"note: no test images for {result['missing_test_classes']}")
    return result


def expected_labels(image: str) -> tuple[str, str | None]:
    """Ground truth (smear label, species label) for a raw image path; species is None if out of scope."""
    def owner(model: str) -> str | None:
        return next((c for c, folders in MODEL_CLASSES[model].items()
                     if any(image.startswith(f + "/") for f in folders)), None)

    smear = owner(SMEAR_MODEL)
    return smear, owner(SPECIES_MODEL_FOR_SMEAR[smear]) if smear else None


def evaluate_pipeline(device: torch.device) -> dict:
    pipeline = Pipeline(str(device))
    with SPLITS_CSV.open(newline="", encoding="utf-8") as f:
        test_images = [r["image"] for r in csv.DictReader(f) if r["split"] == "test"]

    stage1_ok, e2e_ok, e2e_total, errors = 0, 0, 0, []
    out_of_scope = Counter()
    for img in test_images:
        smear_true, species_true = expected_labels(img)
        with Image.open(RAW_DIR / img) as im:
            pred = pipeline.predict(im)
        smear_pred, species_pred = pred["smear_type"]["label"], pred["diagnosis"]["label"]
        stage1_ok += smear_pred == smear_true
        if species_true is None:
            out_of_scope[img.rsplit("/", 1)[0]] += 1
            continue
        e2e_total += 1
        if smear_pred == smear_true and species_pred == species_true:
            e2e_ok += 1
        else:
            errors.append({"image": img, "expected": [smear_true, species_true], "predicted": [smear_pred, species_pred]})

    result = {
        "n_test": len(test_images),
        "stage1_accuracy": stage1_ok / max(len(test_images), 1),
        "end_to_end_n": e2e_total,
        "end_to_end_accuracy": e2e_ok / max(e2e_total, 1),
        "stage2_out_of_scope": dict(out_of_scope),
        "errors": errors,
    }
    (MODELS_DIR / "pipeline_eval_test.json").write_text(json.dumps(result, indent=2))

    print(f"\n=== Two-stage pipeline: {len(test_images)} test images ===")
    print(f"Stage 1 (smear type) accuracy: {result['stage1_accuracy']:.3f}")
    print(f"End-to-end accuracy:           {result['end_to_end_accuracy']:.3f} ({e2e_ok}/{e2e_total})")
    if out_of_scope:
        print(f"Stage-1 only (no stage-2 class): {dict(out_of_scope)}")
    for e in errors:
        print(f"  wrong: {e['image']}  expected {e['expected']}  got {e['predicted']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(MODEL_CLASSES))
    parser.add_argument("--all", action="store_true", help="evaluate every model that has a checkpoint")
    parser.add_argument("--pipeline", action="store_true", help="also evaluate the two-stage pipeline end to end")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if not (args.model or args.all or args.pipeline):
        parser.error("pass --model, --all and/or --pipeline")

    device = pick_device(args.device)
    names = [n for n in MODEL_CLASSES if checkpoint_path(n).exists()] if args.all else ([args.model] if args.model else [])
    for name in names:
        evaluate_model(name, device)
    if args.pipeline:
        evaluate_pipeline(device)


if __name__ == "__main__":
    main()
