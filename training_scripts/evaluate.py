"""Evaluate trained models on the held-out test split.

Per model: accuracy, macro-F1, per-class report and a confusion matrix, written to
models/<model>/eval_test.json and models/<model>/confusion_test.png.

With --pipeline: runs the full smear -> disease -> species pipeline on every test image
from dataset/splits.csv and reports accuracy at each stage, the final result accuracy
and how many results were inconclusive (disease confidence below the threshold),
written to models/pipeline_eval_test.json.

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
    DISEASE_MODEL_FOR_SMEAR,
    MODEL_CLASSES,
    MODELS_DIR,
    NEGATIVE_CLASS,
    RAW_DIR,
    SMEAR_MODEL,
    SPECIES_MODEL_FOR_DISEASE,
    SPLITS_CSV,
    checkpoint_path,
    class_for_image,
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


def expected_result(image: str) -> dict:
    """Ground truth for a raw image path, following the same model chain as the pipeline.

    `disease` is None for images the disease model doesn't cover (thin-smear LF).
    """
    smear = class_for_image(SMEAR_MODEL, image)
    disease = class_for_image(DISEASE_MODEL_FOR_SMEAR[smear], image)
    species = None
    if disease not in (None, NEGATIVE_CLASS):
        species = class_for_image(SPECIES_MODEL_FOR_DISEASE[(smear, disease)], image)
    return {"smear_type": smear, "disease": disease, "species": species,
            "label": NEGATIVE_CLASS if disease == NEGATIVE_CLASS else species}


def evaluate_pipeline(device: torch.device) -> dict:
    pipeline = Pipeline(str(device))
    with SPLITS_CSV.open(newline="", encoding="utf-8") as f:
        test_images = [r["image"] for r in csv.DictReader(f) if r["split"] == "test"]
    if not test_images:
        raise SystemExit("No test images in dataset/splits.csv; run prepare_dataset first.")

    status_counts, rows = Counter(), []
    smear_ok = disease_ok = final_ok = 0
    smear_only = Counter()  # images the pipeline can't diagnose: scored on smear type only
    for img in test_images:
        truth = expected_result(img)
        with Image.open(RAW_DIR / img) as im:
            pred = pipeline.predict(im)
        if truth["disease"] is None:
            smear_only[img.rsplit("/", 1)[0]] += 1
            smear_ok += pred["smear_type"]["label"] == truth["smear_type"]
            continue
        got = {
            "smear_type": pred["smear_type"]["label"],
            "disease": pred["disease"]["label"],
            "species": pred["species"]["label"] if pred["species"] else None,
            "status": pred["result"]["status"],
            "label": pred["result"]["label"],
        }
        status_counts[got["status"]] += 1
        smear_ok += got["smear_type"] == truth["smear_type"]
        disease_ok += got["smear_type"] == truth["smear_type"] and got["disease"] == truth["disease"]
        correct = got["status"] != "inconclusive" and got["label"] == truth["label"]
        final_ok += correct
        if not correct:
            rows.append({"image": img, "expected": truth, "predicted": got})

    n_all = len(test_images)
    n = n_all - sum(smear_only.values())
    decided = n - status_counts["inconclusive"]
    result = {
        "n_test": n_all,
        "n_diagnosable": n,
        "smear_only": dict(smear_only),
        "disease_threshold": pipeline.threshold,
        "smear_type_accuracy": smear_ok / n_all,
        "disease_accuracy": disease_ok / n if n else None,
        "final_accuracy": final_ok / n if n else None,
        "final_accuracy_excluding_inconclusive": final_ok / decided if decided else None,
        "status_counts": dict(status_counts),
        "not_correct": rows,
    }
    (MODELS_DIR / "pipeline_eval_test.json").write_text(json.dumps(result, indent=2))

    print(f"\n=== Pipeline: {n_all} test images, disease threshold {pipeline.threshold:.0%} ===")
    print(f"Smear type correct:            {result['smear_type_accuracy']:.3f}  (all {n_all} images)")
    if smear_only:
        print(f"Smear type only (not diagnosable by the pipeline): {dict(smear_only)}")
    if not n:
        return result
    print(f"Smear type + disease correct:  {result['disease_accuracy']:.3f}")
    print(f"Final result correct:          {result['final_accuracy']:.3f}  ({final_ok}/{n})")
    if decided:
        print(f"  ...excluding inconclusive:   {result['final_accuracy_excluding_inconclusive']:.3f}  ({final_ok}/{decided})")
    print(f"Results: {dict(status_counts)}")
    for r in rows:
        e, g = r["expected"], r["predicted"]
        print(f"  {g['status']:<12} {r['image']}: expected {e['label']}, got {g['label']} "
              f"(smear {g['smear_type']}, disease {g['disease']})")
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
