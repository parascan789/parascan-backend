"""Fine-tune ImageNet DenseNet121 on one of the PARASCAN models.

Training runs in two phases: first only the new classifier head is trained with the
backbone frozen, then the whole network is fine-tuned at a lower learning rate. The
checkpoint with the best validation macro-F1 is saved to models/<model>/model.pt.

Usage:
    uv run python -m training_scripts.train --model thin_thick_pbs_ic
    uv run python -m training_scripts.train --model thick_mlf_ic --epochs 60
    uv run python -m training_scripts.train --model all
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from parascan.config import IMAGE_EXTENSIONS, MODEL_CLASSES, checkpoint_path, split_dir
from parascan.modeling import build_densenet121, eval_transform, pick_device, save_checkpoint, train_transform


class SplitDataset(Dataset):
    def __init__(self, root: Path, classes: list[str], transform):
        self.transform = transform
        self.samples = [
            (p, idx)
            for idx, cls in enumerate(classes)
            for p in sorted((root / cls).glob("*"))
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        with Image.open(path) as im:
            return self.transform(im.convert("RGB")), label

    def class_counts(self, num_classes: int) -> list[int]:
        counts = [0] * num_classes
        for _, label in self.samples:
            counts[label] += 1
        return counts


def resolve_classes(model_name: str, allow_missing: bool) -> list[str]:
    classes = list(MODEL_CLASSES[model_name])
    empty = [c for c in classes if not any((split_dir(model_name, "training") / c).glob("*"))]
    if empty and not allow_missing:
        raise SystemExit(
            f"Classes with no training images: {empty}.\n"
            f"Add raw images and rerun prepare_dataset, or pass --allow-missing-classes "
            f"to train without them (the model will then never predict those classes)."
        )
    for c in empty:
        print(f"warning: training without class '{c}' (no images)")
    classes = [c for c in classes if c not in empty]
    if len(classes) < 2:
        raise SystemExit(f"Need at least 2 classes with training images, have {classes}.")
    return classes


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(y)
            preds += logits.argmax(1).tolist()
            targets += y.tolist()
    n = max(len(targets), 1)
    acc = sum(p == t for p, t in zip(preds, targets)) / n
    f1 = f1_score(targets, preds, average="macro", zero_division=0) if targets else 0.0
    return total_loss / n, acc, f1


def train_model(model_name: str, args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = pick_device(args.device)
    classes = resolve_classes(model_name, args.allow_missing_classes)

    train_ds = SplitDataset(split_dir(model_name, "training"), classes, train_transform())
    val_ds = SplitDataset(split_dir(model_name, "validation"), classes, eval_transform())
    counts = train_ds.class_counts(len(classes))
    print(f"Model {model_name} on {device}; classes {dict(zip(classes, counts))}; "
          f"{len(train_ds)} train / {len(val_ds)} val images")
    if not len(val_ds):
        raise SystemExit("Validation split is empty; rerun prepare_dataset.")

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers,
                     pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > args.batch_size, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = build_densenet121(len(classes)).to(device)
    # Inverse-frequency class weights offset the imbalance between species.
    weights = torch.tensor([len(train_ds) / (len(classes) * c) for c in counts], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    out_path = checkpoint_path(model_name)
    best = {"val_f1": -1.0, "val_loss": float("inf")}
    history, stale = [], 0
    total_epochs = args.warmup_epochs + args.epochs
    start = time.time()

    for epoch in range(1, total_epochs + 1):
        if epoch == 1:
            for p in model.features.parameters():
                p.requires_grad = False
            optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.head_lr, weight_decay=1e-4)
            scheduler = None
            phase = "head"
        if epoch == args.warmup_epochs + 1:
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": model.features.parameters(), "lr": args.lr},
                {"params": model.classifier.parameters(), "lr": args.lr * 10},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            phase, stale = "finetune", 0

        tr_loss, tr_acc, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()

        history.append({"epoch": epoch, "phase": phase, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1})
        improved = va_f1 > best["val_f1"] or (va_f1 == best["val_f1"] and va_loss < best["val_loss"])
        marker = ""
        if improved:
            best = {"epoch": epoch, "val_f1": va_f1, "val_acc": va_acc, "val_loss": va_loss}
            save_checkpoint(out_path, model, classes, {
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "best_epoch": epoch,
                "val_metrics": best,
                "train_class_counts": dict(zip(classes, counts)),
                "args": vars(args),
            })
            stale, marker = 0, "  *saved"
        else:
            stale += 1
        print(f"[{epoch:>3}/{total_epochs}] {phase:<8} train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} F1 {va_f1:.3f}{marker}")

        if phase == "finetune" and stale >= args.patience:
            print(f"Early stopping: no validation improvement for {args.patience} epochs.")
            break

    (out_path.parent / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nDone in {(time.time() - start) / 60:.1f} min. Best epoch {best['epoch']}: "
          f"val F1 {best['val_f1']:.3f}, acc {best['val_acc']:.3f}. Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["all", *MODEL_CLASSES])
    parser.add_argument("--epochs", type=int, default=40, help="max fine-tuning epochs (after head warm-up)")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="epochs with frozen backbone")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4, help="backbone learning rate during fine-tuning")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10, help="early-stopping patience in epochs")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-missing-classes", action="store_true")
    args = parser.parse_args()

    if args.model != "all":
        train_model(args.model, args)
        return

    skipped = {}
    for name in MODEL_CLASSES:
        try:
            train_model(name, args)
        except SystemExit as e:
            print(f"\nSkipping {name}: {e}\n")
            skipped[name] = str(e)
    if skipped:
        raise SystemExit(f"Not trained: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
