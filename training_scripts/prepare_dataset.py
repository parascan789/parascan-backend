"""Split dataset/raw into training/validation/test folders for every model.

Each raw image gets exactly one split, shared by all three models, so the stage-2 test
images were never seen by the stage-1 model either. Images that come from the same
source (same URL in sources.csv) or are near-duplicates (perceptual hash) are kept in
the same split so the test set doesn't leak training images.

The assignment is stored in dataset/splits.csv. Images already listed there keep their
split when new raw images are added, so earlier evaluations stay comparable.

Usage:
    uv run python -m training_scripts.prepare_dataset [--seed 42] [--resplit]
"""

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

from parascan.config import (
    DATASET_DIR,
    IMAGE_EXTENSIONS,
    MODEL_CLASSES,
    RAW_DIR,
    SOURCES_CSV,
    SPLIT_FRACTIONS,
    SPLITS,
    SPLITS_CSV,
    split_dir,
)

NEAR_DUPLICATE_BITS = 6  # max Hamming distance between 64-bit dHashes


def list_raw_images() -> list[str]:
    return sorted(
        p.relative_to(RAW_DIR).as_posix()
        for p in RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def dhash(path: Path) -> int:
    with Image.open(path) as im:
        px = list(im.convert("L").resize((9, 8), Image.LANCZOS).getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def source_urls() -> dict[str, str]:
    if not SOURCES_CSV.exists():
        return {}
    with SOURCES_CSV.open(newline="", encoding="utf-8") as f:
        return {f"{r['folder']}/{r['file']}": r["url"] for r in csv.DictReader(f) if r.get("url")}


def group_images(images: list[str]) -> dict[str, str]:
    """Union-find over shared source URLs and near-duplicate hashes; returns image -> group id."""
    parent = {img: img for img in images}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    by_url: dict[str, list[str]] = defaultdict(list)
    for img, url in source_urls().items():
        if img in parent:
            by_url[url].append(img)
    for members in by_url.values():
        for other in members[1:]:
            union(members[0], other)

    hashes = {img: dhash(RAW_DIR / img) for img in images}
    for i, a in enumerate(images):
        for b in images[i + 1:]:
            if bin(hashes[a] ^ hashes[b]).count("1") <= NEAR_DUPLICATE_BITS:
                if Path(a).parent != Path(b).parent:
                    print(f"  warning: near-duplicate images in different folders: {a} <-> {b}")
                union(a, b)

    return {img: find(img) for img in images}


def assign_splits(images: list[str], groups: dict[str, str], previous: dict[str, str], seed: int) -> dict[str, str]:
    """Stratify by leaf folder; within each folder, deal whole groups out to test, validation, training."""
    rng = random.Random(seed)
    assignment: dict[str, str] = {}

    # A group that already has a split (from a previous run) keeps it.
    group_split = {groups[img]: previous[img] for img in images if img in previous}

    by_folder: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for img in images:
        by_folder[str(Path(img).parent.as_posix())][groups[img]].append(img)

    for folder, folder_groups in sorted(by_folder.items()):
        n = sum(len(m) for m in folder_groups.values())
        targets = {s: (max(1, round(frac * n)) if n >= 3 else 0) for s, frac in SPLIT_FRACTIONS.items()}
        counts = {s: 0 for s in SPLITS}

        pending = []
        for gid, members in folder_groups.items():
            if gid in group_split:
                counts[group_split[gid]] += len(members)
                for img in members:
                    assignment[img] = group_split[gid]
            else:
                pending.append(gid)

        rng.shuffle(pending)
        for gid in pending:
            members = folder_groups[gid]
            split = next(
                (s for s in ("test", "validation") if counts[s] + len(members) <= targets[s] or counts[s] == 0 < targets[s]),
                "training",
            )
            counts[split] += len(members)
            group_split[gid] = split
            for img in members:
                assignment[img] = split
        print(f"  {folder:<60} train={counts['training']:>3} val={counts['validation']:>3} test={counts['test']:>3}")

    return assignment


def read_previous() -> dict[str, str]:
    if not SPLITS_CSV.exists():
        return {}
    with SPLITS_CSV.open(newline="", encoding="utf-8") as f:
        return {r["image"]: r["split"] for r in csv.DictReader(f)}


def materialise(assignment: dict[str, str]) -> None:
    for model_name, classes in MODEL_CLASSES.items():
        for split in SPLITS:
            shutil.rmtree(split_dir(model_name, split), ignore_errors=True)
            for cls in classes:
                split_dir(model_name, split).joinpath(cls).mkdir(parents=True, exist_ok=True)

        counts: dict[tuple[str, str], int] = defaultdict(int)
        for img, split in assignment.items():
            for cls, folders in classes.items():
                if any(img.startswith(folder.rstrip("/") + "/") for folder in folders):
                    # Raw file names repeat across folders, so prefix the folder path.
                    dest = split_dir(model_name, split) / cls / img.replace("/", "__")
                    shutil.copy2(RAW_DIR / img, dest)
                    counts[(cls, split)] += 1

        print(f"\n{model_name}")
        for cls in classes:
            row = "  ".join(f"{s}={counts[(cls, s)]:>3}" for s in SPLITS)
            flag = "   <-- NO TRAINING IMAGES" if counts[(cls, "training")] == 0 else ""
            print(f"  {cls:<16} {row}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resplit", action="store_true", help="ignore dataset/splits.csv and reassign everything")
    args = parser.parse_args()

    images = list_raw_images()
    print(f"Found {len(images)} raw images. Grouping by source and near-duplicates...")
    groups = group_images(images)
    print(f"{len(set(groups.values()))} independent groups.\n\nSplit per raw folder:")

    previous = {} if args.resplit else read_previous()
    assignment = assign_splits(images, groups, previous, args.seed)

    SPLITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SPLITS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "group", "split"])
        for img in images:
            writer.writerow([img, groups[img], assignment[img]])

    materialise(assignment)
    print(f"\nWrote {SPLITS_CSV.relative_to(DATASET_DIR.parent)} and per-model split folders.")


if __name__ == "__main__":
    main()
