"""Paths, model definitions and class-to-folder mappings shared by training, evaluation and serving."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "dataset" / "raw"
DATASET_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
SPLITS_CSV = DATASET_DIR / "splits.csv"
SOURCES_CSV = RAW_DIR / "sources.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SPLITS = ("training", "validation", "test")
SPLIT_FRACTIONS = {"validation": 0.15, "test": 0.15}  # the rest goes to training

# Species are only classified when the disease model is at least this confident;
# below it the result is "inconclusive".
DISEASE_THRESHOLD = float(os.environ.get("PARASCAN_DISEASE_THRESHOLD", "0.80"))
NEGATIVE_CLASS = "negative"

# Stage 1: smear type.
SMEAR_MODEL = "thin_thick_pbs_ic"
# Stage 2: disease model for each smear type.
DISEASE_MODEL_FOR_SMEAR = {
    "thick_pbs": "thick_mlf_ic",
    "thin_pbs": "thin_mlf_ic",
}
# Stage 3: species model for each (smear type, disease).
SPECIES_MODEL_FOR_DISEASE = {
    ("thick_pbs", "malaria"): "thick_m_ic",
    ("thick_pbs", "lymphatic_filariasis"): "thick_lf_ic",
    ("thin_pbs", "malaria"): "thin_m_ic",
    ("thin_pbs", "lymphatic_filariasis"): "thin_lf_ic",
}

# Each model maps its class names (in output order) to the raw folders that feed them.
# A folder includes all images below it.
MODEL_CLASSES: dict[str, dict[str, list[str]]] = {
    "thin_thick_pbs_ic": {
        "thick_pbs": ["thick_blood_smear"],
        "thin_pbs": ["thin_blood_smear"],
    },
    "thick_mlf_ic": {
        "negative": ["thick_blood_smear/normal"],
        "malaria": ["thick_blood_smear/malaria"],
        "lymphatic_filariasis": ["thick_blood_smear/lymphatic_filariasis"],
    },
    "thin_mlf_ic": {
        "negative": ["thin_blood_smear/normal"],
        "malaria": ["thin_blood_smear/malaria"],
        "lymphatic_filariasis": ["thin_blood_smear/lymphatic_filariasis"],
    },
    "thick_m_ic": {
        "m_P_vivax": ["thick_blood_smear/malaria/plasmodium_vivax"],
        "m_P_falciparum": ["thick_blood_smear/malaria/plasmodium_falciparum"],
    },
    "thick_lf_ic": {
        "lf_B_malayi": ["thick_blood_smear/lymphatic_filariasis/brugia_malayi"],
        "lf_W_bancrofti": ["thick_blood_smear/lymphatic_filariasis/wuchereria_bancrofti"],
    },
    "thin_m_ic": {
        "m_P_vivax": ["thin_blood_smear/malaria/plasmodium_vivax"],
        "m_P_falciparum": ["thin_blood_smear/malaria/plasmodium_falciparum"],
    },
    "thin_lf_ic": {
        "lf_B_malayi": ["thin_blood_smear/lymphatic_filariasis/brugia_malayi"],
        "lf_W_bancrofti": ["thin_blood_smear/lymphatic_filariasis/wuchereria_bancrofti"],
    },
}

CLASS_DESCRIPTIONS = {
    "thick_pbs": "Thick peripheral blood smear.",
    "thin_pbs": "Thin peripheral blood smear.",
    "negative": "No relevant organism detected.",
    "malaria": "Malaria-causing Plasmodium detected.",
    "lymphatic_filariasis": "Lymphatic filariasis-causing microfilaria detected.",
    "m_P_vivax": "Malaria-causing Plasmodium vivax detected.",
    "m_P_falciparum": "Malaria-causing Plasmodium falciparum detected.",
    "lf_B_malayi": "Lymphatic filariasis-causing Brugia malayi detected.",
    "lf_W_bancrofti": "Lymphatic filariasis-causing Wuchereria bancrofti detected.",
}


def checkpoint_path(model_name: str) -> Path:
    return MODELS_DIR / model_name / "model.pt"


def split_dir(model_name: str, split: str) -> Path:
    return DATASET_DIR / model_name / split


def class_for_image(model_name: str, image: str) -> str | None:
    """The class of `model_name` that a raw image (path relative to RAW_DIR) belongs to, if any."""
    return next(
        (cls for cls, folders in MODEL_CLASSES[model_name].items()
         if any(image.startswith(folder.rstrip("/") + "/") for folder in folders)),
        None,
    )
