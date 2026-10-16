"""Paths, model definitions and class-to-folder mappings shared by training, evaluation and serving."""

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

# Stage 1: smear type.
SMEAR_MODEL = "thin_thick_pbs_ic"
# Stage 2: which species model handles each smear type.
SPECIES_MODEL_FOR_SMEAR = {
    "thick_pbs": "thick_mlf_ic",
    "thin_pbs": "thin_mlf_ic",
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
        "m_P_vivax": ["thick_blood_smear/malaria/plasmodium_vivax"],
        "m_P_falciparum": ["thick_blood_smear/malaria/plasmodium_falciparum"],
        "lf_B_malayi": ["thick_blood_smear/lymphatic_filariasis/brugia_malayi"],
        "lf_W_bancrofti": ["thick_blood_smear/lymphatic_filariasis/wuchereria_bancrofti"],
    },
    "thin_mlf_ic": {
        "negative": ["thin_blood_smear/normal"],
        "m_P_vivax": ["thin_blood_smear/malaria/plasmodium_vivax"],
        "m_P_falciparum": ["thin_blood_smear/malaria/plasmodium_falciparum"],
    },
}

CLASS_DESCRIPTIONS = {
    "thick_pbs": "Thick peripheral blood smear.",
    "thin_pbs": "Thin peripheral blood smear.",
    "negative": "No relevant organism detected.",
    "m_P_vivax": "Malaria-causing Plasmodium vivax detected.",
    "m_P_falciparum": "Malaria-causing Plasmodium falciparum detected.",
    "lf_B_malayi": "Lymphatic filariasis-causing Brugia malayi detected.",
    "lf_W_bancrofti": "Lymphatic filariasis-causing Wuchereria bancrofti detected.",
}


def checkpoint_path(model_name: str) -> Path:
    return MODELS_DIR / model_name / "model.pt"


def split_dir(model_name: str, split: str) -> Path:
    return DATASET_DIR / model_name / split
