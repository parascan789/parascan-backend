"""Three-stage inference: smear type -> disease -> species.

The species model only runs when the disease model says malaria or lymphatic
filariasis with probability >= DISEASE_THRESHOLD. If the most likely disease is
below the threshold the result is "inconclusive" and should be reviewed manually.
"""

import io

import torch
from PIL import Image

from parascan.config import (
    CLASS_DESCRIPTIONS,
    DISEASE_MODEL_FOR_SMEAR,
    DISEASE_THRESHOLD,
    NEGATIVE_CLASS,
    SMEAR_MODEL,
    SPECIES_MODEL_FOR_DISEASE,
    checkpoint_path,
)
from parascan.modeling import eval_transform, load_checkpoint, pick_device


class Classifier:
    def __init__(self, name: str, device: torch.device):
        self.name = name
        self.device = device
        self.model, self.classes, ckpt = load_checkpoint(checkpoint_path(name), device)
        self.trained_at = ckpt.get("trained_at")
        self.transform = eval_transform()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        x = self.transform(image).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu().tolist()
        best = max(range(len(probs)), key=probs.__getitem__)
        return {
            "model": self.name,
            "label": self.classes[best],
            "description": CLASS_DESCRIPTIONS.get(self.classes[best], ""),
            "confidence": probs[best],
            "probabilities": dict(zip(self.classes, probs)),
        }


def required_models() -> list[str]:
    return [SMEAR_MODEL, *DISEASE_MODEL_FOR_SMEAR.values(), *SPECIES_MODEL_FOR_DISEASE.values()]


class Pipeline:
    def __init__(self, device: str | None = None, threshold: float = DISEASE_THRESHOLD):
        missing = [name for name in required_models() if not checkpoint_path(name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing trained models: {missing}. Train them with training_scripts.train.")
        self.device = pick_device(device)
        self.threshold = threshold
        self.smear = Classifier(SMEAR_MODEL, self.device)
        self.disease = {smear: Classifier(name, self.device) for smear, name in DISEASE_MODEL_FOR_SMEAR.items()}
        self.species = {key: Classifier(name, self.device) for key, name in SPECIES_MODEL_FOR_DISEASE.items()}

    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        smear = self.smear.predict(image)
        disease = self.disease[smear["label"]].predict(image)
        species = None

        if disease["label"] == NEGATIVE_CLASS:
            status, label = "negative", NEGATIVE_CLASS
        elif disease["confidence"] >= self.threshold:
            species = self.species[(smear["label"], disease["label"])].predict(image)
            status, label = "positive", species["label"]
        else:
            status, label = "inconclusive", disease["label"]

        description = {
            "negative": CLASS_DESCRIPTIONS[NEGATIVE_CLASS],
            "positive": CLASS_DESCRIPTIONS.get(label, ""),
            "inconclusive": (f"Possible {disease['label'].replace('_', ' ')} "
                             f"({disease['confidence']:.0%} < {self.threshold:.0%} threshold); manual review advised."),
        }[status]

        return {
            "result": {"status": status, "label": label, "description": description, "threshold": self.threshold},
            "smear_type": smear,
            "disease": disease,
            "species": species,
        }

    def predict_bytes(self, data: bytes) -> dict:
        return self.predict(Image.open(io.BytesIO(data)))

    def info(self) -> dict:
        classifiers = [self.smear, *self.disease.values(), *self.species.values()]
        return {
            "device": str(self.device),
            "disease_threshold": self.threshold,
            "models": {c.name: {"classes": c.classes, "trained_at": c.trained_at} for c in classifiers},
        }
