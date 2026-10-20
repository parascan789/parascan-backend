"""Three-stage inference: smear type -> disease -> species.

The species model only runs when the disease model says malaria or lymphatic
filariasis with probability >= DISEASE_THRESHOLD. If the most likely disease is
below the threshold the result is "inconclusive" and should be reviewed manually.
If a needed disease or species model hasn't been trained, the result is "unavailable".
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
    """Loads whichever models are trained. Only the smear-type model is required; a missing
    disease or species model makes that branch return status "unavailable"."""

    def __init__(self, device: str | None = None, threshold: float = DISEASE_THRESHOLD):
        if not checkpoint_path(SMEAR_MODEL).exists():
            raise FileNotFoundError(f"Missing trained model {SMEAR_MODEL}. Train it with training_scripts.train.")
        self.device = pick_device(device)
        self.threshold = threshold
        self.missing = [name for name in required_models() if not checkpoint_path(name).exists()]

        def load(name: str) -> Classifier | None:
            return None if name in self.missing else Classifier(name, self.device)

        self.smear = Classifier(SMEAR_MODEL, self.device)
        self.disease = {smear: load(name) for smear, name in DISEASE_MODEL_FOR_SMEAR.items()}
        self.species = {key: load(name) for key, name in SPECIES_MODEL_FOR_DISEASE.items()}

    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        smear = self.smear.predict(image)
        disease = species = None
        disease_model = self.disease[smear["label"]]

        if disease_model is None:
            status, label = "unavailable", None
            description = f"Disease model {DISEASE_MODEL_FOR_SMEAR[smear['label']]} is not trained yet."
        else:
            disease = disease_model.predict(image)
            if disease["label"] == NEGATIVE_CLASS:
                status, label = "negative", NEGATIVE_CLASS
                description = CLASS_DESCRIPTIONS[NEGATIVE_CLASS]
            elif disease["confidence"] < self.threshold:
                status, label = "inconclusive", disease["label"]
                description = (f"Possible {disease['label'].replace('_', ' ')} "
                               f"({disease['confidence']:.0%} < {self.threshold:.0%} threshold); "
                               f"manual review advised.")
            elif (species_model := self.species[(smear["label"], disease["label"])]) is None:
                status, label = "unavailable", disease["label"]
                description = (f"{CLASS_DESCRIPTIONS[disease['label']]} Species model "
                               f"{SPECIES_MODEL_FOR_DISEASE[(smear['label'], disease['label'])]} is not trained yet.")
            else:
                species = species_model.predict(image)
                status, label = "positive", species["label"]
                description = CLASS_DESCRIPTIONS.get(label, "")

        return {
            "result": {"status": status, "label": label, "description": description, "threshold": self.threshold},
            "smear_type": smear,
            "disease": disease,
            "species": species,
        }

    def predict_bytes(self, data: bytes) -> dict:
        return self.predict(Image.open(io.BytesIO(data)))

    def info(self) -> dict:
        classifiers = [c for c in (self.smear, *self.disease.values(), *self.species.values()) if c]
        return {
            "device": str(self.device),
            "disease_threshold": self.threshold,
            "models": {c.name: {"classes": c.classes, "trained_at": c.trained_at} for c in classifiers},
            "missing_models": self.missing,
        }
