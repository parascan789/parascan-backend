"""Two-stage inference: classify smear type, then run the matching species model."""

import io

import torch
from PIL import Image

from parascan.config import (
    CLASS_DESCRIPTIONS,
    SMEAR_MODEL,
    SPECIES_MODEL_FOR_SMEAR,
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


class Pipeline:
    def __init__(self, device: str | None = None):
        self.device = pick_device(device)
        self.smear = Classifier(SMEAR_MODEL, self.device)
        self.species = {
            smear: Classifier(model_name, self.device)
            for smear, model_name in SPECIES_MODEL_FOR_SMEAR.items()
        }

    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        smear = self.smear.predict(image)
        diagnosis = self.species[smear["label"]].predict(image)
        return {"smear_type": smear, "diagnosis": diagnosis}

    def predict_bytes(self, data: bytes) -> dict:
        return self.predict(Image.open(io.BytesIO(data)))

    def info(self) -> dict:
        classifiers = [self.smear, *self.species.values()]
        return {
            "device": str(self.device),
            "models": {c.name: {"classes": c.classes, "trained_at": c.trained_at} for c in classifiers},
        }
