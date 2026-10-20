"""Translate a pipeline result into the response format parascan-frontend expects.

Frontend contract (see parascan-frontend/README.md):
    { "malaria":              {"detected": bool, "confidence": 0..1 | null, "species": [...]},
      "lymphatic_filariasis": {...} }
    species item: {"species": "Plasmodium vivax", "detected": true, "confidence": 0..1,
                   "characteristics": "..."}

For each disease, `confidence` is the disease model's probability for that disease and
`detected` means it reached the threshold, so a 65% malaria score shows as "Not detected,
65%". A disease the smear type isn't checked for (lymphatic filariasis on thin smears)
gets `confidence: null`, which hides the meter. The full pipeline result is included
under `parascan` for clients that want smear type and status.
"""

from parascan.config import NEGATIVE_CLASS

FRONTEND_DISEASES = {"malaria": "malaria", "lymphatic_filariasis": "lymphatic_filariasis"}

SPECIES_INFO = {
    "m_P_falciparum": (
        "Plasmodium falciparum",
        "Small, delicate ring forms, often several per red cell, some with double chromatin dots; "
        "infected red cells are not enlarged; crescent (banana-shaped) gametocytes.",
    ),
    "m_P_vivax": (
        "Plasmodium vivax",
        "Infected red cells are enlarged and pale with Schüffner’s dots; amoeboid trophozoites; "
        "schizonts with 12–24 merozoites.",
    ),
    "lf_W_bancrofti": (
        "Wuchereria bancrofti",
        "Sheathed microfilaria with smooth, graceful curves; short head space; nuclei stop before "
        "the tail tip, leaving it clear.",
    ),
    "lf_B_malayi": (
        "Brugia malayi",
        "Sheathed microfilaria with kinked, angular curves; long head space; two separate nuclei at "
        "the tail tip; sheath stains pink with Giemsa.",
    ),
}


def to_frontend(result: dict) -> dict:
    """Assumes result["result"]["status"] != "unavailable" (the caller turns that into an error)."""
    disease = result["disease"]
    probs = disease["probabilities"]
    threshold = result["result"]["threshold"]
    species = result["species"]
    out = {}

    for key, disease_class in FRONTEND_DISEASES.items():
        if disease_class not in probs:
            out[key] = {"detected": False, "confidence": None, "species": []}
            continue
        confidence = probs[disease_class]
        detected = disease["label"] == disease_class and confidence >= threshold
        species_list = []
        if detected and species:
            name, characteristics = SPECIES_INFO[species["label"]]
            species_list.append({
                "species": name,
                "detected": True,
                "confidence": species["confidence"],
                "characteristics": characteristics,
            })
        out[key] = {"detected": detected, "confidence": confidence, "species": species_list}

    out["parascan"] = {
        "status": result["result"]["status"],
        "label": result["result"]["label"],
        "description": result["result"]["description"],
        "smear_type": result["smear_type"]["label"],
        "smear_type_confidence": result["smear_type"]["confidence"],
        "negative_probability": probs.get(NEGATIVE_CLASS),
    }
    return out
