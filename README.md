# PARASCAN backend

A two-stage image classifier for peripheral blood smears (PBS), plus a FastAPI inference server.

```
image ──► thin_thick_pbs_ic ──► thick_pbs ──► thick_mlf_ic ──► negative | m_P_vivax | m_P_falciparum | lf_B_malayi | lf_W_bancrofti
                            └─► thin_pbs  ──► thin_mlf_ic  ──► negative | m_P_vivax | m_P_falciparum
```

All three models are ImageNet-pretrained `densenet121` networks (PyTorch / torchvision), fine-tuned on `dataset/raw`.

| Model | Classes |
|---|---|
| `thin_thick_pbs_ic` | `thick_pbs`, `thin_pbs` |
| `thick_mlf_ic` | `negative`, `m_P_vivax`, `m_P_falciparum`, `lf_B_malayi`, `lf_W_bancrofti` |
| `thin_mlf_ic` | `negative`, `m_P_vivax`, `m_P_falciparum` |

Class names map to raw folders in [parascan/config.py](parascan/config.py). Thin-smear filariasis images (`dataset/raw/thin_blood_smear/lymphatic_filariasis`) are used by the stage-1 model only.

## Setup

Requires [uv](https://docs.astral.sh/uv/). On Windows/Linux this installs CUDA 12.8 PyTorch.

```bash
uv sync
```

## Data

Put images under `dataset/raw/<smear>/<disease>/<species>/`. Uninfected smears go in `dataset/raw/thick_blood_smear/normal/` and `dataset/raw/thin_blood_smear/normal/`; they feed the `negative` class. Record provenance and licence of each image in `dataset/raw/sources.csv`.

Images are not committed to git. `dataset/splits.csv` (which image is in which split) is committed.

## Train, evaluate

```bash
# 1. Split raw images into dataset/<model>/{training,validation,test}/<class>/
uv run python -m training_scripts.prepare_dataset

# 2. Train each model (best checkpoint -> models/<model>/model.pt)
uv run python -m training_scripts.train --model thin_thick_pbs_ic
uv run python -m training_scripts.train --model thick_mlf_ic
uv run python -m training_scripts.train --model thin_mlf_ic

# 3. Test-set metrics per model, plus the full two-stage pipeline
uv run python -m training_scripts.evaluate --all --pipeline
```

Notes:
- **Splits are shared and stable.** Each raw image gets one split used by all three models, so the pipeline test images were never seen by any model during training. Crops from the same source (same URL in `sources.csv`) and near-duplicate images are kept in the same split. Rerunning `prepare_dataset` after adding images keeps existing assignments; `--resplit` starts over.
- **Missing classes.** `train` refuses to run if a class has no training images. `--allow-missing-classes` trains without them; the saved model then never predicts those classes.
- **Training** first trains the classifier head with the backbone frozen (`--warmup-epochs`, default 5), then fine-tunes everything with cosine LR decay and early stopping on validation macro-F1. Loss is class-weighted with label smoothing. Run `--help` for all options.
- **Outputs** in `models/<model>/`: `model.pt` (not committed), `history.json`, `eval_test.json`, `confusion_test.png`; pipeline results in `models/pipeline_eval_test.json`.

## Serve

```bash
uv run uvicorn server.app:app --reload
curl -F "file=@smear.jpg" http://localhost:8000/predict
```

Interactive docs at http://localhost:8000/docs.

| Endpoint | |
|---|---|
| `POST /predict` | multipart `file`; returns `smear_type` and `diagnosis` with label, confidence and per-class probabilities |
| `GET /models` | loaded models, classes, training timestamps |
| `GET /health` | liveness |

Environment variables: `PARASCAN_DEVICE` (`cpu`/`cuda`), `PARASCAN_CORS_ORIGINS` (comma-separated), `PARASCAN_MAX_UPLOAD_MB` (default 15).

### Docker (CPU)

Build after all three `model.pt` files exist:

```bash
docker build -t parascan-backend .
docker run -p 8000:8000 parascan-backend
```

The container honours `$PORT`, which most container hosts set. The hosting provider is not chosen yet.

> PARASCAN is a research prototype trained on a small dataset. It is not a diagnostic device.
