# PARASCAN backend

A three-stage image classifier for peripheral blood smears (PBS), plus a FastAPI inference server.

```
STAGE 1                  STAGE 2                           STAGE 3

                     ┌─ thick_pbs ─► s2_thick_mlf_ic ─┬─ negative
                     │                                ├─ malaria ≥ 80% ──────────────► s3_thick_m_ic  ─► m_P_vivax | m_P_falciparum
                     │                                └─ lymphatic_filariasis ≥ 80% ─► s3_thick_lf_ic ─► lf_B_malayi | lf_W_bancrofti
s1_thin_thick_pbs_ic ┤
                     │                                ┌─ negative
                     └─ thin_pbs ──► s2_thin_m_ic ────┴─ malaria ≥ 80% ──────────────► s3_thin_m_ic   ─► m_P_vivax | m_P_falciparum
```

Model names start with their stage: `s1_` smear type, `s2_` disease, `s3_` species.

1. **Smear type:** thick or thin PBS.
2. **Disease:** using the model for that smear type. Thick smears: negative, malaria or lymphatic filariasis. Thin smears: negative or malaria.
3. **Species:** only if the disease model's probability for malaria or lymphatic filariasis is at least the threshold (80%, `PARASCAN_DISEASE_THRESHOLD`). Below the threshold the result is **inconclusive** and should be reviewed manually.

Thin smears are not checked for lymphatic filariasis. LF microfilariae are assumed to be detected on the thick smear, and there are too few thin-smear LF images to train on. Those images are still used to teach `s1_thin_thick_pbs_ic` what a thin smear looks like.

All six models are ImageNet-pretrained `densenet121` networks (PyTorch / torchvision), fine-tuned on `dataset/raw`.

| Stage | Model | Classes |
|---|---|---|
| 1 | `s1_thin_thick_pbs_ic` | `thick_pbs`, `thin_pbs` |
| 2 | `s2_thick_mlf_ic` | `negative`, `malaria`, `lymphatic_filariasis` |
| 2 | `s2_thin_m_ic` | `negative`, `malaria` |
| 3 | `s3_thick_m_ic`, `s3_thin_m_ic` | `m_P_vivax`, `m_P_falciparum` |
| 3 | `s3_thick_lf_ic` | `lf_B_malayi`, `lf_W_bancrofti` |

Class names map to raw folders in [parascan/config.py](parascan/config.py).

## Setup

Requires [uv](https://docs.astral.sh/uv/). On Windows/Linux this installs CUDA 12.8 PyTorch; training and inference use the GPU automatically when available.

```bash
uv sync
```

## Data

Put images under `dataset/raw/<smear>/<disease>/<species>/`. Uninfected smears go in `dataset/raw/thick_blood_smear/normal/` and `dataset/raw/thin_blood_smear/normal/`; they feed the `negative` class of the disease models. Record provenance and licence of each image in `dataset/raw/sources.csv`.

Images are not committed to git. `dataset/splits.csv` (which image is in which split) is committed.

## Train, evaluate

```bash
# 1. Split raw images into dataset/<model>/{training,validation,test}/<class>/
uv run python -m training_scripts.prepare_dataset

# 2. Train (best checkpoint -> models/<model>/model.pt)
uv run python -m training_scripts.train --model all
uv run python -m training_scripts.train --model s2_thick_mlf_ic --epochs 60   # or one at a time

# 3. Test-set metrics per model, plus the full pipeline
uv run python -m training_scripts.evaluate --all --pipeline
```

Notes:
- **Splits are shared and stable.** Each raw image gets one split used by all models, so the pipeline test images were never seen by any model during training. Crops from the same source (same URL in `sources.csv`) and near-duplicate images are kept in the same split. Rerunning `prepare_dataset` after adding images keeps existing assignments; `--resplit` starts over.
- **Missing classes.** `train` refuses to train a model if one of its classes has no training images; with `--model all` that model is skipped and listed at the end. `--allow-missing-classes` trains without the empty classes; that model then never predicts them.
- **Training** first trains the classifier head with the backbone frozen (`--warmup-epochs`, default 5), then fine-tunes everything with cosine LR decay and early stopping on validation macro-F1. Loss is class-weighted with label smoothing. Run `--help` for all options.
- **Outputs** in `models/<model>/`: `model.pt` (not committed), `history.json`, `eval_test.json`, `confusion_test.png`; pipeline results in `models/pipeline_eval_test.json`, including how many test images came out inconclusive at the current threshold. Thin-smear LF test images are scored on smear type only.

## Serve

```bash
uv run python -m server                          # http://127.0.0.1:8000
uv run python -m server --host 0.0.0.0 --port 8000   # reachable from other machines
curl -F "file=@smear.jpg" http://localhost:8000/predict
```

Interactive docs at http://localhost:8000/docs. The server needs at least `s1_thin_thick_pbs_ic`; if a disease or species model it needs isn't trained yet, the result `status` is `unavailable` (see `GET /models` → `missing_models`). `--reload` restarts on code changes during development. `HOST` and `PORT` environment variables are honoured too.

| Endpoint | |
|---|---|
| `POST /predict` | multipart `file`; returns `result` (`status`: `negative` / `positive` / `inconclusive` / `unavailable`, final `label`) and each stage's prediction (`smear_type`, `disease`, `species`) with confidence and per-class probabilities. `species` is `null` unless the status is `positive`. |
| `POST /classify` | the same pipeline in [parascan-frontend](../parascan-frontend)'s format: multipart field `image`, returns `malaria` / `lymphatic_filariasis` with `detected` (disease probability ≥ threshold), `confidence` (disease probability) and `species`, plus a `parascan` block with smear type and status. Returns 503 if the needed disease model isn't trained. |
| `GET /models` | loaded models, classes, training timestamps, threshold |
| `GET /health` | liveness |

Environment variables: `PARASCAN_DEVICE` (`cpu`/`cuda`), `PARASCAN_DISEASE_THRESHOLD` (default `0.80`), `PARASCAN_CORS_ORIGINS` (comma-separated), `PARASCAN_MAX_UPLOAD_MB` (default 15).

### Connect parascan-frontend

In `parascan-frontend/.env` set `CLASSIFIER_API_URL=http://localhost:8000/classify`, start this server, then start the frontend with `node server.js` and open http://localhost:3000. The frontend's Node server calls this API server-to-server, so no CORS setting is needed.

To deploy on another machine: clone the repo, copy the `models/*/model.pt` files over, `uv sync`, then `uv run python -m server --host 0.0.0.0`. The hosting provider is not chosen yet.

> PARASCAN is a research prototype trained on a small dataset. It is not a diagnostic device.
