# CPU inference image for the PARASCAN API. Train the models first so models/*/model.pt exist.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PARASCAN_DEVICE=cpu
WORKDIR /app

COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

COPY parascan/ parascan/
COPY server/ server/
COPY models/thin_thick_pbs_ic/model.pt models/thin_thick_pbs_ic/model.pt
COPY models/thick_mlf_ic/model.pt models/thick_mlf_ic/model.pt
COPY models/thin_mlf_ic/model.pt models/thin_mlf_ic/model.pt

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
