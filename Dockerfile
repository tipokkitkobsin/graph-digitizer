# Graph Digitizer — Hugging Face Spaces (Docker SDK) Dockerfile
#
# Builds slim Python 3.11 image, installs deps, pre-downloads EasyOCR's English
# detector + recognizer into a baked-in cache so the first /api/predict after
# a cold start doesn't have to wait for a ~500 MB download.
#
# HF Spaces convention:
#   * runs as a non-root user (UID 1000)
#   * app listens on port 7860 (matches `app_port: 7860` in README frontmatter)

FROM python:3.11-slim

# System deps for OpenCV-headless + EasyOCR + ffmpeg shim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# --- Python deps (cached layer) ---
COPY --chown=user requirements.txt .
RUN pip install --user --no-cache-dir --upgrade pip wheel setuptools \
    && pip install --user --no-cache-dir -r requirements.txt

# --- Pre-download EasyOCR models into a known dir, baked into the image ---
# server.py reads EASYOCR_MODULE_PATH and points the EasyOCR.Reader at it.
ENV EASYOCR_MODULE_PATH=/home/user/app/.easyocr
RUN mkdir -p $EASYOCR_MODULE_PATH \
    && python -c "import easyocr; \
       easyocr.Reader(['en'], gpu=False, verbose=False, \
                      model_storage_directory='$EASYOCR_MODULE_PATH', \
                      user_network_directory='$EASYOCR_MODULE_PATH')"

# --- App code (own layer; changes here don't bust the deps cache above) ---
COPY --chown=user . .

# HF Spaces routes external traffic to the port declared in README frontmatter (7860)
EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
