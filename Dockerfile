# Service image. When the pipeline needs torch/CUDA, switch the base to a CUDA runtime image
# (e.g. pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime) and add requirements-gpu.txt.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY cashaudit cashaudit
COPY data data
COPY tools tools
ENV DATA_DIR=/app/var
EXPOSE 8000
# ONE uvicorn process: the job queue lives in memory. Scale with WORKERS (GPU threads), never uvicorn --workers.
CMD ["uvicorn", "cashaudit.api:app", "--host", "0.0.0.0", "--port", "8000"]
