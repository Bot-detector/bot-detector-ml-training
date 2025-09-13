FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /uvx /bin/

WORKDIR /app


RUN uv venv && uv pip install mlflow==3.1.4 \
    cloudpickle==3.1.1 \
    numpy==2.3.2 \
    pandas==2.3.1 \
    psutil==7.0.0 \
    pyarrow==20.0.0 \
    scikit-learn==1.7.1 \
    scipy==1.16.1 \
    boto3 
EXPOSE 5000

ENTRYPOINT ["bash", "-c", "uv run mlflow models serve -m $MODEL_URI --host 0.0.0.0 --port 5000 --no-conda"]
