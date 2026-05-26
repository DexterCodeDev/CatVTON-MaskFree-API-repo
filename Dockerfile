FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/huggingface
ENV TORCH_HOME=/tmp/torch

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Auto pull CatVTON source
RUN git clone https://github.com/Zheng-Chong/CatVTON.git /app/catvton

WORKDIR /app/catvton

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

COPY . .

ENV PYTHONPATH="/app/catvton"

EXPOSE 8080

CMD ["python", "app.py"]
