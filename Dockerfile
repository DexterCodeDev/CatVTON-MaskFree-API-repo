FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/huggingface
ENV TORCH_HOME=/tmp/torch
ENV PYTHONPATH=/app/catvton

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

# Install ONLY pinned dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clone CatVTON source only (NO requirements install)
RUN git clone https://github.com/Zheng-Chong/CatVTON.git /app/catvton

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
