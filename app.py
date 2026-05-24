FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Externally load the public CatVTON repository
RUN git clone https://github.com/Zheng-Chong/CatVTON.git /external_modules/CatVTON

# Add the cloned repo to the Python path
ENV PYTHONPATH="${PYTHONPATH}:/external_modules/CatVTON"

COPY . .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
