import os
import io
import torch
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from huggingface_hub import snapshot_download
from PIL import Image

# -------------------
# CUDA optimization
# -------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

print(f"Using device: {DEVICE}")

# -------------------
# Import CatVTON
# -------------------
from model.pipeline import CatVTONPix2PixPipeline
from utils import resize_and_crop, resize_and_padding

# -------------------
# Load model once
# -------------------
HF_TOKEN = os.getenv("HF_TOKEN")

print("Downloading CatVTON-MaskFree...")

repo_path = snapshot_download(
    repo_id="zhengchong/CatVTON-MaskFree",
    token=HF_TOKEN
)

print("Loading pipeline...")

pipeline = CatVTONPix2PixPipeline(
    base_ckpt="timbrooks/instruct-pix2pix",
    attn_ckpt=repo_path,
    attn_ckpt_version="mix-48k-1024",
    weight_dtype=DTYPE,
    use_tf32=True,
    device=DEVICE
)

print("Model loaded.")

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "CatVTON MaskFree API running"}


@app.get("/health")
async def health():
    return JSONResponse(
        {
            "status": "healthy",
            "gpu": DEVICE,
            "cuda_available": torch.cuda.is_available()
        }
    )


@app.post("/tryon")
async def tryon(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
):
    try:

        person_bytes = await person_image.read()
        cloth_bytes = await cloth_image.read()

        person = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        cloth = Image.open(io.BytesIO(cloth_bytes)).convert("RGB")

        width = 768
        height = 1024

        person = resize_and_crop(person, (width, height))
        cloth = resize_and_padding(cloth, (width, height))

        generator = torch.Generator(device=DEVICE).manual_seed(42)

        with torch.inference_mode():
            result = pipeline(
                image=person,
                condition_image=cloth,
                num_inference_steps=30,
                guidance_scale=2.5,
                generator=generator
            )[0]

        img_io = io.BytesIO()
        result.save(img_io, format="PNG")
        img_io.seek(0)

        torch.cuda.empty_cache()

        return StreamingResponse(
            img_io,
            media_type="image/png"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port
    )
