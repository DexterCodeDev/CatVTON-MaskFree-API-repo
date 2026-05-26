import os
import io
import torch

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from huggingface_hub import snapshot_download
from PIL import Image

# -----------------------------
# CUDA optimization
# -----------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

print(f"Using device: {DEVICE}")

# -----------------------------
# CatVTON imports
# -----------------------------
from model.pipeline import CatVTONPix2PixPipeline
from utils import resize_and_crop, resize_and_padding

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI()

pipeline = None


# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
async def startup_event():
    global pipeline

    try:
        print("Loading CatVTON model...")

        hf_token = os.getenv("HF_TOKEN")

        if not hf_token:
            raise RuntimeError("HF_TOKEN environment variable missing")

        print("Downloading MaskFree model...")

        repo_path = snapshot_download(
            repo_id="zhengchong/CatVTON-MaskFree",
            token=hf_token
        )

        print("Initializing pipeline...")

        pipeline = CatVTONPix2PixPipeline(
            base_ckpt="booksforcharlie/stable-diffusion-inpainting",
            attn_ckpt=repo_path,
            attn_ckpt_version="mix",
            weight_dtype=DTYPE,
            device=DEVICE
        )

        print("CatVTON model loaded successfully.")

    except Exception as e:
        print(f"Startup failed: {str(e)}")
        raise e


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
async def root():
    return {
        "message": "CatVTON MaskFree API running"
    }


@app.get("/health")
async def health():
    return JSONResponse({
        "status": "healthy",
        "gpu": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": pipeline is not None
    })


@app.post("/tryon")
async def tryon(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...)
):
    global pipeline

    try:

        if pipeline is None:
            raise HTTPException(
                status_code=500,
                detail="Model not loaded"
            )

        print("Reading images...")

        person_bytes = await person_image.read()
        cloth_bytes = await cloth_image.read()

        person = Image.open(
            io.BytesIO(person_bytes)
        ).convert("RGB")

        cloth = Image.open(
            io.BytesIO(cloth_bytes)
        ).convert("RGB")

        width = 768
        height = 1024

        print("Preprocessing images...")

        person = resize_and_crop(
            person,
            (width, height)
        )

        cloth = resize_and_padding(
            cloth,
            (width, height)
        )

        generator = torch.Generator(
            device=DEVICE
        ).manual_seed(42)

        print("Running inference...")

        with torch.inference_mode():

            result = pipeline(
                image=person,
                condition_image=cloth,
                num_inference_steps=30,
                guidance_scale=2.5,
                generator=generator
            )[0]

        print("Creating response image...")

        img_io = io.BytesIO()

        result.save(
            img_io,
            format="PNG"
        )

        img_io.seek(0)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return StreamingResponse(
            img_io,
            media_type="image/png"
        )

    except Exception as e:
        print(f"Inference error: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(
        os.environ.get("PORT", 8080)
    )

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port
    )
