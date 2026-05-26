import os
import io
import threading
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
# FastAPI
# -----------------------------
app = FastAPI()

pipeline = None
model_loading = False
startup_error = None


# -----------------------------
# Model Loader
# -----------------------------
def load_model():
    global pipeline, model_loading, startup_error

    try:
        model_loading = True

        print("Importing CatVTON...")

        from model.pipeline import CatVTONPix2PixPipeline
        from utils import resize_and_crop, resize_and_padding

        globals()["resize_and_crop"] = resize_and_crop
        globals()["resize_and_padding"] = resize_and_padding

        hf_token = os.getenv("HF_TOKEN")

        print("Downloading model from HuggingFace...")

        repo_path = snapshot_download(
            repo_id="zhengchong/CatVTON-MaskFree",
            token=hf_token
        )

        print("Loading pipeline...")

        pipeline = CatVTONPix2PixPipeline(
            base_ckpt="booksforcharlie/stable-diffusion-inpainting",
            attn_ckpt=repo_path,
            attn_ckpt_version="mix",
            weight_dtype=DTYPE,
            device=DEVICE
        )

        print("Model loaded successfully.")

    except Exception as e:
        startup_error = str(e)
        print(f"MODEL LOAD FAILED: {startup_error}")

    finally:
        model_loading = False


# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
async def startup_event():
    thread = threading.Thread(
        target=load_model,
        daemon=True
    )
    thread.start()


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
async def root():
    return {
        "message": "CatVTON API running"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": pipeline is not None,
        "model_loading": model_loading,
        "startup_error": startup_error
    }


@app.post("/tryon")
async def tryon(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...)
):
    global pipeline

    if startup_error:
        raise HTTPException(
            status_code=500,
            detail=startup_error
        )

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model still loading"
        )

    try:
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

        with torch.inference_mode():

            result = pipeline(
                image=person,
                condition_image=cloth,
                num_inference_steps=30,
                guidance_scale=2.5,
                generator=generator
            )[0]

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
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
