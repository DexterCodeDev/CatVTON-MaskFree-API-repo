import io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

from model.pipeline import CatVTONPipeline

app = FastAPI(
    title="Serene Clothing Virtual Try-On API",
    description="Serverless GPU inference endpoint (Mask-Free).",
    contact={
        "email": "support@sereneclothing.store"
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_pipeline = None

@app.on_event("startup")
async def load_model():
    """Loads the Mask-Free CatVTON weights into the GPU."""
    global model_pipeline
    try:
        # Specifically pulling the Mask-Free checkpoint from HuggingFace
        model_pipeline = CatVTONPipeline.from_pretrained("zhengchong/CatVTON-MaskFree").to("cuda")
        print("Mask-Free Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.get("/")
def health_check():
    return {"status": "active", "model_loaded": model_pipeline is not None}

@app.post("/try-on")
async def generate_try_on(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...)
    # The mask requirement has been entirely removed
):
    """Processes the 2-image try-on request."""
    if not model_pipeline:
        raise HTTPException(status_code=503, detail="GPU Model is still initializing.")

    try:
        person_img = Image.open(io.BytesIO(await person_image.read())).convert("RGB")
        garment_img = Image.open(io.BytesIO(await garment_image.read())).convert("RGB")

        # Run inference on the GPU without a mask
        with torch.no_grad():
            result_image = model_pipeline(
                image=person_img,
                condition_image=garment_img,
                num_inference_steps=30, 
                guidance_scale=2.5
            )[0]

        memory_stream = io.BytesIO()
        result_image.save(memory_stream, format="PNG")
        memory_stream.seek(0)

        return StreamingResponse(memory_stream, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
