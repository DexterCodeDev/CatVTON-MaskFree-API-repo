import io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

# The Dockerfile handles adding the external CatVTON repo to the PYTHONPATH
from model.pipeline import CatVTONPipeline

app = FastAPI(
    title="Serene Clothing Virtual Try-On API",
    description="Serverless GPU inference endpoint loading CatVTON externally.",
    contact={
        "email": "support@sereneclothing.store"
    }
)

# Crucial: Allow your web storefront to make requests to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Restrict this to your specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to hold the model in VRAM
model_pipeline = None

@app.on_event("startup")
async def load_model():
    """Loads the CatVTON weights into the GPU during container boot."""
    global model_pipeline
    try:
        model_pipeline = CatVTONPipeline.from_pretrained("zhengchong/CatVTON").to("cuda")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.get("/")
def health_check():
    return {"status": "active", "model_loaded": model_pipeline is not None}

@app.post("/try-on")
async def generate_try_on(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    mask_image: UploadFile = File(...)
):
    """Processes the try-on request and returns the synthesized image."""
    if not model_pipeline:
        raise HTTPException(status_code=503, detail="GPU Model is still initializing.")

    try:
        # Convert incoming multipart files into RGB PIL Images
        person_img = Image.open(io.BytesIO(await person_image.read())).convert("RGB")
        garment_img = Image.open(io.BytesIO(await garment_image.read())).convert("RGB")
        mask_img = Image.open(io.BytesIO(await mask_image.read())).convert("L")

        # Run inference on the GPU
        with torch.no_grad():
            result_image = model_pipeline(
                image=person_img,
                condition_image=garment_img,
                mask=mask_img,
                num_inference_steps=30, 
                guidance_scale=2.5
            )[0]

        # Convert output to a byte stream
        memory_stream = io.BytesIO()
        result_image.save(memory_stream, format="PNG")
        memory_stream.seek(0)

        # Stream directly back to the frontend
        return StreamingResponse(memory_stream, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
