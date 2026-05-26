import io
import os
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from huggingface_hub import login

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
    """Loads the Mask-Free CatVTON weights into the GPU during container boot."""
    global model_pipeline
    try:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("WARNING: HF_TOKEN environment variable is missing!")
        
        login(token=hf_token)
        
        # Using torch.float16 for hardware compatibility with the NVIDIA L4
        model_pipeline = CatVTONPipeline(
            base_ckpt="runwayml/stable-diffusion-inpainting",
            attn_ckpt="zhengchong/CatVTON-MaskFree",
            attn_ckpt_version="mix", 
            weight_dtype=torch.float16, 
            device='cuda'
        )
        
        # THE FIX: Replace the safety checker with a dummy pass-through function.
        # This returns an iterable list of [False], preventing the 'NoneType' crash 
        # while successfully bypassing the NSFW filter.
        dummy_safety_checker = lambda images, **kwargs: (images, [False] * len(images))
        
        if hasattr(model_pipeline, 'pipe'):
            model_pipeline.pipe.safety_checker = dummy_safety_checker
        else:
            model_pipeline.safety_checker = dummy_safety_checker
            
        print("Mask-Free Model loaded successfully with dummy safety checker.")
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.get("/")
def health_check():
    """Endpoint to check if the GPU has finished downloading and loading the model."""
    return {"status": "active", "model_loaded": model_pipeline is not None}

@app.post("/try-on")
async def generate_try_on(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...)
):
    """Processes the 2-image try-on request and returns the synthesized image."""
    if not model_pipeline:
        raise HTTPException(status_code=503, detail="GPU Model is still initializing or failed to load.")

    try:
        # Convert incoming multipart files into RGB PIL Images
        person_img = Image.open(io.BytesIO(await person_image.read())).convert("RGB")
        garment_img = Image.open(io.BytesIO(await garment_image.read())).convert("RGB")

        # Generate a dummy white mask (255) to allow the Mask-Free AI to repaint the clothes
        dummy_mask = Image.new("L", person_img.size, 255)

        # Run inference on the GPU
        with torch.no_grad():
            result_image = model_pipeline(
                image=person_img,
                condition_image=garment_img,
                mask=dummy_mask,
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
