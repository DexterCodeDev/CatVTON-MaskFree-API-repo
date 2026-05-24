import io
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import torch

from pipeline import CatVTONPipeline 

app = FastAPI(title="CatVTON-MaskFree API")
logging.basicConfig(level=logging.INFO)

try:
    logging.info("Initializing CatVTON pipeline...")
    pipeline = CatVTONPipeline(
        base_ckpt="zhengchong/CatVTON-MaskFree", 
        device="cuda", 
        torch_dtype=torch.float16
    )
    logging.info("Pipeline loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load pipeline: {e}")

@app.post("/try-on")
async def generate_tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...)
):
    try:
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        
        person_pil = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment_pil = Image.open(io.BytesIO(garment_bytes)).convert("RGB")

        # Run Inference
        result_image = pipeline(person_pil, garment_pil)
        
        if isinstance(result_image, list):
            result_image = result_image[0]
        elif hasattr(result_image, 'images'):
            result_image = result_image.images[0]

        output_buf = io.BytesIO()
        result_image.save(output_buf, format="PNG")
        output_buf.seek(0)

        return StreamingResponse(output_buf, media_type="image/png")

    except Exception as e:
        logging.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during generation.")
