import io
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
# Depending on your exact CatVTON implementation, you may need to import your custom pipeline script here.
# from pipeline import CatVTONPipeline 

app = FastAPI(title="CatVTON-MaskFree API")
logging.basicConfig(level=logging.INFO)

# --- Model Initialization ---
# Load the model outside the request handler so it initializes during the container's 
# cold start, making subsequent requests much faster.
try:
    logging.info("Initializing CatVTON pipeline...")
    # NOTE: Replace with actual CatVTON loading logic
    # pipeline = CatVTONPipeline(
    #     base_ckpt="zhengchong/CatVTON-MaskFree", 
    #     device="cuda", 
    #     torch_dtype=torch.float16
    # )
    logging.info("Pipeline loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load pipeline: {e}")

@app.post("/try-on")
async def generate_tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...)
):
    try:
        # 1. Read and convert uploaded images to PIL Images
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        
        person_pil = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment_pil = Image.open(io.BytesIO(garment_bytes)).convert("RGB")

        # 2. Run Inference
        # result_image = pipeline(person_pil, garment_pil)
        
        # --- Placeholder for successful compilation ---
        result_image = person_pil # Replace with actual output from pipeline

        # 3. Convert generated image back to byte stream for response
        output_buf = io.BytesIO()
        result_image.save(output_buf, format="PNG")
        output_buf.seek(0)

        return StreamingResponse(output_buf, media_type="image/png")

    except Exception as e:
        logging.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during generation.")
