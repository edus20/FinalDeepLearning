import requests
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoTokenizer, AutoModel
import logging

app = FastAPI()

# Configuración de Roboflow (API REST)
API_URL = "https://detect.roboflow.com/clasificacion-videojuegos-2.0/8"
API_KEY = "zuGPdxGRyCVN3EtRgecJ"

# Configuración de OCR
OCR_TOKENIZER = AutoTokenizer.from_pretrained(
    'ucaslcl/GOT-OCR2_0', trust_remote_code=True
)
OCR_MODEL = AutoModel.from_pretrained(
    'ucaslcl/GOT-OCR2_0',
    trust_remote_code=True,
    use_safetensors=True,
    pad_token_id=OCR_TOKENIZER.eos_token_id,
).to(torch.device("cpu")).eval()

# Configurar logging detallado
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@app.post("/classify")
async def classify_image(file: UploadFile = None):
    if not file:
        return JSONResponse(
            status_code=400,
            content={"error": "No se proporcionó ninguna imagen."}
        )

    try:
        # Procesar la imagen recibida
        logger.info("Procesando imagen proporcionada...")
        image = Image.open(file.file)

        # **OCR**: Procesar texto desde la imagen
        ocr_result = "No disponible"
        try:
            logger.info("Procesando OCR...")
            ocr_result = OCR_MODEL.chat(OCR_TOKENIZER, image, ocr_type="ocr")
            logger.info("OCR procesado correctamente.")
        except Exception as e:
            logger.error(f"Error procesando la imagen con OCR: {e}")
            ocr_result = f"Error de OCR: {str(e)}"

        # **Roboflow**: Enviar imagen a la API
        clase1 = "No proporcionada"
        clase2 = "No proporcionada"
        try:
            logger.info("Enviando imagen a Roboflow...")
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            image_bytes = buffered.getvalue()

            response = requests.post(
                f"{API_URL}?api_key={API_KEY}",
                files={"file": ("image.jpg", image_bytes, "image/jpeg")}
            )
            response.raise_for_status()
            yolo_result = response.json()

            # Procesar resultados de YOLO
            predictions = yolo_result.get("predictions", [])
            clase1 = predictions[0]["class"] if len(predictions) > 0 else "Sin detección"
            clase2 = predictions[1]["class"] if len(predictions) > 1 else "Sin detección"
            logger.info(f"Clases detectadas: {clase1}, {clase2}")
        except Exception as e:
            logger.error(f"Error procesando la imagen con Roboflow: {e}")
            clase1 = "Error procesando la imagen"
            clase2 = "Error procesando la imagen"

        # Responder con los resultados
        return {
            "class_1": clase1,
            "class_2": clase2,
            "ocr_result": ocr_result,
        }

    except Exception as e:
        logger.error(f"Error procesando la solicitud: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error procesando la solicitud: {str(e)}",
                "class_1": "Error",
                "class_2": "Error",
                "ocr_result": f"Error de OCR: {str(e)}",
            },
        )
