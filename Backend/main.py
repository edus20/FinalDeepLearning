import requests
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import torch
import base64
from transformers import AutoTokenizer, AutoModel
import logging

app = FastAPI()

# Configuración de Roboflow (API REST)
API_URL = "https://detect.roboflow.com/clasificacion-videojuegos-2.0/2"
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
).to(torch.device("cpu")).eval()  # Forzar CPU


# Configurar logging detallado
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.post("/classify")
async def classify_image(file: UploadFile):
    """Clasifica una imagen usando Roboflow (YOLO) y OCR."""
    try:
        logger.info("Iniciando procesamiento de la imagen...")

        # Leer la imagen subida
        image = Image.open(file.file)

        # Convertir la imagen a bytes para enviarla a la API de Roboflow
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        logger.info("Imagen convertida a bytes con éxito.")

        # Llamar a la API de Roboflow
        response = requests.post(
            f"{API_URL}?api_key={API_KEY}",
            files={"file": image_bytes},
        )
        response.raise_for_status()  # Asegurarse de que la respuesta no tenga errores HTTP
        logger.info("Respuesta de Roboflow recibida con éxito.")
        yolo_result = response.json()

        # Procesar resultados de YOLO
        predictions = yolo_result.get("predictions", [])
        clase1 = predictions[0]["class"] if len(predictions) > 0 else "Sin detección"
        clase2 = predictions[1]["class"] if len(predictions) > 1 else "Sin detección"
        logger.info(f"Clases detectadas: {clase1}, {clase2}")

        # Procesar OCR
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        ocr_result = OCR_MODEL.chat(OCR_TOKENIZER, image, ocr_type="ocr")
        logger.info("OCR procesado correctamente.")

        # Devolver resultados
        return {
            "class_1": clase1,
            "class_2": clase2,
            "ocr_result": ocr_result,
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Error al comunicarse con Roboflow: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error al comunicarse con Roboflow: {str(e)}"},
        )
    except Exception as e:
        logger.error(f"Error procesando la imagen: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error procesando la imagen: {str(e)}"},
        )