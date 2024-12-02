import gradio as gr
import requests

def clasificacion(image_path, url):
    try:
        if image_path:
            with open(image_path, "rb") as image_file:
                response = requests.post(
                    "http://backend:8000/classify",  # Dirección del backend
                    files={"file": image_file},
                    data={"url": url},
                )
        else:
            response = requests.post(
                "http://backend:8000/classify",
                data={"url": url},
            )

        result = response.json()
        platform = result.get("class_1", "Sin detección")
        name = result.get("class_2", "Sin detección")
        ocr_result = result.get("ocr_result", "Sin texto")

        return platform, name, ocr_result

    except Exception as e:
        return "Error", "Error", f"Error: {str(e)}"


# Interfaz Gradio
inputs = [
    gr.Image(type="filepath", label="Sube una imagen JPG"),
    gr.Textbox(label="URL", placeholder="Introduce la URL"),
]
outputs = [
    gr.Textbox(label="Plataforma"),
    gr.Textbox(label="Clasificación por edades"),
    gr.Textbox(label="Resultado OCR"),
]

# Crear la interfaz
gr.Interface(
    fn=clasificacion,
    inputs=inputs,
    outputs=outputs,
    title="Clasificación y OCR con YOLO y Roboflow",
    description="Sube una imagen JPG para obtener las clases detectadas por YOLO o una URL con la imagen deseada."
).launch(server_name="0.0.0.0", server_port=7860)
