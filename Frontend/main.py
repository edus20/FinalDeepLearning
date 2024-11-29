import gradio as gr
import requests

def clasificacion(image_path):
    with open(image_path, "rb") as image_file:
        response = requests.post(
            "http://backend:8000/classify",  # Dirección del backend
            files={"file": image_file}
        )
    result = response.json()
    return result["class_1"], result["class_2"], result["ocr_result"]

# Configurar la interfaz Gradio
inputs = gr.Image(type="filepath", label="Sube una imagen JPG")
outputs = [
    gr.Textbox(label="Clase 1"),
    gr.Textbox(label="Clase 2"),
    gr.Textbox(label="Resultado OCR")
]

gr.Interface(
    fn=clasificacion,
    inputs=inputs,
    outputs=outputs,
    title="Clasificación y OCR con YOLO y Roboflow",
    description="Sube una imagen JPG para obtener las clases detectadas por YOLO y el texto reconocido por OCR."
).launch(server_name="0.0.0.0", server_port=7860)
