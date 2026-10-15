import gradio as gr
import numpy as np
from PIL import Image, ImageOps, ImageFilter

from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier


digits = load_digits()
modelo = MLPClassifier(hidden_layer_sizes=(64, ), max_iter=1000, random_state=0)
modelo.fit(digits.data, digits.target)
print("Modelo de digitos entrenado")


def preprocesar(img: Image.Image) -> np.ndarray:
    img = img.convert("L")                 # escala de grises
    img = ImageOps.invert(img)             # trazo blanco sobre fondo negro

    # Recortar alrededor del dibujo y dejarlo cuadrado y centrado
    caja = img.getbbox()
    if caja is None:                       # lienzo vacío
        return None
    img = img.crop(caja)
    lado = max(img.size)
    lienzo = Image.new("L", (lado, lado), 0)
    lienzo.paste(img, ((lado - img.width) // 2, (lado - img.height) // 2))
    # Sin margen: en `digits` el número ocupa todo el cuadro de 8x8.
    # Un leve desenfoque imita el trazo "grueso" de esas imágenes.
    lienzo = lienzo.filter(ImageFilter.GaussianBlur(lado / 30))

    # Reducir a 8x8 y escalar a 0-16
    pequena = lienzo.resize((8, 8), Image.LANCZOS)
    arr = np.asarray(pequena, dtype=float)
    arr = arr / arr.max() * 16 if arr.max() > 0 else arr
    return arr.reshape(1, -1)


def reconocer(dibujo):
    print(dibujo)
    if dibujo is None:
        return {}

    img = dibujo["composite"]
    fondo = Image.new("RGBA", img.size, "white")
    img = Image.alpha_composite(fondo, img.convert("RGBA"))

    x = preprocesar(img)
    if x is None:
        return {}

    probs = modelo.predict_proba(x)[0]
    return {str(d): float(p) for d, p in enumerate(probs)}

demo = gr.Interface(
    fn=reconocer,
    inputs=gr.Sketchpad(
        type="pil",
        label="Dibuje un digito",
        brush=gr.Brush(default_size=25, colors=["#000000"], color_mode="fixed"),
        canvas_size=(280, 280)
    ),
    outputs=gr.Label(num_top_classes=10, label="Prediccion"),
    title="Reconocedor de digitos",
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch()
