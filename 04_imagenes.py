import gradio as gr
import numpy as np
from PIL import Image, ImageOps

from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier


digits = load_digits()
modelo = MLPClassifier(hidden_layer_sizes=(64, ), max_iter=1000, random_state=0)
modelo.fit(digits.data, digits.target)
print("Modelo de digitos entrenado")

def reconocer(dibujo):
    print(dibujo)
    if dibujo is None:
        return {}

    img = dibujo["composite"]
    fondo = Image.new("RGBA", img.size, "white")
    img = Image.alpha_composite(fondo, img.convert("RGBA")).convert("L")
    img = ImageOps.invert(img)

    rescala = img.resize((8,8))
    arr = np.asarray(rescala, dtype=float)
    arr = arr / arr.max() * 16 if arr.max() > 0 else arr
    arr = arr.reshape(1, -1)

    probs = modelo.predict_proba(arr)[0]
    return {str(d): float(p) for d, p in enumerate(probs)}

demo = gr.Interface(
    fn=reconocer,
    inputs=gr.Sketchpad(
        type="pil",
        label="Dibuje un digito",
        brush=gr.Brush(default_size=15, colors=["#000000"], color_mode="fixed"),
        canvas_size=(280, 280)
    ),
    outputs=gr.Label(num_top_classes=10, label="Prediccion"),
    title="Reconocedor de digitos",
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch()
