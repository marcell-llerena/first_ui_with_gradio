# Instalar scikit-learn uv add scikit-learn

import gradio as gr
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


iris = load_iris()
X, y = iris.data, iris.target
clases = iris.target_names

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y)
print("Modelo entrenado")


def predecir(
    largo_sepalo,
    ancho_sepalo,
    largo_petalo,
    ancho_petalo
):
    muestra = [
        [largo_sepalo, ancho_sepalo, largo_petalo, ancho_petalo]
    ]

    probabilidades = modelo.predict_proba(muestra)[0]

    return {clase: float(p) for clase, p in zip(clases, probabilidades)}


demo = gr.Interface(
    fn=predecir,
    inputs=[
        gr.Slider(0.1, 8.0, value=3.8, step=0.1, label="Largo del sepalo (cm)"),
        gr.Slider(0.1, 8.0, value=3.8, step=0.1, label="Ancho del sepalo (cm)"),
        gr.Slider(0.1, 8.0, value=3.8, step=0.1, label="Largo del petalo (cm)"),
        gr.Slider(0.1, 8.0, value=3.8, step=0.1, label="Ancho del petalo (cm)")
    ],
    outputs=gr.Label(num_top_classes=3, label="Especie predicha"),
    title="Clasificador de iris",
    description="Mueve los sliders",
    flagging_mode="never",
    live=True,
    examples=[
        [5.1, 3.5, 1.4, 0.2],
        [6.0, 2.7, 4.5, 1.5],
        [6.9, 3.1, 5.8, 2.2]
    ],
)

if __name__ == "__main__":
    demo.launch()
