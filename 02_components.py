import gradio as gr


def calc_imc(nombre, peso_kg, altura_cm, detalle):
    altura_m = altura_cm / 100
    imc = peso_kg / (altura_m ** 2)

    if imc < 18.5:
        categoria = "Bajo peso"
    elif imc < 25:
        categoria = "Normal"
    elif imc < 30:
        categoria = "Sobrepeso"
    else:
        categoria = "Obesidad"

    mensaje = f"{nombre}, tu IMC es {imc:.1f} -> {categoria}"

    if detalle:
        mensaje += f"\npeso={peso_kg} kg, altura={altura_m:.2f} m"

    return mensaje, round(imc, 1)


demo = gr.Interface(
    fn=calc_imc,
    inputs=[
        gr.Textbox(label="Nombre", placeholder="Escribe tu nombre"),
        gr.Slider(30, 150, value=70, step=1, label="Peso (kg)"),
        gr.Number(value=170, label="Altura (cm)"),
        gr.Checkbox(label="Mostrar detalle")
    ],
    outputs=[
        gr.Textbox(label="Resultado", lines=2),
        gr.Number(label="IMC")
    ],
    title="Calculadora de IMC",
    description="Primer ejemplo con varios inputs y outputs",
    flagging_mode="never",
    examples=[
        ["Ana", 55, 150, False]
    ]
)


if __name__ == "__main__":
    demo.launch()
