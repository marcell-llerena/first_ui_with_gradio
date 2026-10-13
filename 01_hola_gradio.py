import gradio as gr

# Documentación :https://gradio.app/guides/quickstart
# Instalar uv add gradio
# Correr gradio uv run 01_hola_gradio.py
# Probar la UI en el navegador http://127.0.0.1:7860

def saludar(nombre):
    return f"Hola {nombre}, bienvenido a gradio!"

demo = gr.Interface(
    fn=saludar,
    inputs="text",
    outputs="text"
)

if __name__ == "__main__":
    demo.launch()
