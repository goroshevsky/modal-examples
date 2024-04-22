from modal import Image, Stub, web_server, asgi_app
from fastapi import FastAPI
import gradio as gr
import asyncio
import os
import subprocess

DOCKER_IMAGE = "nvidia/cuda:12.3.1-base-ubuntu22.04"
PORT = 8000

# Initialize Fooocus
def init_Fooocus():
    os.chdir("/Fooocus")
    os.system("pip install -r requirements_versions.txt")
    os.chdir("./models/checkpoints")
    subprocess.run("wget -O juggernautXL_v8Rundiffusion.safetensors 'https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_v8Rundiffusion.safetensors'", shell=True)

# Define container image
web_image = web_server(
    PORT,
    DOCKER_IMAGE
)

stub = Stub()

# Define the web app
web_app = FastAPI()

@stub.function(image=web_image, keep_warm=1, container_idle_timeout=60 * 20)
@asgi_app()
def ui():
    """A simple Gradio interface around our Fooocus inference."""

    async def predict(prompt):
        # Assuming entry_with_update.py handles the prompt and generates an image
        # The following code is a placeholder and should be replaced with actual implementation
        process = subprocess.run(["python", "/Fooocus/entry_with_update.py", "--prompt", prompt], capture_output=True, text=True)
        # The path where Fooocus saves the generated image needs to be provided here
        # This is a placeholder path and should be replaced with the actual path used by Fooocus
        # For now, we simulate the prediction process with a placeholder image path
        output_path = "/Fooocus/outputs"  # This path is based on the grep search results
        generated_image_path = os.path.join(output_path, "generated_image.png")  # Assuming the generated image is named 'generated_image.png'
        return generated_image_path

    iface = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(label="Enter your prompt"),
        outputs=gr.Image(label="Generated Image"),
        title="Fooocus Image Generation",
        description="Enter a prompt to generate an image.",
        theme="default"
    )

    from gradio.routes import mount_gradio_app
    return mount_gradio_app(app=web_app, blocks=iface, path="/")
