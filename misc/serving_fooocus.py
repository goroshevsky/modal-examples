from modal import Image, Stub, web_server, asgi_app, App
from fastapi import FastAPI
import gradio as gr
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

# Define container image using the static factory method from_registry
# The base image is now a Python image that includes Python pre-installed
web_image = Image.debian_slim(python_version="3.8").pip_install(["gradio", "fastapi", "uvicorn"])

stub = Stub()

# Define the web app
web_app = FastAPI()

# Define the app with a name
app = App("fooocus-ui")

@app.function(image=web_image, keep_warm=1, container_idle_timeout=60 * 20)
def ui():
    """A simple Gradio interface around our Fooocus inference."""

    def predict(prompt):
        # Mock image generation process
        # This is a simple simulation of the image generation process
        # In a real scenario, this would call the actual image generation script or function
        output_path = "/Fooocus/outputs"
        os.makedirs(output_path, exist_ok=True)
        generated_image_name = prompt.replace(" ", "_") + ".png"
        generated_image_path = os.path.join(output_path, generated_image_name)
        # Create a blank image file as a placeholder for the generated image
        with open(generated_image_path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\xdac`\x00\x00\x00\x02\x00\x01\xe2!\xbc\x33\x00\x00\x00\x00IEND\xaeB`\x82')
        # Return the file path of the generated image
        return generated_image_path

    iface = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(label="Enter your prompt"),
        outputs=gr.Image(label="Generated Image", type="filepath"),
        title="Fooocus Image Generation",
        description="Enter a prompt to generate an image.",
        theme="default",
        analytics_enabled=False  # Disable analytics to prevent background threads
    )

    from gradio.routes import mount_gradio_app
    return mount_gradio_app(app=web_app, blocks=iface, path="/")
