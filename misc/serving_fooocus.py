from modal import Image, Stub, gpu, web_server, asgi_app
from fastapi import FastAPI
import gradio as gr
from Fooocus.async_worker import AsyncWorker
import asyncio

DOCKER_IMAGE = "nvidia/cuda:12.3.1-base-ubuntu22.04"
PYTHON_VER = "3.10"
GPU_CONFIG = gpu.T4()
PORT = 8000

# Initialize Fooocus
def init_Fooocus():
    import os
    import subprocess

    os.chdir("/Fooocus")
    os.system("pip install -r requirements_versions.txt")
    os.chdir("./models/checkpoints")
    subprocess.run("wget -O juggernautXL_v8Rundiffusion.safetensors 'https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_v8Rundiffusion.safetensors'", shell=True)

# Define container image
web_image = web_server(
    DOCKER_IMAGE,
    python_version=PYTHON_VER,
    gpu=GPU_CONFIG,
    port=PORT,
    startup_command="pip install -r /Fooocus/requirements_versions.txt && python /Fooocus/launch.py --always-high-vram",
    files=["/Fooocus"],
    python_packages=["fastapi", "uvicorn", "gradio"],
    apt_packages=["wget"],
    requirements_file="/Fooocus/requirements_versions.txt",
    startup_script=init_Fooocus,
)

stub = Stub()

# Define the web app
web_app = FastAPI()

@stub.function(image=web_image, keep_warm=1, container_idle_timeout=60 * 20)
@asgi_app()
def ui():
    """A simple Gradio interface around our Fooocus inference."""

    # Create an instance of the AsyncWorker
    worker = AsyncWorker()

    def predict(prompt):
        # Run the prediction in an event loop
        loop = asyncio.get_event_loop()
        image_path = loop.run_until_complete(worker.run(prompt))
        return image_path

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
