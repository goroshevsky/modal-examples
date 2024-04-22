# Generate: Fooocus
#
# This example is similar to the A1111 example, but it uses Fooocus as frontend. Fooocus is geared towards beginners to get started with SDXL 1.0.
#
# SDXL model is predownloaded to avoid longer cold-starts. However it will download some smaller models (vae etc) before starting.

# Basic setup

from modal import Image, Stub, gpu, web_server, asgi_app
from fastapi import FastAPI

DOCKER_IMAGE = "nvidia/cuda:12.3.1-base-ubuntu22.04"
PYTHON_VER = "3.10"
GPU_CONFIG = gpu.T4()
PORT = 8000

# Initialize Fooocus
#
# Install requirements and download SDXL 1.0 model as part of Modal image to avoid large downloads during cold-starts.

def init_Fooocus():
    import os
    import subprocess

    os.chdir("/Fooocus")
    os.system("pip install -r requirements_versions.txt")
    os.chdir("./models/checkpoints")
    subprocess.run("wget -O juggernautXL_v8Rundiffusion.safetensors 'https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_v8Rundiffusion.safetensors'", shell=True)

# Define container image
#
# Install essentials and setup Fooocus repository.

image = (
    Image.from_registry(DOCKER_IMAGE, add_python=PYTHON_VER)
    .run_commands("apt update -y")
    .apt_install(
        "software-properties-common",
        "git",
        "git-lfs",
        "coreutils",
        "aria2",
        "libgl1",
        "libglib2.0-0",
        'curl',
        "wget",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
    )
    .run_commands("git clone https://github.com/lllyasviel/Fooocus.git")
    .pip_install("pygit2==1.12.2")
    .run_function(init_Fooocus, gpu=GPU_CONFIG)
)

# Define web interface image
web_image = Image.debian_slim().pip_install("gradio", "fastapi", "uvicorn")

# Run Fooocus

stub = Stub("Fooocus", image=image)

@stub.function(gpu=GPU_CONFIG, timeout=60 * 10) # Set GPU configuration and function timeout
@web_server(port=PORT, startup_timeout=180)
def run():
    import os
    import subprocess

    os.chdir("/Fooocus")
    subprocess.Popen(
        [
            "python",
            "launch.py",
            "--listen",
            "0.0.0.0",
            "--port",
            "8000",
            "--always-high-vram"
        ]
    )

# Define the web app
web_app = FastAPI()

@app.function(image=web_image, keep_warm=1, container_idle_timeout=60 * 20)
@asgi_app()
def ui():
    """A simple Gradio interface around our Fooocus inference."""
    import gradio as gr

    def predict(prompt):
        # This function will call the Fooocus model and return the image.
        # Placeholder for actual Fooocus model prediction logic.
        pass

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
