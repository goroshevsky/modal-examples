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
        # Assuming entry_with_update.py handles the prompt and generates an image
        # The following code is a placeholder and should be replaced with actual implementation
        try:
            output = subprocess.check_output(["python", "/Fooocus/entry_with_update.py", "--prompt", prompt], cwd="/Fooocus")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error in image generation: {e.stderr.decode()}")

        # The path where Fooocus saves the generated image needs to be provided here
        # This is a placeholder path and should be replaced with the actual path used by Fooocus
        # For now, we simulate the prediction process with a placeholder image path
        output_path = "/Fooocus/outputs"  # This path is based on the grep search results
        # Check if the output directory exists and contains the generated image
        if os.path.isdir(output_path):
            # Assuming the generated image is named after the prompt with spaces replaced by underscores and in PNG format
            generated_image_name = prompt.replace(" ", "_") + ".png"
            generated_image_path = os.path.join(output_path, generated_image_name)
            if os.path.isfile(generated_image_path):
                # Return the file path of the generated image
                return generated_image_path
            else:
                raise FileNotFoundError(f"Generated image not found: {generated_image_path}")
        else:
            raise FileNotFoundError(f"Output directory not found: {output_path}")

    iface = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(label="Enter your prompt"),
        outputs=gr.Image(label="Generated Image", type="filepath"),
        title="Fooocus Image Generation",
        description="Enter a prompt to generate an image.",
        theme="default"
    )

    from gradio.routes import mount_gradio_app
    return mount_gradio_app(app=web_app, blocks=iface, path="/")
