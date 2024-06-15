import modal

app = modal.App()


@app.cls()
class ComfyServer:
    def __init__(self, model_name: str, plugins: list[str]):
        self.model_name = model_name
        self.plugins = plugins

    @modal.build()
    def setup_disk(self):
        # Downloads the models and plugins to disk to be saved into the image
        return

    @modal.enter()
    def load_model_and_plugins_from_disk(self):
        # All plugins and models should be available on the image
        # We selectively only load the ones specified in self.plugins
        return

    @modal.method()
    def infer(self):
        return


@app.function()
@modal.web_endpoint()
def backend(model_name: str, plugins: list[str], gpu_request: str):
    ComfyServer.with_options(gpu=gpu_request)(model_name, plugins).infer()


@app.local_entrypoint()
def apply_config():
    # Hot pool config
    # modal run test_comfy.py would apply the config.
    # Recommended to keep it in code so you don't forget which params are kept warm.
    ComfyServer.with_options(gpu="A100", container_idle_timeout=300)("infer", ["plugin1", "plugin2"]).keep_warm(1)
    ComfyServer.with_options(gpu="A100")("infer", ["plugin2", "plugin3"]).keep_warm(1)
    ComfyServer.with_options(gpu="A100")("infer", ["plugin1"]).keep_warm(1)