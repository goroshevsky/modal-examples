from pathlib import Path

import modal

from .common import stub
from .llm import Vicuna
from .transcriber import Whisper
from .tts import Tortoise

static_path = Path(__file__).with_name("frontend").resolve()


@stub.function(
    mounts=[modal.Mount.from_local_dir(static_path, remote_path="/assets")],
    container_idle_timeout=300,
    timeout=600,
)
@stub.asgi_app()
def web():
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from fastapi.staticfiles import StaticFiles

    web_app = FastAPI()
    transcriber = Whisper()
    llm = Vicuna()
    tts = Tortoise()

    @web_app.post("/transcribe")
    async def transcribe(request: Request):
        bytes = await request.body()
        result = transcriber.transcribe_segment.call(bytes)
        return result["text"]

    @web_app.post("/submit")
    async def submit(request: Request):
        body = await request.json()

        if "warm" in body:
            llm.generate.spawn("")
            tts.speak.spawn("")
            return

        def generate():
            audio_futs = []
            sentence = ""

            # for segment in llm.generate.call(body["input"], body["history"]):
            #     yield f"text: {segment}\n"
            #     sentence += segment
            #     if "." in sentence:
            #         prev_sentence, new_sentence = sentence.rsplit(".", 1)
            #         fut = tts.speak.spawn(prev_sentence)
            #         audio_futs.append(fut)
            #         sentence = new_sentence

            audio_futs.append(tts.speak.spawn("What is the meaning of life?"))
            audio_futs.append(tts.speak.spawn("What is the meaning of life?"))
            yield "text: temp\n"

            yield "text_done\n"

            print("done generating")
            for fut in audio_futs:
                audio = fut.get()
                bytes = audio.read()
                yield f"wav: {len(bytes):08}\n"
                yield bytes

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )

    web_app.mount("/", StaticFiles(directory="/assets", html=True))
    return web_app
