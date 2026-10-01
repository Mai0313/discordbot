"""Local text-to-speech smoke test: turns one line of text into ./data/speech.mp3."""

from pathlib import Path

from openai import OpenAI

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import RuntimeModelCatalog

config = LLMConfig()

TTS_MODEL = RuntimeModelCatalog().tts_model


def gen_speech(text: str) -> None:
    """Synthesizes `text` through LiteLLM and saves the clip to `./data/speech.mp3`.

    Args:
        text (str): The line to speak, sent verbatim.
    """
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    audio_responses = client.audio.speech.create(
        input=text,
        model=TTS_MODEL.name,
        voice="Zephyr",
        instructions="",
        speed=1.3,
        extra_headers={"x-litellm-end-user-id": "voice_dev"},
    )
    output_path = Path("./data/speech.mp3")
    output_path.parent.mkdir(exist_ok=True)
    audio_responses.write_to_file(file=output_path)


if __name__ == "__main__":
    gen_speech(text="為何 37 是質數?")
