"""Local text-to-speech smoke test: drives the bot's own `VoiceGenerator.generate`.

The clip is saved to ./data/speech.wav.
"""

import asyncio
from pathlib import Path

from openai import AsyncOpenAI
import logfire

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.cogs.gen_reply.generation import VoiceGenerator

config = LLMConfig()


def gen_speech(text: str) -> None:
    """Synthesizes `text` as a spoken reply and saves the clip to `./data/speech.wav`.

    Args:
        text (str): The reply text to speak.

    Raises:
        RuntimeError: Synthesis produced no clip; the message names why.
    """
    generator = VoiceGenerator(
        client=AsyncOpenAI(base_url=config.base_url, api_key=config.api_key),
        model_name=RuntimeModelCatalog().tts_model.name,
    )
    clip = asyncio.run(main=generator.generate(text=text, end_user_id="voice_dev"))
    if clip.audio is None:
        raise RuntimeError(f"Voice synthesis produced no clip: {clip.outcome}")
    output_path = Path("./data/speech.wav")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_bytes(data=clip.audio)


if __name__ == "__main__":
    # The generator reports a failed synthesis only as a log record, so print records here.
    logfire.configure(send_to_logfire=False)
    gen_speech(text="為何 37 是質數?")
