"""Tests for the media generators and the prompt director."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
import asyncio

# openai 3.x builds its exceptions on httpx2, so a request or response handed to one has to
# come from there. google-genai is still on httpx 0.x, and both live in the environment.
import httpx2
from openai import APITimeoutError
import pytest
from google.genai.types import FileState

from discordbot.typings.media import LoadedMedia
from discordbot.typings.models import ModelSettings, RuntimeModelCatalog
from discordbot.cogs.gen_reply.prompts import IMAGE_PROMPT
from discordbot.cogs.gen_reply.generation import (
    MusicGenerator,
    VideoGenerator,
    VoiceGenerator,
    PromptGenerator,
    music_filename,
    speechify_discord_markup,
)

from tests.helpers.gen_reply import FakeGeminiClient
from tests.helpers.reply_fakes import FakeClient, _stalled, _png_bytes

if TYPE_CHECKING:
    from openai.types.responses.response_input_param import ResponseInputParam


def _recorded_content_parts(
    request: ResponseInputParam | str, index: int = 0
) -> list[dict[str, Any]]:
    """Returns the content parts of one item of a recorded `responses.create` input.

    The recorder keeps the real `ResponseInputParam` annotation, a union of ~30 TypedDicts
    that no structural assertion can index into, while the recorded payloads are plain
    heterogeneous JSON. This is the single place that narrows them back to JSON.
    """
    assert not isinstance(request, str)
    item = cast("dict[str, Any]", request[index])
    parts = item["content"]
    assert isinstance(parts, list)
    return parts


def test_speechify_discord_markup_rewrites_and_drops() -> None:
    """Mentions resolve to names; emoji / timestamps drop; slash commands keep their words."""
    names = {239270225441193986: "小明", 42: "管理員", 7: "general"}

    def _resolve(target_id: int) -> str | None:
        return names.get(target_id)

    assert speechify_discord_markup(text="嗆爆 <@239270225441193986>", resolve_name=_resolve) == (
        "嗆爆 小明"
    )
    # Role and channel mentions resolve through the same snowflake lookup.
    assert speechify_discord_markup(text="<@&42> 去 <#7> 集合", resolve_name=_resolve) == (
        "管理員 去 general 集合"
    )
    # An unresolved mention is dropped, leaving no doubled space behind.
    assert speechify_discord_markup(text="哈囉 <@999> 你好", resolve_name=_resolve) == "哈囉 你好"
    # Custom emoji and timestamp tags are dropped; a slash-command reference keeps its words.
    assert speechify_discord_markup(text="讚啦 <:blobcheer:123>", resolve_name=_resolve) == "讚啦"
    assert speechify_discord_markup(
        text="活動在 <t:1700000000:F> 開始", resolve_name=_resolve
    ) == ("活動在 開始")
    assert (
        speechify_discord_markup(text="用 </play:456> 點歌", resolve_name=_resolve)
        == "用 play 點歌"
    )


async def test_music_filename_follows_returned_mime() -> None:
    """The attachment extension follows the returned audio mime, falling back to .mp3."""
    assert music_filename(mime_type="audio/wav").endswith(".wav")
    assert music_filename(mime_type="audio/mpeg").endswith(".mp3")
    assert music_filename(mime_type="audio/ogg").endswith(".ogg")
    assert music_filename(mime_type=None).endswith(".mp3")


async def test_music_generator_drops_clip_on_bad_audio_payload() -> None:
    """A non-decodable audio payload returns None instead of raising into the attach gather."""

    class _Interactions:
        async def create(self, **kwargs: object) -> SimpleNamespace:
            """Returns an interaction whose audio data cannot be base64-decoded."""
            del kwargs
            return SimpleNamespace(
                output_audio=SimpleNamespace(data="not-valid-base64-x", mime_type="audio/mpeg")
            )

    client = FakeGeminiClient(interactions=_Interactions())
    generator = MusicGenerator(client=client, music_model=RuntimeModelCatalog().music_model)

    # The decode failure is swallowed (best-effort), so the streamer's media gather is never aborted.
    assert await generator.generate(user_prompt="a calm beat") is None


async def test_video_generator_drops_clip_on_provider_error() -> None:
    """A provider error from render returns None instead of raising into the attach gather."""

    class _Interactions:
        async def create(self, **kwargs: object) -> object:
            """Raises as if the omni Interactions call failed."""
            del kwargs
            raise RuntimeError("omni unavailable")

    client = FakeGeminiClient(interactions=_Interactions())
    generator = VideoGenerator(client=client, video_model=RuntimeModelCatalog().video_model)

    # The failure is swallowed (best-effort), so the streamer's media gather is never aborted.
    assert await generator.generate(user_prompt="a wave at sunset") is None


class _FakeSpeechResponse:
    """Async binary-response stand-in exposing aread() like the OpenAI speech result."""

    def __init__(self, data: bytes) -> None:
        """Stores the audio bytes to return from aread()."""
        self._data = data

    async def aread(self) -> bytes:
        """Returns the preset audio bytes."""
        return self._data


class _FakeSpeech:
    """Records audio.speech.create calls and returns or raises a preset result."""

    def __init__(self, data: bytes = b"RIFFwav", error: Exception | None = None) -> None:
        """Stores the bytes to return and an optional error to raise."""
        self.data = data
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: object) -> _FakeSpeechResponse:
        """Records the call and returns the preset response or raises the preset error."""
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return _FakeSpeechResponse(self.data)


def _fake_audio_client(speech: _FakeSpeech) -> SimpleNamespace:
    """A minimal AsyncOpenAI stand-in exposing client.audio.speech.create."""
    return SimpleNamespace(audio=SimpleNamespace(speech=speech))


async def test_voice_generator_prepends_style_and_returns_bytes() -> None:
    """A normal reply renders to bytes with the style directive prepended to the input."""
    speech = _FakeSpeech(data=b"RIFFwav")
    synth = VoiceGenerator(client=_fake_audio_client(speech=speech), model_name="tts-test")

    clip = await synth.generate(text="閉嘴", end_user_id="tester")

    assert clip.outcome == "ok"
    assert clip.audio == b"RIFFwav"
    assert speech.calls[0]["input"].endswith("閉嘴")
    assert speech.calls[0]["input"] != "閉嘴"
    # The generator holds no model of its own; the name it is handed is the one dispatched.
    assert speech.calls[0]["model"] == "tts-test"
    # response_format is intentionally never sent (the proxy 500s on it).
    assert "response_format" not in speech.calls[0]


async def test_voice_generator_swallows_provider_errors() -> None:
    """A provider error reports ERROR with no audio so the reply stays text-only."""
    speech = _FakeSpeech(error=RuntimeError("boom"))
    synth = VoiceGenerator(client=_fake_audio_client(speech=speech), model_name="tts-test")

    clip = await synth.generate(text="嗆你", end_user_id="tester")

    assert clip.audio is None
    assert clip.outcome == "error"


async def test_voice_generator_reports_timeout() -> None:
    """A request timeout is reported as TIMEOUT so the caller can hint distinctly."""
    speech = _FakeSpeech(error=APITimeoutError(request=httpx2.Request("POST", "http://proxy")))
    synth = VoiceGenerator(client=_fake_audio_client(speech=speech), model_name="tts-test")

    clip = await synth.generate(text="嗆你", end_user_id="tester")

    assert clip.audio is None
    assert clip.outcome == "timeout"


async def test_voice_generator_gives_up_at_its_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """A synthesis call that outlasts the bound is given up on, whatever the SDK does inside it."""
    speech = _FakeSpeech(data=b"RIFFwav")
    monkeypatch.setattr(target=speech, name="create", value=_stalled(call=speech.create))
    monkeypatch.setattr("discordbot.cogs.gen_reply.generation.VOICE_TIMEOUT_SECONDS", 0.01)
    synth = VoiceGenerator(client=_fake_audio_client(speech=speech), model_name="tts-test")

    clip = await synth.generate(text="嗆你", end_user_id="tester")

    assert clip.audio is None
    assert clip.outcome == "timeout"


async def test_voice_generator_reports_blank_text_as_empty() -> None:
    """Text that is blank after stripping is reported as EMPTY without a provider call."""
    speech = _FakeSpeech(data=b"RIFFwav")
    synth = VoiceGenerator(client=_fake_audio_client(speech=speech), model_name="tts-test")

    clip = await synth.generate(text=" \n ", end_user_id="tester")

    assert clip.audio is None
    assert clip.outcome == "empty"
    assert speech.calls == []


# ---- prompt director (PromptGenerator) ----


async def test_prompt_generator_refines_with_grounding() -> None:
    """An enabled director expands the request and records model, instructions, and grounding tools."""
    client = FakeClient()
    client.responses.refine_output_text = "a rich, detailed scene"
    generator = PromptGenerator(client=client, prompt_model=RuntimeModelCatalog().fast_model)

    refined = await generator.refine(
        user_prompt="draw a cat", instructions=IMAGE_PROMPT, end_user_id="alice", enabled=True
    )

    assert refined == "a rich, detailed scene"
    assert client.responses.create_models == [RuntimeModelCatalog().fast_model.name]
    assert client.responses.create_streams == [False]
    assert client.responses.create_instructions == [IMAGE_PROMPT]
    assert client.responses.create_tools == [list(RuntimeModelCatalog().fast_model.tools)]
    # The raw request rides as the input_text part the director rewrites.
    request_text = _recorded_content_parts(request=client.responses.create_inputs[0])[0]["text"]
    assert "draw a cat" in request_text


async def test_prompt_generator_disabled_returns_raw_without_call() -> None:
    """A disabled director returns the raw prompt and never calls the model."""
    client = FakeClient()
    generator = PromptGenerator(client=client, prompt_model=RuntimeModelCatalog().fast_model)

    refined = await generator.refine(
        user_prompt="draw a cat", instructions=IMAGE_PROMPT, end_user_id="alice", enabled=False
    )

    assert refined == "draw a cat"
    assert client.responses.create_models == []


async def test_prompt_generator_empty_draft_falls_back_to_raw() -> None:
    """An empty draft (no output_text) falls back to the raw prompt."""
    client = FakeClient()  # refine_output_text defaults to None
    generator = PromptGenerator(client=client, prompt_model=RuntimeModelCatalog().fast_model)

    refined = await generator.refine(
        user_prompt="draw a cat", instructions=IMAGE_PROMPT, end_user_id="alice", enabled=True
    )

    assert refined == "draw a cat"


async def test_prompt_generator_error_falls_back_to_raw() -> None:
    """Any director error falls back to the raw prompt instead of raising into the route."""
    client = FakeClient()

    async def _boom(*args: object, **kwargs: object) -> object:
        """Fails the director call."""
        del args, kwargs
        raise RuntimeError("director boom")

    client.responses.__dict__["create"] = _boom  # instance attr shadows the recorder method
    generator = PromptGenerator(client=client, prompt_model=RuntimeModelCatalog().fast_model)

    refined = await generator.refine(
        user_prompt="draw a cat", instructions=IMAGE_PROMPT, end_user_id="alice", enabled=True
    )

    assert refined == "draw a cat"


async def test_prompt_generator_rides_source_images_as_input() -> None:
    """Source bytes ride along as input_image parts so an edit draft is grounded in the picture."""
    client = FakeClient()
    client.responses.refine_output_text = "edited result"
    generator = PromptGenerator(client=client, prompt_model=RuntimeModelCatalog().fast_model)

    await generator.refine(
        user_prompt="make it blue",
        instructions=IMAGE_PROMPT,
        end_user_id="alice",
        enabled=True,
        image_bytes_list=[_png_bytes()],
    )

    director_content = _recorded_content_parts(request=client.responses.create_inputs[0])
    assert director_content[0]["type"] == "input_text"
    assert any(part.get("type") == "input_image" for part in director_content)


async def test_download_output_video_retries_until_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """A URI-delivered clip whose first download fails (file still finalizing) is retried."""
    calls = {"n": 0}

    async def flaky_download(file: object) -> bytes:
        """Fails the first download (file not yet servable), then succeeds."""
        del file
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("file not ready")
        return b"mp4"

    async def fast_sleep(delay: float) -> None:
        """Skips the retry backoff."""
        del delay

    monkeypatch.setattr("discordbot.cogs.gen_reply.generation.asyncio.sleep", fast_sleep)
    client = FakeGeminiClient(files=SimpleNamespace(download=flaky_download))
    generator = VideoGenerator(
        client=client, video_model=ModelSettings(name="gemini-omni-flash-preview")
    )

    result = await generator._download_output_video(uri="https://files.test/v:download?alt=media")

    assert result == b"mp4"
    assert calls["n"] == 2


async def test_a_stalled_clip_download_fails_the_video_within_its_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A download that never returns fails the render at the bound instead of hanging it."""

    async def download(file: object) -> bytes:
        """Returns the clip, once the stall lets it."""
        del file
        return b"mp4"

    monkeypatch.setattr("discordbot.cogs.gen_reply.generation.FILES_READY_TIMEOUT_SECONDS", 0.01)
    client = FakeGeminiClient(files=SimpleNamespace(download=_stalled(call=download)))
    generator = VideoGenerator(
        client=client, video_model=ModelSettings(name="gemini-omni-flash-preview")
    )

    with pytest.raises(TimeoutError):
        await generator._download_output_video(uri="https://files.test/v:download?alt=media")


async def test_a_never_servable_clip_fails_with_the_download_error_at_the_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clip that keeps refusing to download fails with that refusal, not a bare timeout."""

    async def download(file: object) -> bytes:
        """Refuses the download the way a file that is not servable yet does, after a round trip."""
        del file
        await asyncio.sleep(0)
        raise RuntimeError("404 NOT_FOUND: file is not servable yet")

    monkeypatch.setattr("discordbot.cogs.gen_reply.generation.FILES_READY_TIMEOUT_SECONDS", 0.05)
    client = FakeGeminiClient(files=SimpleNamespace(download=download))
    generator = VideoGenerator(
        client=client, video_model=ModelSettings(name="gemini-omni-flash-preview")
    )

    with pytest.raises(RuntimeError, match="not servable yet"):
        await generator._download_output_video(uri="https://files.test/v:download?alt=media")


async def test_a_stalled_source_video_upload_fails_the_edit_within_its_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An edit's source upload that never returns fails as one that never became ACTIVE."""

    async def upload(file: object, config: dict[str, str]) -> SimpleNamespace:
        """Returns the uploaded clip ACTIVE, once the stall lets it."""
        del file, config
        return SimpleNamespace(
            name="files/vid", uri="https://files.test/vid", state=FileState.ACTIVE
        )

    monkeypatch.setattr("discordbot.cogs.gen_reply.generation.FILES_READY_TIMEOUT_SECONDS", 0.01)
    client = FakeGeminiClient(files=SimpleNamespace(upload=_stalled(call=upload)))
    generator = VideoGenerator(
        client=client, video_model=ModelSettings(name="gemini-omni-flash-preview")
    )

    with pytest.raises(RuntimeError) as raised:
        await generator._upload_source_video(
            source_video=LoadedMedia(data=b"clip", mime_type="video/mp4")
        )
    assert isinstance(raised.value.__cause__, TimeoutError)
