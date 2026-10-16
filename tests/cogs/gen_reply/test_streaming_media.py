"""Tests for the media an answer's inline markers attach to its reply."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
import asyncio
from datetime import datetime, timedelta
from collections import Counter

import pytest
import nextcord

from discordbot.typings.media import LoadedMedia
from discordbot.utils.llm_transcript import USAGE_FOOTER_RE
from discordbot.utils.media_delivery import MediaItem
from discordbot.cogs.gen_reply.markers import MAX_INLINE_IMAGES, InlineMarkers
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.cogs.gen_reply.streaming import (
    DISCORD_MESSAGE_LIMIT,
    ResponseStreamer,
    stream_answer_with_retry,
)
from discordbot.cogs.gen_reply.generation import (
    MusicClip,
    VoiceClip,
    VoiceOutcome,
    ImageGenerator,
    MusicGenerator,
    VideoGenerator,
    VoiceGenerator,
)

from tests.helpers.casting import as_message, make_not_found
from tests.helpers.reply_fakes import (
    FakeGuild,
    FakeReply,
    FakeMessage,
    _cog,
    _stream,
    _streamer,
    _text_event,
    _media_routes,
    _stream_events,
    _completed_event,
    _expiring_surface,
    _stream_events_from,
)
from tests.helpers.link_sources import hosting_planner, hosting_off_planner
from tests.helpers.logfire_capture import capture_logs

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import AsyncIterator

    from nextcord.ext import commands
    from openai.types.responses import ResponseStreamEvent

    from discordbot.cogs.gen_reply.input import MessageInputBuilder


async def test_deleted_reply_skips_media_attach_without_hint() -> None:
    """Media requested on a since-deleted reply is dropped silently, with no ⚠️ on the source."""
    message = FakeMessage()
    reply = FakeReply()
    reply.edit_error = make_not_found()
    synthesizer = _FakeVoiceGenerator()

    await _stream(
        streamer=_streamer(
            message=message, reply=reply, voice_generator=cast("VoiceGenerator", synthesizer)
        ),
        responses=_stream_events_from(events=_voice_marker_events()),
    )

    assert synthesizer.calls == []
    assert message.added_reactions == []


@pytest.mark.parametrize(
    argnames=("markers", "hinted"),
    argvalues=[
        (InlineMarkers(cleaned_text="", voice_requested=True), True),
        (InlineMarkers(cleaned_text="", image_prompts=["a cat"]), True),
        (InlineMarkers(cleaned_text="", music_prompt="a song"), True),
        (InlineMarkers(cleaned_text="", video_prompt="a clip"), True),
        (InlineMarkers(cleaned_text=""), False),
    ],
    ids=["voice", "image", "music", "video", "none"],
)
async def test_media_a_never_sent_reply_asked_for_leaves_a_hint(
    markers: InlineMarkers, hinted: bool
) -> None:
    """With no reply to attach to, any requested media is dropped with a ⚠️; none, no hint."""
    message = FakeMessage()
    streamer = _streamer(message=message)
    streamer.markers = markers

    await streamer._attach_generated_media()

    assert message.added_reactions == (["⚠️"] if hinted else [])


# ---- voice (spoken reply) ----


class _FakeVoiceGenerator:
    """Records generate calls and returns a configurable VoiceClip for streamer voice tests."""

    def __init__(
        self, audio: bytes | None = b"RIFFfake-wav", outcome: VoiceOutcome = "ok"
    ) -> None:
        """Stores the audio bytes (None to simulate failure) and the reported outcome."""
        self.audio = audio
        self.outcome = outcome
        self.calls: list[dict[str, str]] = []

    async def generate(self, text: str, end_user_id: str) -> VoiceClip:
        """Records the spoken-text request and returns the preset VoiceClip."""
        self.calls.append({"text": text, "end_user_id": end_user_id})
        return VoiceClip(audio=self.audio, outcome=self.outcome)


def _voice_marker_events() -> list[SimpleNamespace]:
    """A single-turn stream whose reply wraps one segment in <generate-voice> tags."""
    return [
        _text_event(delta="閉嘴啦白痴 "),
        _text_event(delta="<generate-voice>嗆爆你</generate-voice>"),
        _text_event(delta=" 滾"),
        _completed_event(input_tokens=3, output_tokens=4),
    ]


def _assert_no_voice_tags(text: str) -> None:
    """Asserts neither voice tag leaked into the visible reply."""
    assert "<generate-voice>" not in text
    assert "</generate-voice>" not in text


async def test_voice_marker_triggers_synthesis_and_strips_tag() -> None:
    """A <generate-voice> segment is spoken (only that part), its tags stripped, the clip attached."""
    message = FakeMessage()
    synthesizer = _FakeVoiceGenerator()

    result = await _stream(
        streamer=_streamer(message=message, voice_generator=cast("VoiceGenerator", synthesizer)),
        responses=_stream_events_from(events=_voice_marker_events()),
    )

    _assert_no_voice_tags(result)
    # The wrapped content stays visible alongside the rest of the reply.
    assert "嗆爆你" in result
    assert "閉嘴啦白痴" in result
    # Only the wrapped segment (not the whole reply) is spoken.
    assert synthesizer.calls == [{"text": "嗆爆你", "end_user_id": message.author.name}]
    assert message.replies[0].file is not None
    assert message.replies[0].file.filename == "reply.wav"
    # The source message is marked with the voice app emoji while the clip is produced.
    assert message.added_reactions == ["<:voice:1517558121092878376>"]


async def test_voice_marker_absent_no_synthesis() -> None:
    """A normal reply (no <generate-voice>) never calls the synthesizer and attaches no file."""
    message = FakeMessage()
    synthesizer = _FakeVoiceGenerator()

    await _stream(
        streamer=_streamer(message=message, voice_generator=cast("VoiceGenerator", synthesizer)),
        responses=_stream_events(),
    )

    assert synthesizer.calls == []
    assert message.replies[0].file is None
    # The model chose no voice, so there is nothing to hint about.
    assert message.added_reactions == []


async def test_voice_disabled_still_strips_marker() -> None:
    """With no synthesizer (voice off) the tags are still stripped and no file attaches."""
    message = FakeMessage()

    result = await _stream(
        streamer=_streamer(message=message),
        responses=_stream_events_from(events=_voice_marker_events()),
    )

    _assert_no_voice_tags(result)
    assert "嗆爆你" in result
    assert message.replies[0].file is None


async def test_voice_synthesis_failure_leaves_text_reply() -> None:
    """A synthesis error leaves a clean text reply, no file, and hints with a warning emoji."""
    message = FakeMessage()
    synthesizer = _FakeVoiceGenerator(audio=None, outcome="error")

    result = await _stream(
        streamer=_streamer(message=message, voice_generator=cast("VoiceGenerator", synthesizer)),
        responses=_stream_events_from(events=_voice_marker_events()),
    )

    _assert_no_voice_tags(result)
    assert message.replies[0].file is None
    # The voice marker is added before synth; a non-timeout failure then hints with the warning.
    assert message.added_reactions == ["<:voice:1517558121092878376>", "⚠️"]


async def test_voice_synthesis_timeout_hints_with_clock() -> None:
    """A synthesis timeout leaves a text reply and hints with the clock emoji, staying silent."""
    message = FakeMessage()
    synthesizer = _FakeVoiceGenerator(audio=None, outcome="timeout")

    result = await _stream(
        streamer=_streamer(message=message, voice_generator=cast("VoiceGenerator", synthesizer)),
        responses=_stream_events_from(events=_voice_marker_events()),
    )

    _assert_no_voice_tags(result)
    assert message.replies[0].file is None
    assert message.added_reactions == ["<:voice:1517558121092878376>", "⏱️"]


async def test_voice_with_nothing_to_say_leaves_no_hint() -> None:
    """An empty clip is not a failure: no file attaches and nothing follows the voice mark."""
    message = FakeMessage()
    synthesizer = _FakeVoiceGenerator(audio=None, outcome="empty")

    await _stream(
        streamer=_streamer(message=message, voice_generator=cast("VoiceGenerator", synthesizer)),
        responses=_stream_events_from(events=_voice_marker_events()),
    )

    assert message.replies[0].file is None
    assert message.added_reactions == ["<:voice:1517558121092878376>"]


async def test_voice_too_big_falls_back_to_hosted_url(tmp_path: Path) -> None:
    """A voice clip past the upload limit is hosted and its URL appended, not silently dropped."""
    message = FakeMessage()
    # 4-byte ceiling so the fake WAV (larger) exceeds it, like a long WAV in a 20 MiB DM.
    message.guild = FakeGuild(filesize_limit=4)
    synthesizer = _FakeVoiceGenerator()

    result = await _stream(
        streamer=_streamer(
            message=message,
            voice_generator=cast("VoiceGenerator", synthesizer),
            media_delivery=hosting_planner(serve_dir=tmp_path),
        ),
        responses=_stream_events_from(events=_voice_marker_events()),
    )

    _assert_no_voice_tags(result)
    # The clip was hosted, not attached; its URL (a .wav) rides the reply content instead.
    assert message.replies[0].file is None
    content = message.replies[0].content or ""
    assert any(line.startswith("https://media.test/") for line in content.splitlines())
    assert ".wav" in content
    # The hosted link rides BEFORE the usage footer, so USAGE_FOOTER_RE still strips the footer from
    # later history; the link must survive that strip, and the footer must not (else it would leak
    # the model/token/cost line into the bot's answer in history / memory).
    assert USAGE_FOOTER_RE.search(content) is not None
    stripped = USAGE_FOOTER_RE.sub("", content)
    assert "media.test" in stripped
    assert "⬆" not in stripped
    # The media edit that appended the URL must carry AllowedMentions.none() so the already-pinged
    # author is never re-pinged; a regression dropping the kwarg would record None here.
    assert message.replies[0].allowed_mentions_seen[-1] is not None


async def test_finalize_media_edit_posts_followup_when_content_would_overflow() -> None:
    """A hosted URL on an already-near-2000-char reply rides a follow-up, not the main edit."""
    streamer = _streamer(message=FakeMessage())
    reply = FakeReply()
    streamer.reply = as_message(fake=reply)
    streamer.stored_content = "x" * (DISCORD_MESSAGE_LIMIT - 10)

    await streamer._finalize_media_edit(
        reply=as_message(fake=reply), files=[], hosted_urls=["https://media.test/abc.wav"]
    )

    # The URL did not fit the main content, so it was posted as a follow-up reply (which must keep
    # AllowedMentions.none()), and the parent content was left unchanged.
    assert "media.test" not in (reply.content or "")
    assert any("media.test" in (child.content or "") for child in reply.replies)
    assert reply.allowed_mentions_seen[-1] is not None


async def test_finalize_media_edit_hints_when_the_hosted_followup_fails() -> None:
    """A follow-up that never lands is the whole clip, so it earns the ⚠️ hint, not silence."""
    message = FakeMessage()
    streamer = _streamer(message=message)
    reply = FakeReply()
    reply.reply_error = RuntimeError("follow-up refused")
    streamer.reply = as_message(fake=reply)
    streamer.stored_content = "x" * (DISCORD_MESSAGE_LIMIT - 10)

    await streamer._finalize_media_edit(
        reply=as_message(fake=reply), files=[], hosted_urls=["https://media.test/abc.wav"]
    )

    assert reply.replies == []
    assert "⚠️" in message.added_reactions


@pytest.mark.parametrize("refused", [False, True], ids=["attached", "refused"])
async def test_only_a_landed_media_attach_is_logged_as_attached(
    monkeypatch: pytest.MonkeyPatch, refused: bool
) -> None:
    """The step's closing line says the media landed, so a refused edit must not reach it.

    Discord refuses the edit for inline music in a guild that limits uploads.
    """
    message = FakeMessage()
    reply = FakeReply()
    if refused:
        reply.edit_error = RuntimeError("file uploads are limited here")
    streamer = _streamer(message=message, reply=as_message(fake=reply))

    async def voice_clip() -> MediaItem:
        """Stands in for a synthesized clip ready to attach."""
        return MediaItem(source=b"RIFF", filename="reply.wav")

    monkeypatch.setattr(streamer, "_build_voice_candidate", voice_clip)
    logged = capture_logs(monkeypatch=monkeypatch, level="info")

    await streamer._attach_generated_media()

    assert ("⚠️" in message.added_reactions) is refused
    assert ("Generated media attached" in [text for text, _ in logged]) is not refused


class _AskInteraction:
    """The `/ask` interaction as far as landing one media reply needs it."""

    @property
    def expires_at(self) -> datetime:
        """A token with the whole window still ahead of it."""
        return nextcord.utils.utcnow() + timedelta(minutes=15)

    async def edit_original_message(self, **kwargs: Any) -> FakeReply:  # noqa: ANN401 -- Discord kwargs
        """Lands the media as the original response."""
        del kwargs
        return FakeReply()


async def test_ask_media_is_sized_by_the_interaction_it_goes_out_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the interaction carries the invoker's Nitro allowance, and both media paths use it.

    The synthesized `/ask` message has no guild, so a ceiling read off it alone is the 20 MiB base.
    """
    seen: list[object] = []

    def record_ceiling(guild: object, interaction: object = None) -> int:
        """Answers a Nitro-sized ceiling and records which interaction asked."""
        seen.append(interaction)
        return 1024**3

    monkeypatch.setattr("discordbot.cogs.gen_reply.media_reply.upload_limit_for", record_ceiling)
    monkeypatch.setattr("discordbot.cogs.gen_reply.streaming.upload_limit_for", record_ceiling)
    interaction = _AskInteraction()
    message = FakeMessage()
    surface = TurnSurface(
        message=as_message(fake=message),
        interaction=cast("nextcord.Interaction[commands.Bot]", interaction),
    )
    streamer = ResponseStreamer(
        message=message,
        surface=surface,
        media_delivery=hosting_off_planner(),
        reply=as_message(fake=FakeReply()),
    )

    async def voice_clip() -> MediaItem:
        """Stands in for a synthesized clip ready to attach."""
        return MediaItem(source=b"RIFF", filename="reply.wav")

    monkeypatch.setattr(streamer, "_build_voice_candidate", voice_clip)

    await _media_routes(cog=_cog(), message=as_message(fake=message), surface=surface)._deliver(
        data=b"PNG", filename="image.png"
    )
    await streamer._attach_generated_media()

    assert Counter(seen) == Counter([interaction, interaction])


async def test_media_past_the_attachment_cap_earns_the_dropped_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every inline marker at once outnumbers one message's attachments; the overflow is a loss.

    The planner reports it apart from a hosting failure, which this site treats alike.
    """
    message = FakeMessage()
    reply = FakeReply()
    streamer = _streamer(message=message, reply=as_message(fake=reply))

    async def clip(name: str) -> MediaItem:
        """Stands in for one rendered clip ready to attach."""
        return MediaItem(source=b"clip", filename=name)

    async def images(source_images_task: object) -> list[MediaItem]:
        """Stands in for every inline image a reply may carry."""
        del source_images_task
        return [
            MediaItem(source=b"png", filename=f"generated_{index}.png")
            for index in range(MAX_INLINE_IMAGES)
        ]

    monkeypatch.setattr(streamer, "_build_voice_candidate", lambda: clip(name="reply.wav"))
    monkeypatch.setattr(streamer, "_build_music_candidate", lambda: clip(name="music.mp3"))
    monkeypatch.setattr(
        streamer, "_build_video_candidate", lambda source_images_task: clip(name="video.mp4")
    )
    monkeypatch.setattr(streamer, "_build_image_candidates", images)

    await streamer._attach_generated_media()

    assert reply.files is not None
    assert len(reply.files) == 10
    assert "⚠️" in message.added_reactions


async def test_ask_media_still_rendering_when_the_window_ends_is_dropped_with_a_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clip still rendering when `/ask` runs out is given up while the ⏱️ can still be written.

    Past the token both the attach edit and the hint line 404, so a render left running loses the
    clip with nothing to say it was ever coming (#628). What finished in time still goes out.
    """
    message = FakeMessage()
    reply = FakeReply()
    surface = _expiring_surface(message=as_message(fake=message), seconds_left=0.05)
    streamer = ResponseStreamer(
        message=message,
        surface=surface,
        media_delivery=hosting_off_planner(),
        reply=as_message(fake=reply),
    )

    async def voice_clip() -> MediaItem:
        """Stands in for a synthesized clip ready to attach."""
        return MediaItem(source=b"RIFF", filename="reply.wav")

    async def never_finishes(source_images_task: object) -> MediaItem:
        """Stands in for a video render that outlasts the window."""
        del source_images_task
        await asyncio.sleep(60)
        return MediaItem(source=b"mp4", filename="video.mp4")

    monkeypatch.setattr(streamer, "_build_voice_candidate", voice_clip)
    monkeypatch.setattr(streamer, "_build_video_candidate", never_finishes)

    await asyncio.wait_for(streamer._attach_generated_media(), timeout=5)

    assert reply.files is not None
    assert [file.filename for file in reply.files] == ["reply.wav"]
    assert surface.take_hints() == ["⏱️"]


async def test_a_cancelled_turn_takes_its_inline_media_renders_with_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A render must not outlive the turn that wanted it, as it could not under `gather`."""
    message = FakeMessage()
    streamer = _streamer(message=message, reply=as_message(fake=FakeReply()))
    rendering = asyncio.Event()
    stopped = asyncio.Event()

    async def never_finishes(source_images_task: object) -> MediaItem:
        """Stands in for a video render, noting when it is stopped."""
        del source_images_task
        rendering.set()
        try:
            await asyncio.sleep(60)
        finally:
            stopped.set()
        return MediaItem(source=b"mp4", filename="video.mp4")

    monkeypatch.setattr(streamer, "_build_video_candidate", never_finishes)
    turn = asyncio.ensure_future(streamer._attach_generated_media())
    await rendering.wait()
    turn.cancel()

    with pytest.raises(asyncio.CancelledError):
        await turn
    await asyncio.wait_for(stopped.wait(), timeout=5)


def _voice_marker_mention_events() -> list[SimpleNamespace]:
    """A stream whose <generate-voice> segment contains a raw user mention."""
    return [
        _text_event(delta="<generate-voice>嗆爆 <@239270225441193986></generate-voice>"),
        _completed_event(input_tokens=3, output_tokens=4),
    ]


async def test_voice_text_strips_discord_markup() -> None:
    """The spoken clip narrates the resolved name while the visible reply keeps the mention."""
    message = FakeMessage()
    message.guild = FakeGuild(members={239270225441193986: SimpleNamespace(display_name="小明")})
    synthesizer = _FakeVoiceGenerator()

    result = await _stream(
        streamer=_streamer(message=message, voice_generator=cast("VoiceGenerator", synthesizer)),
        responses=_stream_events_from(events=_voice_marker_mention_events()),
    )

    # The visible reply keeps the clickable mention; only the spoken text is normalised.
    assert "<@239270225441193986>" in result
    assert synthesizer.calls == [{"text": "嗆爆 小明", "end_user_id": message.author.name}]


# ---- inline image (<generate-image>) ----


class _FakeImageGenerator:
    """Records generate calls and returns configurable PNG bytes for streamer image tests."""

    def __init__(self, image: bytes | None = b"\x89PNG-fake") -> None:
        """Stores the PNG bytes (None to simulate a failed render) returned by generate."""
        self.image = image
        self.calls: list[dict[str, str]] = []
        self.image_bytes_lists: list[list[bytes] | None] = []

    async def generate(
        self, user_prompt: str, end_user_id: str, image_bytes_list: list[bytes] | None = None
    ) -> bytes | None:
        """Records the description request (and any edit source bytes) and returns the image."""
        self.calls.append({"user_prompt": user_prompt, "end_user_id": end_user_id})
        self.image_bytes_lists.append(image_bytes_list)
        return self.image


def _image_marker_events() -> list[SimpleNamespace]:
    """A single-turn stream whose reply wraps an <generate-image> description."""
    return [
        _text_event(delta="這是你要的圖 "),
        _text_event(delta="<generate-image>a cute black cat</generate-image>"),
        _completed_event(input_tokens=3, output_tokens=4),
    ]


async def test_image_marker_generates_and_attaches() -> None:
    """An <generate-image> block is pulled from the reply, rendered, and the PNG attached to the reply."""
    message = FakeMessage()
    generator = _FakeImageGenerator()

    result = await _stream(
        streamer=_streamer(message=message, image_generator=cast("ImageGenerator", generator)),
        responses=_stream_events_from(events=_image_marker_events()),
    )

    # The block (tags AND description) never shows in chat.
    assert "<generate-image>" not in result
    assert "a cute black cat" not in result
    assert "這是你要的圖" in result
    # The rough description is handed to the generator and the PNG attached afterward.
    assert generator.calls == [
        {"user_prompt": "a cute black cat", "end_user_id": message.author.name}
    ]
    assert message.replies[0].file is not None
    assert message.replies[0].file.filename == "generated.png"
    # The source message is marked with the image app emoji while the image is rendered.
    assert message.added_reactions == ["<:image:1517559727880667226>"]
    # No input_builder wired -> no source bytes -> a plain generation (not an edit).
    assert generator.image_bytes_lists == [None]


async def test_image_marker_edits_uploaded_image_with_source_bytes() -> None:
    """An uploaded image rides into the inline <generate-image> render as edit source, without refinement."""
    message = FakeMessage()
    generator = _FakeImageGenerator()

    async def _load(message: object, replied_to: object) -> list[LoadedMedia]:
        """Stands in for the input builder loading the message's uploaded image."""
        del message, replied_to
        return [LoadedMedia(data=b"uploaded-bytes", mime_type="image/png")]

    builder = SimpleNamespace(get_turn_image_sources=_load)

    await _stream(
        streamer=_streamer(
            message=message,
            image_generator=cast("ImageGenerator", generator),
            input_builder=cast("MessageInputBuilder", builder),
        ),
        responses=_stream_events_from(events=_image_marker_events()),
    )

    # The uploaded bytes (mime stripped for the edit path) ride through to generate, so the inline
    # <generate-image> edits them.
    assert generator.image_bytes_lists == [[b"uploaded-bytes"]]
    # The marker description itself is passed through verbatim (the marker path never refines).
    assert generator.calls == [
        {"user_prompt": "a cute black cat", "end_user_id": message.author.name}
    ]


async def test_image_disabled_still_strips_marker() -> None:
    """With no generator (inline image off) the block is still pulled and no file attaches."""
    message = FakeMessage()

    result = await _stream(
        streamer=_streamer(message=message),
        responses=_stream_events_from(events=_image_marker_events()),
    )

    assert "<generate-image>" not in result
    assert "a cute black cat" not in result
    assert message.replies[0].file is None


async def test_image_generation_failure_hints() -> None:
    """A failed render leaves a clean text reply with no file and a warning hint."""
    message = FakeMessage()
    generator = _FakeImageGenerator(image=None)

    result = await _stream(
        streamer=_streamer(message=message, image_generator=cast("ImageGenerator", generator)),
        responses=_stream_events_from(events=_image_marker_events()),
    )

    assert "a cute black cat" not in result
    assert message.replies[0].file is None
    assert message.added_reactions == ["<:image:1517559727880667226>", "⚠️"]


async def test_multiple_image_markers_attach_distinct_files() -> None:
    """Several <generate-image> blocks each render and attach under distinct filenames in one edit."""
    message = FakeMessage()
    generator = _FakeImageGenerator()

    result = await _stream(
        streamer=_streamer(message=message, image_generator=cast("ImageGenerator", generator)),
        responses=_stream_events_from(
            events=[
                _text_event(delta="兩張圖 "),
                _text_event(
                    delta="<generate-image>a red cat</generate-image><generate-image>a blue dog</generate-image>"
                ),
                _completed_event(input_tokens=3, output_tokens=4),
            ]
        ),
    )

    assert "<generate-image>" not in result
    # Each description renders independently, in order.
    assert [call["user_prompt"] for call in generator.calls] == ["a red cat", "a blue dog"]
    files = message.replies[0].files
    assert files is not None
    assert [item.filename for item in files] == ["generated_1.png", "generated_2.png"]


async def test_image_markers_capped_at_limit() -> None:
    """More <generate-image> blocks than the per-reply cap render only up to MAX_INLINE_IMAGES."""
    message = FakeMessage()
    generator = _FakeImageGenerator()
    blocks = "".join(
        f"<generate-image>image {index}</generate-image>" for index in range(MAX_INLINE_IMAGES + 3)
    )

    await _stream(
        streamer=_streamer(message=message, image_generator=cast("ImageGenerator", generator)),
        responses=_stream_events_from(
            events=[
                _text_event(delta=f"好多圖 {blocks}"),
                _completed_event(input_tokens=3, output_tokens=4),
            ]
        ),
    )

    # Only the first MAX_INLINE_IMAGES render and attach; the extra blocks are dropped.
    assert len(generator.calls) == MAX_INLINE_IMAGES
    files = message.replies[0].files
    assert files is not None
    assert len(files) == MAX_INLINE_IMAGES


# ---- inline music (<generate-music>) ----


class _FakeMusicGenerator:
    """Records generate calls and returns a configurable MusicClip (or None) for streamer tests."""

    def __init__(
        self, audio: bytes | None = b"ID3-fake-mp3", mime_type: str = "audio/mp3"
    ) -> None:
        """Stores the clip (None audio simulates a failed render) returned by generate."""
        self.clip = MusicClip(audio=audio, mime_type=mime_type) if audio is not None else None
        self.calls: list[str] = []

    async def generate(self, user_prompt: str) -> MusicClip | None:
        """Records the music description request and returns the preset clip."""
        self.calls.append(user_prompt)
        return self.clip


def _music_marker_events() -> list[SimpleNamespace]:
    """A single-turn stream whose reply wraps a <generate-music> description."""
    return [
        _text_event(delta="這首給你 "),
        _text_event(delta="<generate-music>upbeat anime J-pop, female vocals</generate-music>"),
        _completed_event(input_tokens=3, output_tokens=4),
    ]


async def test_music_marker_generates_and_attaches() -> None:
    """A <generate-music> block is pulled from the reply, generated, and the clip attached to the reply."""
    message = FakeMessage()
    generator = _FakeMusicGenerator()

    result = await _stream(
        streamer=_streamer(message=message, music_generator=cast("MusicGenerator", generator)),
        responses=_stream_events_from(events=_music_marker_events()),
    )

    # The block (tags AND description) never shows in chat.
    assert "<generate-music>" not in result
    assert "anime" not in result
    assert "這首給你" in result
    # The description is handed to the generator and the clip attached afterward.
    assert generator.calls == ["upbeat anime J-pop, female vocals"]
    assert message.replies[0].file is not None
    assert message.replies[0].file.filename == "music.mp3"
    # The source message is marked with the music emoji while the clip renders.
    assert message.added_reactions == ["🎵"]


async def test_music_disabled_still_strips_marker() -> None:
    """With no generator (music off) the block is still pulled and no file attaches."""
    message = FakeMessage()

    result = await _stream(
        streamer=_streamer(message=message),
        responses=_stream_events_from(events=_music_marker_events()),
    )

    assert "<generate-music>" not in result
    assert "anime" not in result
    assert message.replies[0].file is None


async def test_music_generation_failure_hints() -> None:
    """A failed render leaves a clean text reply with no file and a warning hint."""
    message = FakeMessage()
    generator = _FakeMusicGenerator(audio=None)

    result = await _stream(
        streamer=_streamer(message=message, music_generator=cast("MusicGenerator", generator)),
        responses=_stream_events_from(events=_music_marker_events()),
    )

    assert "anime" not in result
    assert message.replies[0].file is None
    assert message.added_reactions == ["🎵", "⚠️"]


# ---- inline video (<generate-video>) ----

_VIDEO_EMOJI = "<:video:1517560671913377842>"


class _FakeVideoGenerator:
    """Records generate calls and returns configurable MP4 bytes (or None) for streamer tests."""

    def __init__(self, video: bytes | None = b"\x00\x00\x00\x18ftypmp4") -> None:
        """Stores the MP4 bytes (None simulates a failed render) returned by generate."""
        self.video = video
        self.calls: list[str] = []
        self.reference_sources: list[list[tuple[bytes, str]] | None] = []

    async def generate(
        self, user_prompt: str, reference_image_sources: list[tuple[bytes, str]] | None = None
    ) -> bytes | None:
        """Records the description request (and any reference source images) and returns the clip."""
        self.calls.append(user_prompt)
        self.reference_sources.append(reference_image_sources)
        return self.video


def _video_marker_events() -> list[SimpleNamespace]:
    """A single-turn stream whose reply wraps a <generate-video> description."""
    return [
        _text_event(delta="幫你動起來 "),
        _text_event(delta="<generate-video>a wave crashing on rocks at sunset</generate-video>"),
        _completed_event(input_tokens=3, output_tokens=4),
    ]


async def test_video_marker_generates_and_attaches() -> None:
    """A <generate-video> block is pulled from the reply, generated, and the clip attached to the reply."""
    message = FakeMessage()
    generator = _FakeVideoGenerator()

    result = await _stream(
        streamer=_streamer(message=message, video_generator=cast("VideoGenerator", generator)),
        responses=_stream_events_from(events=_video_marker_events()),
    )

    # The block (tags AND description) never shows in chat.
    assert "<generate-video>" not in result
    assert "wave" not in result
    assert "幫你動起來" in result
    # The description is handed to the generator and the clip attached afterward.
    assert generator.calls == ["a wave crashing on rocks at sunset"]
    assert message.replies[0].file is not None
    assert message.replies[0].file.filename == "generated.mp4"
    # The source message is marked with the video emoji while the clip renders.
    assert message.added_reactions == [_VIDEO_EMOJI]
    # No input_builder wired -> no source images -> plain text-to-video (not a reference render).
    assert generator.reference_sources == [None]


async def test_video_marker_uses_uploaded_image_as_reference() -> None:
    """An uploaded image rides into the inline <generate-video> render as a subject reference."""
    message = FakeMessage()
    generator = _FakeVideoGenerator()

    async def _load(message: object, replied_to: object) -> list[LoadedMedia]:
        """Stands in for the input builder loading the message's uploaded image."""
        del message, replied_to
        return [LoadedMedia(data=b"uploaded-bytes", mime_type="image/png")]

    builder = SimpleNamespace(get_turn_image_sources=_load)

    await _stream(
        streamer=_streamer(
            message=message,
            video_generator=cast("VideoGenerator", generator),
            input_builder=cast("MessageInputBuilder", builder),
        ),
        responses=_stream_events_from(events=_video_marker_events()),
    )

    # The uploaded (bytes, mime) pair rides through to generate, so the inline <generate-video>
    # animates it and omni infers the task.
    assert generator.reference_sources == [
        [LoadedMedia(data=b"uploaded-bytes", mime_type="image/png")]
    ]
    assert generator.calls == ["a wave crashing on rocks at sunset"]


async def test_video_disabled_still_strips_marker() -> None:
    """With no generator (video off) the block is still pulled and no file attaches."""
    message = FakeMessage()

    result = await _stream(
        streamer=_streamer(message=message),
        responses=_stream_events_from(events=_video_marker_events()),
    )

    assert "<generate-video>" not in result
    assert "wave" not in result
    assert message.replies[0].file is None
    # The disabled path returns before the video emoji, so no spurious reaction is added.
    assert message.added_reactions == []


async def test_video_generation_failure_hints() -> None:
    """A failed render leaves a clean text reply with no file and a warning hint."""
    message = FakeMessage()
    generator = _FakeVideoGenerator(video=None)

    result = await _stream(
        streamer=_streamer(message=message, video_generator=cast("VideoGenerator", generator)),
        responses=_stream_events_from(events=_video_marker_events()),
    )

    assert "wave" not in result
    assert message.replies[0].file is None
    assert message.added_reactions == [_VIDEO_EMOJI, "⚠️"]


async def test_voice_music_video_image_attach_in_one_edit() -> None:
    """A reply with all four markers rides one edit carrying the WAV, music, video, and PNG."""
    message = FakeMessage()
    voice_generator = _FakeVoiceGenerator()
    music_generator = _FakeMusicGenerator()
    video_generator = _FakeVideoGenerator()
    image_generator = _FakeImageGenerator()

    result = await _stream(
        streamer=_streamer(
            message=message,
            voice_generator=cast("VoiceGenerator", voice_generator),
            music_generator=cast("MusicGenerator", music_generator),
            video_generator=cast("VideoGenerator", video_generator),
            image_generator=cast("ImageGenerator", image_generator),
        ),
        responses=_stream_events_from(
            events=[
                _text_event(delta="來囉 <generate-voice>聽好</generate-voice> "),
                _text_event(
                    delta="<generate-music>a calm lo-fi beat</generate-music><generate-video>a wave</generate-video><generate-image>a red balloon</generate-image>"
                ),
                _completed_event(input_tokens=3, output_tokens=4),
            ]
        ),
    )

    assert "聽好" in result
    assert "<generate-music>" not in result
    assert "lo-fi" not in result
    assert "<generate-video>" not in result
    assert "a wave" not in result
    assert "<generate-image>" not in result
    assert "a red balloon" not in result
    files = message.replies[0].files
    assert files is not None
    assert {item.filename for item in files} == {
        "reply.wav",
        "music.mp3",
        "generated.mp4",
        "generated.png",
    }


async def test_voice_oversized_clip_not_attached() -> None:
    """A clip past the guild's upload limit is dropped, leaving a text-only reply."""
    message = FakeMessage()
    message.guild = FakeGuild(filesize_limit=8)
    synthesizer = _FakeVoiceGenerator(audio=b"x" * 16)

    result = await _stream(
        streamer=_streamer(message=message, voice_generator=cast("VoiceGenerator", synthesizer)),
        responses=_stream_events_from(events=_voice_marker_events()),
    )

    _assert_no_voice_tags(result)
    assert message.replies[0].file is None
    # An oversized clip is dropped for a non-timeout reason, so it hints with the warning emoji.
    assert message.added_reactions == ["<:voice:1517558121092878376>", "⚠️"]


async def test_a_late_clip_on_a_finished_ask_answer_costs_the_clip_not_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalizing runs outside the answer's window, so its media's own bound gets to answer.

    Inside it, the window would cancel the clip and the ⏱️ edit with it and report the whole
    finished answer as a command that ran out of time.
    """
    message = FakeMessage()
    reply = FakeReply()
    streamer = ResponseStreamer(
        message=message,
        surface=_expiring_surface(message=as_message(fake=message), seconds_left=0.2),
        media_delivery=hosting_off_planner(),
        reply=as_message(fake=reply),
    )

    async def never_finishes() -> MediaItem:
        """Stands in for a voice clip that outlasts the window."""
        await asyncio.sleep(60)
        return MediaItem(source=b"RIFF", filename="reply.wav")

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        """Answers at once."""
        return _stream_events()

    monkeypatch.setattr(streamer, "_build_voice_candidate", never_finishes)

    answer = await asyncio.wait_for(
        stream_answer_with_retry(
            streamer=streamer, open_stream=open_stream, message_id=message.id
        ),
        timeout=5,
    )

    assert answer.startswith("hello from stream")
    assert reply.content is not None
    assert "⏱️" in reply.content
