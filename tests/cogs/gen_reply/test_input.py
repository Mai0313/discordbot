"""Tests for rendering Discord messages into model input."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from datetime import UTC, datetime

import pytest
from nextcord import Embed, Message
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.media import LoadedMedia, RenderedPart, RenderedAttachment
from discordbot.typings.models import ModelSettings, RuntimeModelCatalog
from discordbot.cogs.gen_reply.input import MessageInputBuilder
from discordbot.utils.llm_transcript import USAGE_FOOTER_RE, FORWARDED_MESSAGE_MARKER
from discordbot.cogs.gen_reply.toolkit import ReplyToolkit
from discordbot.typings.context_budgets import MAX_INLINE_ATTACHMENT_BYTES

from tests.helpers.casting import as_bot, as_message, step_dicts
from tests.helpers.gen_reply import FakeGeminiFiles
from tests.helpers.reply_fakes import (
    FakeAuthor,
    FakeClient,
    FakeChannel,
    FakeMessage,
    FakeSnapshot,
    FakeReference,
    FakeAttachment,
    _att,
    _cog,
    _png_bytes,
    _image_post,
    _fake_uploader,
    _media_builder,
    _context_builder,
)
from tests.helpers.logfire_capture import capture_logs

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _pin_gemini_slow_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the answer tier to Gemini so Files API selection tests stay independent of catalog choices."""
    monkeypatch.setattr(
        RuntimeModelCatalog,
        "slow_model",
        property(lambda _self: ModelSettings(name="gemini-test", effort="high")),
    )


async def _attachment_parts(builder: MessageInputBuilder, message: object) -> list[RenderedPart]:
    """Renders a message's attachments from its own gated sources, as the answer render does."""
    discord_message = as_message(fake=message)
    sources = builder._supported_sources(
        sources=builder.collect_attachment_sources(message=discord_message),
        message_id=discord_message.id,
    )
    return await builder.get_attachment_parts(message=discord_message, sources=sources)


def test_collect_sources_skips_bot_own_voice_clip() -> None:
    """The bot's own generated voice clip is dropped from history input; others survive."""
    builder = _media_builder()  # bot user id 999

    bot_msg = FakeMessage(author=FakeAuthor(user_id=999))
    bot_msg.attachments = [
        FakeAttachment(filename="reply.wav", content_type="audio/wav", attachment_id=1),
        FakeAttachment(filename="note.txt", content_type="text/plain", attachment_id=2),
    ]
    # The bot's voice clip is skipped; a normal attachment on its message is kept.
    assert [
        s.cache_key for s in builder.collect_attachment_sources(message=as_message(fake=bot_msg))
    ] == [2]

    # The same filename on a human's message is NOT skipped (only the bot's own clip is).
    user_msg = FakeMessage(author=FakeAuthor(user_id=1))
    user_msg.attachments = [
        FakeAttachment(filename="reply.wav", content_type="audio/wav", attachment_id=3)
    ]
    assert [
        s.cache_key for s in builder.collect_attachment_sources(message=as_message(fake=user_msg))
    ] == [3]


def test_collect_sources_keeps_bot_own_music_clip() -> None:
    """The bot's own generated music clip is deliberately retained (unlike the voice clip).

    The `<generate-music>` description is stripped from the visible reply, so the clip is the only trace
    of the song the bot made; keeping it lets a later turn reference it. Only the spoken `reply.wav`
    (whose text is already in the transcript) is skipped.
    """
    builder = _media_builder()  # bot user id 999

    bot_msg = FakeMessage(author=FakeAuthor(user_id=999))
    bot_msg.attachments = [
        FakeAttachment(filename="music.mp3", content_type="audio/mpeg", attachment_id=1),
        FakeAttachment(filename="reply.wav", content_type="audio/wav", attachment_id=2),
    ]
    # The music clip is kept (cache_key 1); only the voice clip (cache_key 2) is skipped.
    assert [
        s.cache_key for s in builder.collect_attachment_sources(message=as_message(fake=bot_msg))
    ] == [1]


def test_collect_sources_includes_forwarded_snapshot_media() -> None:
    """A forwarded message's attachments (in message.snapshots) are collected, not dropped."""
    builder = _media_builder()

    msg = FakeMessage(author=FakeAuthor(user_id=1))
    # The forwarder also dragged along their own attachment; both it and the forwarded one count.
    msg.attachments = [
        FakeAttachment(filename="own.txt", content_type="text/plain", attachment_id=1)
    ]
    msg.snapshots = [
        FakeSnapshot(
            content="forwarded",
            attachments=[
                FakeAttachment(filename="pic.png", content_type="image/png", attachment_id=2)
            ],
        )
    ]
    assert [
        s.cache_key for s in builder.collect_attachment_sources(message=as_message(fake=msg))
    ] == [1, 2]


async def test_cleaned_content_includes_forwarded_snapshot_text() -> None:
    """Forwarded snapshot text is folded in and tagged so a forward is never blank."""
    builder = _media_builder()

    # Text-only forward: the snapshot content surfaces under the tag.
    forward_only = FakeMessage(author=FakeAuthor(user_id=1))
    forward_only.snapshots = [FakeSnapshot(content="hello from elsewhere")]
    rendered = await builder.get_cleaned_content(message=as_message(fake=forward_only))
    assert FORWARDED_MESSAGE_MARKER in rendered
    assert "hello from elsewhere" in rendered

    # The forwarder's own comment is kept alongside the forwarded body (append, not replace).
    with_comment = FakeMessage(content="look at this", author=FakeAuthor(user_id=1))
    with_comment.snapshots = [FakeSnapshot(content="original text")]
    rendered = await builder.get_cleaned_content(message=as_message(fake=with_comment))
    assert "look at this" in rendered
    assert "original text" in rendered

    # A media-only forward still emits the bare tag (its attachment rides separately).
    media_only = FakeMessage(author=FakeAuthor(user_id=1))
    media_only.snapshots = [
        FakeSnapshot(
            attachments=[
                FakeAttachment(filename="pic.png", content_type="image/png", attachment_id=2)
            ]
        )
    ]
    assert (
        await builder.get_cleaned_content(message=as_message(fake=media_only))
        == FORWARDED_MESSAGE_MARKER
    )

    # Forwarding the bot's own reply (snapshot has no author) still strips the usage footer.
    forwarded_bot_reply = FakeMessage(author=FakeAuthor(user_id=1))
    forwarded_bot_reply.snapshots = [
        FakeSnapshot(content="real answer\n\n-# model · ⬆ 1 ⬇ 2 · $0.0")
    ]
    rendered = await builder.get_cleaned_content(message=as_message(fake=forwarded_bot_reply))
    assert "real answer" in rendered
    assert "⬆" not in rendered

    footer_only_forward = FakeMessage(author=FakeAuthor(user_id=1))
    footer_only_forward.snapshots = [
        FakeSnapshot(
            content="\n\n-# model · ⬆ 1 ⬇ 2 · $0.0",
            embeds=[Embed(url="https://youtu.be/jNQXAC9IVRw")],
        )
    ]
    assert (
        await builder.get_cleaned_content(message=as_message(fake=footer_only_forward))
        == FORWARDED_MESSAGE_MARKER
    )

    # A captioned forward renders the caption only; an embed-only URL is not shown (nor scanned).
    captioned = FakeMessage(author=FakeAuthor(user_id=1))
    captioned.snapshots = [
        FakeSnapshot(content="funny", embeds=[Embed(url="https://youtu.be/jNQXAC9IVRw")])
    ]
    rendered = await builder.get_cleaned_content(message=as_message(fake=captioned))
    assert "funny" in rendered
    assert "youtu.be" not in rendered


def test_forwarded_request_text_is_untagged() -> None:
    """The media-prompt helper returns raw forwarded text without the `[forwarded message]` tag."""
    builder = _media_builder()

    forward = FakeMessage(author=FakeAuthor(user_id=1))
    forward.snapshots = [FakeSnapshot(content="draw a cat")]
    assert builder.forwarded_request_text(message=as_message(fake=forward)) == "draw a cat"

    # A normal message (no snapshots) yields no forwarded request text.
    assert builder.forwarded_request_text(message=as_message(fake=FakeMessage(content="hi"))) == ""


def test_extract_embed_text_includes_embed_url() -> None:
    """A link card's own url is rendered, so the answer model sees the link, not just a title."""
    builder = _media_builder()
    url = "https://youtu.be/jNQXAC9IVRw"
    assert url in builder.extract_embed_text(embeds=[Embed(url=url)])


def test_required_modality_gate_keeps_code_and_text() -> None:
    """The MIME gate drops unknown binaries but keeps source-code / structured-text types."""
    modality = MessageInputBuilder.required_modality
    # Known binary application types are dropped before any upload.
    assert modality(content_type="application/octet-stream") == "unknown"
    assert modality(content_type="application/x-tar") == "unknown"
    # Office / OpenDocument binaries the Gemini backend rejects are dropped, not uploaded.
    assert modality(content_type="application/msword") == "unknown"
    assert modality(content_type="application/vnd.ms-excel") == "unknown"
    assert (
        modality(
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        == "unknown"
    )
    assert modality(content_type="application/vnd.oasis.opendocument.text") == "unknown"
    # Source-code / script application types still proxy through (.rb -> application/x-ruby).
    assert modality(content_type="application/x-ruby") == "image"
    assert modality(content_type="application/x-perl") == "image"
    # Structured-text suffixes and text/* pass too.
    assert modality(content_type="application/geo+json") == "image"
    assert modality(content_type="application/atom+xml") == "image"
    assert modality(content_type="text/x-go") == "image"
    assert modality(content_type="video/mp4") == "video"
    assert modality(content_type="audio/mpeg") == "audio"
    assert modality(content_type="application/pdf") == "image"


async def test_gen_reply_message_content_and_attachment_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies prompt cleanup, embed extraction, and attachment conversion."""
    _pin_gemini_slow_model(monkeypatch=monkeypatch)
    cog = _cog()
    embed = Embed(title="Title", description="Body")
    embed.set_author(name="Author")
    embed.add_field(name="Field", value="Value")
    embed.set_footer(text="Footer")

    assert await cog.toolkit.input_builder.get_user_prompt(content="hi <@999>") == "hi"
    assert await cog.toolkit.input_builder.get_user_prompt(content="hi <@!999>") == "hi"
    assert "Author" in cog.toolkit.input_builder.extract_embed_text(embeds=[embed])

    bot_message = FakeMessage(
        content="answer\n\n-# model · ⬆ 1 ⬇ 2 · $0.0", author=FakeAuthor(bot=True, user_id=999)
    )
    assert (
        await cog.toolkit.input_builder.get_cleaned_content(message=as_message(fake=bot_message))
        == "answer"
    )
    assert USAGE_FOOTER_RE.search(string=bot_message.content)
    bot_message.content = "\n\n-# model · ⬆ 1 ⬇ 2 · $0.0"
    bot_message.embeds = [Embed(url="https://youtu.be/jNQXAC9IVRw")]
    assert (
        await cog.toolkit.input_builder.get_cleaned_content(message=as_message(fake=bot_message))
        == ""
    )

    embed_message = FakeMessage()
    embed_message.embeds = [embed]
    assert "Title" in await cog.toolkit.input_builder.get_cleaned_content(
        message=as_message(fake=embed_message)
    )

    system_message = FakeMessage()
    system_message.system_content = "joined"
    assert (
        await cog.toolkit.input_builder.get_cleaned_content(
            message=as_message(fake=system_message)
        )
        == "joined"
    )

    file_rendered = await cog.toolkit.input_builder.attachment_handler.render_file(
        attachment=_att(filename="note.txt", content_type="text/plain", payload=b"abc"),
        cache_key="note.txt",
    )
    assert file_rendered is not None
    file_part = file_rendered.part
    file_expiry = file_rendered.expires_at
    assert file_part["type"] == "input_file"
    assert file_part["file_id"] == "https://files.test/note.txt"
    assert file_expiry == datetime(2099, 1, 1, tzinfo=UTC)

    image_rendered = await cog.toolkit.input_builder.attachment_handler.render_image(
        source=_att(filename="pixel.png", content_type="image/png", payload=_png_bytes()),
        cache_key="pixel.png",
    )
    assert image_rendered is not None
    image_part = image_rendered.part
    assert image_part["type"] == "input_file"
    assert image_part["file_id"] == "https://files.test/pixel.png"

    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.input.get_supported_modalities", lambda model_name: {"image"}
    )
    message = FakeMessage()
    message.attachments = [
        FakeAttachment(filename="pixel.png", content_type="image/png", payload=_png_bytes()),
        FakeAttachment(filename="clip.mp4", content_type="video/mp4", payload=b"video"),
    ]
    message.stickers = [
        FakeAttachment(filename="sticker.png", content_type="image/png", payload=_png_bytes())
    ]
    img_embed = Embed()
    img_embed.set_image(url="https://example.test/image.png")
    message.embeds = [img_embed]
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.attachment.loaders.get_image_data",
        lambda image_file: LoadedMedia(data=_png_bytes(), mime_type="image/png"),
    )
    parts = await _attachment_parts(builder=cog.toolkit.input_builder, message=message)
    assert [part["type"] for part in parts] == ["input_file", "input_file", "input_file"]


async def test_a_keyless_deployment_inlines_attachments_instead_of_dropping_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a Gemini key, a Gemini answer model still reads attachments, each fetched once.

    Nothing can upload without a key, so handing a Gemini answer model the uploader downloaded
    every attachment only to drop it, and left the message uncached to do it all again on the
    next reply. A clip the inline renderer cannot carry is neither marked nor downloaded.
    """
    _pin_gemini_slow_model(monkeypatch=monkeypatch)
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.input.get_supported_modalities",
        lambda model_name: {"text", "image", "audio", "video"},
    )
    logged = capture_logs(monkeypatch=monkeypatch, level="error")
    bot = as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=999, name="bot")))
    toolkit = ReplyToolkit(bot=bot, openai_client=FakeClient(), gemini_api_key="")
    builder = toolkit.input_builder
    pic = FakeAttachment(
        filename="pic.png", content_type="image/png", payload=_png_bytes(), attachment_id=1
    )
    notes = FakeAttachment(filename="notes.txt", content_type="text/plain", attachment_id=2)
    clip = FakeAttachment(filename="clip.mp4", content_type="video/mp4", attachment_id=3)
    message = FakeMessage(content="<@999> look", author=FakeAuthor(user_id=1))
    message.attachments = [pic, notes, clip]

    text_only = await builder.process_single_message(
        message=as_message(fake=message), text_only=True
    )
    await builder.process_single_message(message=as_message(fake=message))
    full = await builder.process_single_message(message=as_message(fake=message))

    assert [part["text"] for part in step_dicts(steps=text_only["content"])[1:]] == [
        "[attachment: image]",
        "[attachment: file]",
    ]
    assert [part["type"] for part in step_dicts(steps=full["content"])[1:]] == [
        "input_image",
        "input_text",
    ]
    assert (pic.read_count, notes.read_count, clip.read_count) == (1, 1, 0)
    assert logged == []


async def test_gen_reply_processes_history_reference_and_current_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies message processing for history, references, and current prompts."""
    cog = _cog()
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.input.get_supported_modalities",
        lambda model_name: {"text", "image"},
    )
    bot_msg = FakeMessage(content="bot answer", author=FakeAuthor(bot=True, user_id=999))
    user_msg = FakeMessage(content="hello", author=FakeAuthor(user_id=1))
    with_attachment = FakeMessage(content="see file", author=FakeAuthor(user_id=2))
    with_attachment.attachments = [FakeAttachment(filename="note.txt", content_type="text/plain")]

    bot_processed = await cog.toolkit.input_builder.process_single_message(
        message=as_message(fake=bot_msg)
    )
    user_processed = await cog.toolkit.input_builder.process_single_message(
        message=as_message(fake=user_msg)
    )
    attachment_processed = await cog.toolkit.input_builder.process_single_message(
        message=as_message(fake=with_attachment)
    )
    assert bot_processed["role"] == "assistant"
    assert user_processed["role"] == "user"
    assert attachment_processed["role"] == "user"
    assert isinstance(attachment_processed["content"], list)

    async def fake_history(limit: int, before: FakeMessage) -> AsyncIterator[FakeMessage]:
        """Yields two messages newest first, as Discord does."""
        yield bot_msg
        yield user_msg

    current = FakeMessage(content="current", author=FakeAuthor(user_id=3))
    current.channel = FakeChannel(history=fake_history)
    raw_history = await _context_builder(cog=cog, message=as_message(fake=current)).fetch_history(
        limit=30
    )
    rendered = await _context_builder(cog=cog, message=as_message(fake=current)).render_history(
        hist_messages=raw_history
    )
    assert len(rendered) == 3
    assert rendered[0]["role"] == "system"
    assert [m.content for m in raw_history] == ["hello", "bot answer"]

    parent = FakeMessage(content="parent", author=FakeAuthor(user_id=4))
    grandparent = FakeMessage(content="grandparent", author=FakeAuthor(user_id=5))
    parent.reference = FakeReference(resolved=grandparent)
    current.reference = FakeReference(resolved=parent)
    reference = await _context_builder(
        cog=cog, message=as_message(fake=current)
    ).render_reference_message()
    # One header plus one message, and the grandparent these fakes nest is not among them.
    # Discord never nests `referenced_message`, so a real parent's own `.reference.resolved` is
    # always None and only a fake can offer a second link at all. This assertion is what makes a
    # walk that follows one fail here.
    assert len(reference) == 2
    assert reference[0]["role"] == "system"
    assert "grandparent" not in str(reference[1]["content"])
    assert (
        len(
            await _context_builder(
                cog=cog, message=as_message(fake=current)
            ).render_current_message()
        )
        == 2
    )


async def test_gen_reply_preserves_bot_mention_in_text_context() -> None:
    """Regression: self-mentions can be the subject of a normal QA message."""
    cog = _cog()
    message = FakeMessage(
        content="你的審美跟 <@999> 一樣 這樣算誇獎嗎", author=FakeAuthor(user_id=1)
    )

    processed = await cog.toolkit.input_builder.process_single_message(
        message=as_message(fake=message)
    )
    rendered = processed["content"]

    assert isinstance(rendered, str)
    assert "你的審美跟 <@999> 一樣 這樣算誇獎嗎" in rendered


async def test_render_history_survives_a_message_the_collector_chokes_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unexpected message shape costs that message its files, never the whole reply.

    The budget walk runs inside `ReplyContextBuilder.build`'s gather, which has no except of its
    own, so anything raised here would reach `on_message`'s generic error path and lose the
    answer. Both renders already swallow this same collect step for exactly that reason.
    """
    cog = _cog()
    posts = [_image_post(index=i, count=2) for i in range(2)]
    # Patched on the class: `MessageInputBuilder` is a pydantic model, so an instance rejects a
    # setattr of anything that is not one of its fields.
    collect = MessageInputBuilder.collect_attachment_sources

    def explode(self: MessageInputBuilder, message: Message) -> object:
        if message.id == posts[0].id:
            raise RuntimeError("unexpected nextcord shape")
        return collect(self, message=message)

    monkeypatch.setattr(MessageInputBuilder, "collect_attachment_sources", explode)

    rendered = await _context_builder(
        cog=cog, message=as_message(fake=FakeMessage())
    ).render_history(hist_messages=[as_message(fake=m) for m in posts])

    # Header plus both messages: the broken one degrades to empty text, the other is untouched.
    assert len(rendered) == 3


async def test_render_history_degrades_over_budget_attachments_to_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the cap a history post renders as the route's marker, not as uploaded files."""
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.input.get_supported_modalities",
        lambda model_name: {"text", "image"},
    )
    cog = _cog()
    posts = [_image_post(index=i, count=6) for i in range(3)]

    rendered = await _context_builder(
        cog=cog, message=as_message(fake=FakeMessage())
    ).render_history(hist_messages=[as_message(fake=m) for m in posts])

    # rendered[0] is the history header; the rest follow the posts in order.
    oldest, newest = rendered[1]["content"], rendered[3]["content"]
    assert isinstance(oldest, list)
    assert isinstance(newest, list)
    assert [part["type"] for part in oldest[1:]] == ["input_text"] * 6
    assert {part.get("text") for part in oldest[1:]} == {"[attachment: image]"}
    assert all(part["type"] != "input_text" for part in newest[1:])


async def test_uploaded_image_without_extension_marks_as_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An image attachment whose filename lacks an extension still marks as an image."""
    cog = _cog()
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.input.get_supported_modalities", lambda model_name: {"image"}
    )
    message = FakeMessage(content="<@999> see", author=FakeAuthor(user_id=1))
    message.attachments = [
        FakeAttachment(
            filename="screenshot",
            content_type="image/png",
            payload=_png_bytes(),
            url="https://example.test/screenshot",
        )
    ]

    # Classification is by content_type, not filename, so the marker render needs no upload.
    rendered = await cog.toolkit.input_builder.process_single_message(
        message=as_message(fake=message), text_only=True
    )
    parts = rendered["content"]
    assert isinstance(parts, list)
    assert step_dicts(steps=parts)[-1]["text"] == "[attachment: image]"


async def test_text_only_render_names_a_sticker_instead_of_calling_it_an_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sticker marks as a sticker, so a text-only reader can tell a reaction from a screenshot."""
    cog = _cog()
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.input.get_supported_modalities", lambda model_name: {"image"}
    )
    message = FakeMessage(content="", author=FakeAuthor(user_id=1))
    message.stickers = [
        FakeAttachment(
            filename="sticker.png",
            content_type="image/png",
            payload=_png_bytes(),
            url="https://example.test/sticker.png",
        )
    ]

    rendered = await cog.toolkit.input_builder.process_single_message(
        message=as_message(fake=message), text_only=True
    )
    parts = rendered["content"]
    assert isinstance(parts, list)
    assert step_dicts(steps=parts)[-1]["text"] == "[attachment: sticker]"


def _attachment_slots(
    text_only: EasyInputMessageParam, full: EasyInputMessageParam
) -> tuple[int, int]:
    """Counts the attachment markers in a text-only render and the files in the full render."""
    text_markers = [
        part
        for part in text_only["content"]
        if isinstance(part, dict) and str(part.get("text", "")).startswith("[attachment:")
    ]
    full_files = [
        part
        for part in full["content"]
        if isinstance(part, dict) and part.get("type") == "input_file"
    ]
    return len(text_markers), len(full_files)


async def test_text_only_and_full_render_agree_on_attachment_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The marker render and the upload render keep the same supported-attachment slots."""
    _pin_gemini_slow_model(monkeypatch=monkeypatch)
    cog = _cog()
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.input.get_supported_modalities", lambda model_name: {"image"}
    )
    message = FakeMessage(content="<@999> mix", author=FakeAuthor(user_id=1))
    message.attachments = [
        FakeAttachment(filename="pic.png", content_type="image/png", payload=_png_bytes()),
        FakeAttachment(filename="clip.mp4", content_type="video/mp4", payload=b"v"),
    ]

    text_only = await cog.toolkit.input_builder.process_single_message(
        message=as_message(fake=message), text_only=True
    )
    full = await cog.toolkit.input_builder.process_single_message(message=as_message(fake=message))

    text_markers, full_files = _attachment_slots(text_only=text_only, full=full)
    assert text_markers == full_files == 1


class _RefusingFiles(FakeGeminiFiles):
    """Fake Files resource that refuses one file's upload, as a dead API key refuses every one."""

    def __init__(self, refused: str) -> None:
        """Initializes the fake with the display name whose upload raises."""
        super().__init__()
        self.refused = refused

    async def upload(self, file: object, config: dict[str, str]) -> SimpleNamespace:
        """Raises for the refused display name and uploads anything else."""
        if config["display_name"] == self.refused:
            raise RuntimeError("401 UNAUTHENTICATED")
        return await super().upload(file=file, config=config)


class _UnreadableAttachment(FakeAttachment):
    """An attachment whose download fails, as an unreachable CDN URL does."""

    async def read(self) -> bytes:
        """Raises instead of returning the attachment's bytes."""
        raise OSError("connection reset")


@pytest.mark.parametrize("failure", ["download", "upload"])
async def test_a_failed_attachment_reaches_the_model_as_a_notice(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """An attachment that fails to download or upload is named in its place, not dropped.

    Dropped outright, an attachment-only message reaches the answer as an author header and
    nothing else, and the model tells the sender their message was blank.
    """
    _pin_gemini_slow_model(monkeypatch=monkeypatch)
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.input.get_supported_modalities", lambda model_name: {"image"}
    )
    builder = _media_builder()
    # Each case breaks one step only, so neither passes on the other's failure.
    refused = "photo.png" if failure == "upload" else ""
    builder.attachment_handler = _fake_uploader(files=_RefusingFiles(refused=refused))
    failing = _UnreadableAttachment if failure == "download" else FakeAttachment
    message = FakeMessage(content="", author=FakeAuthor(user_id=1))
    message.attachments = [
        failing(
            filename="photo.png", content_type="image/png", payload=_png_bytes(), attachment_id=1
        ),
        FakeAttachment(
            filename="pic.png", content_type="image/png", payload=_png_bytes(), attachment_id=2
        ),
    ]

    rendered = await builder.process_single_message(message=as_message(fake=message))

    parts = step_dicts(steps=rendered["content"])[1:]
    assert [part["type"] for part in parts] == ["input_text", "input_file"]
    assert parts[0]["text"].startswith("[attachment: image")
    assert "fail" in parts[0]["text"]


@pytest.mark.parametrize(
    ("file_api_enabled", "markers"),
    [("true", ["[attachment: image]", "[attachment: file]"]), ("false", ["[attachment: image]"])],
    ids=["files-api", "inline"],
)
async def test_the_gate_passes_only_what_the_selected_renderer_carries(
    monkeypatch: pytest.MonkeyPatch, file_api_enabled: str, markers: list[str]
) -> None:
    """A clip the inline renderer drops is not marked or budgeted, nor keeps its message uncached.

    The model accepts video either way; only the renderer the switch selects decides whether the
    clip reaches the answer, so the route marker, the history budget and the render cache must
    follow the renderer. A dropped part keeps its whole message out of the cache, which would
    download the image beside it again on every reply.
    """
    _pin_gemini_slow_model(monkeypatch=monkeypatch)
    monkeypatch.setenv(name="FILE_API_ENABLED", value=file_api_enabled)
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.input.get_supported_modalities",
        lambda model_name: {"text", "image", "audio", "video"},
    )
    builder = _cog().toolkit.input_builder
    pic = FakeAttachment(
        filename="pic.png", content_type="image/png", payload=_png_bytes(), attachment_id=1
    )
    clip = FakeAttachment(filename="clip.mp4", content_type="video/mp4", attachment_id=2)
    message = FakeMessage(content="<@999> look", author=FakeAuthor(user_id=1))
    message.attachments = [pic, clip]

    text_only = await builder.process_single_message(
        message=as_message(fake=message), text_only=True
    )
    for _ in range(2):
        await builder.process_single_message(message=as_message(fake=message))

    assert [part["text"] for part in step_dicts(steps=text_only["content"])[1:]] == markers
    assert builder.count_supported_sources(message=as_message(fake=message)) == len(markers)
    assert pic.read_count == 1


@pytest.mark.parametrize(
    ("filename", "content_type", "payload", "reads"),
    [
        ("app.apk", "application/vnd.android.package-archive", b"PK\x03\x04\xff", (1, 0)),
        ("font.ttf", "font/ttf", b"\x00\x01\x00\x00\xff", (1, 0)),
        ("notes.txt", "text/plain", "繁體中文".encode("big5"), (2, 1)),
    ],
    ids=["apk", "font", "big5-text"],
)
async def test_the_inline_gate_stops_passing_a_file_the_renderer_cannot_read(
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    content_type: str,
    payload: bytes,
    reads: tuple[int, int],
) -> None:
    """A file the inline renderer drops is not marked or budgeted, nor keeps its message uncached.

    A font or an Android package is refused on its type, before any download. A `text/plain` file
    may hold UTF-8 or not, so it is read once; once its bytes rule it out it is refused like the
    others, and the message caches on the next render rather than downloading the image beside it
    on every reply.
    """
    monkeypatch.setenv(name="FILE_API_ENABLED", value="false")
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.input.get_supported_modalities",
        lambda model_name: {"text", "image", "audio", "video"},
    )
    builder = _cog().toolkit.input_builder
    pic = FakeAttachment(
        filename="pic.png", content_type="image/png", payload=_png_bytes(), attachment_id=1
    )
    other = FakeAttachment(
        filename=filename, content_type=content_type, payload=payload, attachment_id=2
    )
    message = FakeMessage(content="<@999> look", author=FakeAuthor(user_id=1))
    message.attachments = [pic, other]

    for _ in range(3):
        await builder.process_single_message(message=as_message(fake=message))
    logged = capture_logs(monkeypatch=monkeypatch, level="info")
    text_only = await builder.process_single_message(
        message=as_message(fake=message), text_only=True
    )

    assert [part["text"] for part in step_dicts(steps=text_only["content"])[1:]] == [
        "[attachment: image]"
    ]
    assert builder.count_supported_sources(message=as_message(fake=message)) == 1
    assert (pic.read_count, other.read_count) == reads
    # The record has to say the renderer refused it: the model takes the `image` it proxies as.
    assert [
        (fields["modality"], fields["carried"])
        for text, fields in logged
        if text == "gen_reply skipping unsupported attachment"
    ] == [("image", False)]


@pytest.mark.parametrize(
    ("file_api_enabled", "markers", "big_reads", "skipped"),
    [
        ("true", ["[attachment: image]", "[attachment: file]", "[attachment: image]"], 1, []),
        (
            "false",
            ["[attachment: image]"],
            0,
            [
                ("application/pdf", MAX_INLINE_ATTACHMENT_BYTES + 1),
                ("image/gif", MAX_INLINE_ATTACHMENT_BYTES + 1),
            ],
        ),
    ],
    ids=["files-api", "inline"],
)
async def test_the_inline_gate_refuses_an_attachment_too_large_to_inline(
    monkeypatch: pytest.MonkeyPatch,
    file_api_enabled: str,
    markers: list[str],
    big_reads: int,
    skipped: list[tuple[str, int]],
) -> None:
    """An attachment Discord reports past the inline cap is refused before it is downloaded.

    So it is not marked to the route, not budgeted, and its message caches on the first render
    rather than downloading the image beside it on every reply. An image is held to its posted
    size too. An upload sends a handle rather than the bytes, so the Files API path takes all three.
    """
    _pin_gemini_slow_model(monkeypatch=monkeypatch)
    monkeypatch.setenv(name="FILE_API_ENABLED", value=file_api_enabled)
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.input.get_supported_modalities",
        lambda model_name: {"text", "image"},
    )
    builder = _cog().toolkit.input_builder
    pic = FakeAttachment(
        filename="pic.png", content_type="image/png", payload=_png_bytes(), attachment_id=1
    )
    pdf = FakeAttachment(
        filename="big.pdf", content_type="application/pdf", payload=b"%PDF-1.7", attachment_id=2
    )
    gif = FakeAttachment(
        filename="big.gif", content_type="image/gif", payload=b"GIF89a", attachment_id=3
    )
    # What Discord reports, without allocating it.
    pdf.size = gif.size = MAX_INLINE_ATTACHMENT_BYTES + 1
    message = FakeMessage(content="<@999> look", author=FakeAuthor(user_id=1))
    message.attachments = [pic, pdf, gif]

    logged = capture_logs(monkeypatch=monkeypatch, level="info")
    text_only = await builder.process_single_message(
        message=as_message(fake=message), text_only=True
    )
    for _ in range(2):
        await builder.process_single_message(message=as_message(fake=message))

    assert [part["text"] for part in step_dicts(steps=text_only["content"])[1:]] == markers
    assert builder.count_supported_sources(message=as_message(fake=message)) == len(markers)
    assert (pic.read_count, pdf.read_count, gif.read_count) == (1, big_reads, big_reads)
    # Logged once per render, so the text-only one and the two full ones.
    assert [
        (fields["content_type"], fields["size_bytes"])
        for text, fields in logged
        if text == "gen_reply skipping unsupported attachment"
    ] == skipped * 3


@pytest.mark.parametrize(
    "content_type",
    [None, " ", "; charset=utf-8", "application/zip; x", "Application/Zip"],
    ids=["none", "blank", "parameter-only", "denylisted-with-parameter", "denylisted-mixed-case"],
)
async def test_an_attachment_with_no_resolvable_mime_is_neither_marked_nor_counted(
    monkeypatch: pytest.MonkeyPatch, content_type: str | None
) -> None:
    """A file no renderer can type, or a denylisted one, is left out of the marker and budget.

    The gate and the renderers read the same normalised MIME, so a parameter or a case change
    can neither pass a blank type the renderer then drops nor carry a denylisted one through.
    """
    _pin_gemini_slow_model(monkeypatch=monkeypatch)
    cog = _cog()
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.input.get_supported_modalities", lambda model_name: {"image"}
    )
    message = FakeMessage(content="<@999> build this", author=FakeAuthor(user_id=1))
    message.attachments = [
        FakeAttachment(
            filename="pic.png", content_type="image/png", payload=_png_bytes(), attachment_id=1
        ),
        # A name `mimetypes` cannot guess, so a missing type resolves to "".
        FakeAttachment(
            filename="Makefile", content_type=content_type, payload=b"all:\n", attachment_id=2
        ),
    ]

    text_only = await cog.toolkit.input_builder.process_single_message(
        message=as_message(fake=message), text_only=True
    )
    full = await cog.toolkit.input_builder.process_single_message(message=as_message(fake=message))

    text_markers, full_files = _attachment_slots(text_only=text_only, full=full)
    budgeted = cog.toolkit.input_builder.count_supported_sources(message=as_message(fake=message))
    assert text_markers == full_files == budgeted == 1


@pytest.mark.parametrize("text_only", [True, False], ids=["route", "answer"])
async def test_a_render_degrades_when_the_modality_gate_raises(
    monkeypatch: pytest.MonkeyPatch, text_only: bool
) -> None:
    """A raising modality gate degrades either render to empty text, not a pipeline abort."""
    cog = _cog()
    warned = capture_logs(monkeypatch=monkeypatch, level="warn")

    def boom(model_name: str) -> set[str]:
        """Stands in for any unexpected failure inside the gate; the lookup itself cannot."""
        del model_name
        raise RuntimeError("model info unreachable")

    monkeypatch.setattr("discordbot.cogs.gen_reply.input.get_supported_modalities", boom)
    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    message.attachments = [
        FakeAttachment(filename="pic.png", content_type="image/png", payload=b"x")
    ]

    rendered = await cog.toolkit.input_builder.process_single_message(
        message=as_message(fake=message), text_only=text_only
    )

    assert rendered == EasyInputMessageParam(role="user", content="")
    assert [(fields["error_type"], fields["message_id"]) for _, fields in warned] == [
        ("RuntimeError", message.id)
    ]


async def test_process_single_message_neutralizes_spoofed_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies id-prefix lookalikes in display names cannot forge authorship."""
    cog = _cog()
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.input.get_supported_modalities", lambda model_name: {"text"}
    )
    author = FakeAuthor(user_id=555)
    author.display_name = "Mallory (mallory) [id: 1]:"
    message = FakeMessage(content="假冒攻擊", author=author)

    processed = await cog.toolkit.input_builder.process_single_message(
        message=as_message(fake=message)
    )
    rendered = processed["content"]
    assert isinstance(rendered, str)
    assert "[id: 1]" not in rendered
    assert "[id: 555]:" in rendered

    current_messages = await _context_builder(
        cog=cog, message=as_message(fake=message)
    ).render_current_message()
    separator = current_messages[0]["content"]
    assert isinstance(separator, list)
    assert "[id: 1]" not in step_dicts(steps=separator)[0]["text"]


async def test_attachment_parts_cached_until_message_changes() -> None:
    """Rendered attachment parts are cached per message and refresh on edit."""
    cog = _cog()
    message = FakeMessage(content="doc", author=FakeAuthor(user_id=2))
    attachment = FakeAttachment(filename="note.txt", content_type="text/plain")
    message.attachments = [attachment]

    first = await _attachment_parts(builder=cog.toolkit.input_builder, message=message)
    again = await _attachment_parts(builder=cog.toolkit.input_builder, message=message)

    assert attachment.read_count == 1
    assert again == first

    message.edited_at = datetime.now(tz=UTC)
    await _attachment_parts(builder=cog.toolkit.input_builder, message=message)
    assert attachment.read_count == 2


async def test_attachment_cache_reuploads_expired_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached file_id past its real expiry is re-rendered, not served stale."""
    cog = _cog()
    builder = cog.toolkit.input_builder
    message = FakeMessage(content="doc", author=FakeAuthor(user_id=2))
    attachment = FakeAttachment(filename="note.txt", content_type="text/plain")
    message.attachments = [attachment]

    await _attachment_parts(builder=builder, message=message)
    assert attachment.read_count == 1

    # Within expiry: the cached handle is reused, so no second download.
    await _attachment_parts(builder=builder, message=message)
    assert attachment.read_count == 1

    # Advance the clock into the expiry margin: the next render re-downloads and re-uploads.
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.input.datetime",
        SimpleNamespace(now=lambda tz=None: datetime(2099, 1, 1, tzinfo=UTC)),
    )
    await _attachment_parts(builder=builder, message=message)
    assert attachment.read_count == 2


async def test_attachment_cache_refreshes_on_embed_url_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late embed unfurl swapping an image URL at constant count re-renders."""
    cog = _cog()
    message = FakeMessage(content="link", author=FakeAuthor(user_id=2))
    rendered_urls: list[str] = []

    async def fake_render_image(
        self: object, source: object, cache_key: object, allow_dead_cache: bool = False
    ) -> RenderedAttachment:
        """Records each rendered source instead of hitting the network."""
        del self, cache_key, allow_dead_cache
        rendered_urls.append(str(source))
        return RenderedAttachment(
            part={"type": "input_image", "image_url": str(source), "detail": "auto"},
            expires_at=datetime(2099, 1, 1, tzinfo=UTC),
        )

    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.attachment.gemini_file_api.GeminiFileUploader.render_image",
        fake_render_image,
    )

    def _embed(url: str) -> SimpleNamespace:
        """Builds a fake embed whose image carries a swappable proxy URL."""
        return SimpleNamespace(image=SimpleNamespace(proxy_url=url, url=url), thumbnail=None)

    message.embeds = [cast("Embed", _embed("https://media.test/a.png"))]
    await _attachment_parts(builder=cog.toolkit.input_builder, message=message)
    await _attachment_parts(builder=cog.toolkit.input_builder, message=message)
    assert rendered_urls == ["https://media.test/a.png"]

    # Same embed count, different image URL: the cache must not serve the stale part.
    message.embeds = [cast("Embed", _embed("https://media.test/b.png"))]
    await _attachment_parts(builder=cog.toolkit.input_builder, message=message)
    # order-contract: each awaited cache lookup renders its source before returning.
    assert rendered_urls == ["https://media.test/a.png", "https://media.test/b.png"]
