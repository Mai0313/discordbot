"""Tests for the IMAGE and VIDEO routes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, cast
import asyncio

import pytest
import nextcord
from google.genai.types import FileState
from google.genai.errors import ClientError
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.media import LoadedMedia
from discordbot.cogs.gen_reply import streaming as streaming_module
from discordbot.typings.timeouts import ANSWER_STREAM_MAX_ATTEMPTS
from discordbot.cogs.gen_reply.context import ReplyContext
from discordbot.cogs.gen_reply.prompts import IMAGE_PROMPT, VIDEO_PROMPT
from discordbot.cogs.gen_reply.surface import WINDOW_EXPIRED_NOTICE
from discordbot.typings.context_budgets import MAX_VIDEO_REFERENCE_IMAGES
from discordbot.cogs.gen_reply.generation import INLINE_IMAGE_FILENAME, INLINE_VIDEO_FILENAME
from discordbot.cogs.gen_reply.status_marks import RETRY_HINT_EMOJI

from tests.helpers.casting import as_message, step_dicts, make_not_found, make_invalid_form_body
from tests.helpers.reply_fakes import (
    FakeGuild,
    FakeAuthor,
    FakeMessage,
    FakeAttachment,
    _cog,
    _recorded,
    _png_bytes,
    _media_routes,
    _recorded_video,
    _expiring_surface,
    _install_streamer,
    _no_retry_backoff,
    _ready_context_task,
    _default_turn_events,
    _mid_stream_unavailable,
)
from tests.helpers.link_sources import hosting_planner
from tests.helpers.logfire_capture import capture_logs

if TYPE_CHECKING:
    from pathlib import Path

    from aiohttp import ClientResponse


async def test_a_media_persona_reply_never_offers_the_deliverable_to_the_error_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The IMAGE route's streamer renders onto the delivered image, so it publishes nothing.

    Its own failure is swallowed, but a later one in the same turn reaches `on_message`, and
    the picture the user was handed must not be the message that gets an error embed written
    over it.
    """
    _no_retry_backoff(monkeypatch=monkeypatch)
    cog = _cog()
    message = FakeMessage(content="draw a cat", author=FakeAuthor(user_id=1))
    _recorded(cog).responses.stream_queue = [
        _mid_stream_unavailable()
    ] * ANSWER_STREAM_MAX_ATTEMPTS

    await _media_routes(cog=cog, message=as_message(fake=message)).handle_image(
        user_prompt="a cat", context_task=_ready_context_task()
    )

    # The image was delivered and every persona attempt then died on it.
    assert message.replies[-1].file is not None
    assert _recorded(cog).responses.create_streams.count(True) == ANSWER_STREAM_MAX_ATTEMPTS
    assert streaming_module.current_answer_streamer.get() is None


async def test_handle_image_reply_edits_attached_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """An attached image routes the IMAGE handler through images.edit with raw bytes."""
    cog = _cog()
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.input.get_supported_modalities", lambda model_name: {"image"}
    )
    logged = capture_logs(monkeypatch=monkeypatch, level="info")
    message = FakeMessage(content="改這張圖", author=FakeAuthor(user_id=1))
    message.attachments = [
        FakeAttachment(filename="pic.png", content_type="image/png", payload=_png_bytes())
    ]

    await _media_routes(cog=cog, message=as_message(fake=message)).handle_image(
        user_prompt="make it blue", context_task=_ready_context_task()
    )

    assert _recorded(cog).images.edit_calls == 1
    assert _recorded(cog).images.generate_calls == 0
    # The message replies to nothing, yet its own image is what makes this an edit.
    assert [
        fields["has_source_images"]
        for text, fields in logged
        if text == "gen_reply image generation start"
    ] == [True]


async def test_an_empty_prompt_falls_back_to_an_english_instruction() -> None:
    """With nothing refined, the image edit and the video render still get an English prompt."""
    cog = _cog()

    await cog.toolkit.image_generator.render(
        prompt="", end_user_id="user", image_bytes_list=[_png_bytes()]
    )
    await cog.toolkit.video_generator.render(prompt="", reference_image_sources=[])

    assert _recorded(cog).images.edit_prompts == [
        "Edit or refine according to the attached content."
    ]
    create_input = _recorded_video(cog).create_inputs[0]
    assert [part["text"] for part in create_input if part["type"] == "text"] == [
        "Generate a video from the message content."
    ]


async def test_handle_image_reply_refines_prompt_before_generate() -> None:
    """The prompt director expands the raw request and the refined prompt reaches images.generate."""
    cog = _cog()
    _recorded(cog).responses.refine_output_text = "a photorealistic tabby cat, studio lighting"
    message = FakeMessage(content="畫一隻貓", author=FakeAuthor(user_id=1))

    await _media_routes(cog=cog, message=as_message(fake=message)).handle_image(
        user_prompt="draw a cat", context_task=_ready_context_task()
    )

    # The refined prompt (not the raw request) reaches images.generate.
    assert _recorded(cog).images.generate_prompts == [
        "a photorealistic tabby cat, studio lighting"
    ]
    # Two responses.create calls: the non-streaming director first, then the streaming persona reply.
    assert _recorded(cog).responses.create_streams == [False, True]
    assert _recorded(cog).responses.create_models == [
        cog.toolkit.runtime_models.fast_model.name,
        cog.toolkit.runtime_models.fast_model.name,
    ]
    # The director runs on IMAGE_PROMPT with the grounding tools available.
    assert _recorded(cog).responses.create_instructions[0] == IMAGE_PROMPT
    assert _recorded(cog).responses.create_tools[0] == [{"googleSearch": {}}, {"urlContext": {}}]


async def test_handle_image_reply_refine_disabled_sends_raw_prompt() -> None:
    """With IMAGE_REFINE_PROMPT_ENABLED off, the raw request reaches images.generate with no director call."""
    cog = _cog()
    cog.config.image_refine_prompt_enabled = False
    message = FakeMessage(content="畫一隻貓", author=FakeAuthor(user_id=1))

    await _media_routes(cog=cog, message=as_message(fake=message)).handle_image(
        user_prompt="draw a cat", context_task=_ready_context_task()
    )

    # The raw prompt reaches images.generate; the only create is the streaming persona reply.
    assert _recorded(cog).images.generate_prompts == ["draw a cat"]
    assert _recorded(cog).responses.create_streams == [True]
    assert _recorded(cog).responses.create_models == [cog.toolkit.runtime_models.fast_model.name]


async def test_handle_image_reply_injects_only_user_memory() -> None:
    """The conversational reply carries the requester's memory and tone note, never the server memory."""
    cog = _cog()
    message = FakeMessage(content="畫一隻貓", author=FakeAuthor(user_id=1))
    context = ReplyContext(
        memory_block=EasyInputMessageParam(role="assistant", content="USER_MEM_MARKER"),
        server_memory_block=EasyInputMessageParam(role="assistant", content="SERVER_MEM_MARKER"),
        tone_block=EasyInputMessageParam(role="assistant", content="TONE_MARKER"),
    )

    async def _ready() -> ReplyContext:
        """Hands the prepared context to the handler."""
        return context

    await _media_routes(cog=cog, message=as_message(fake=message)).handle_image(
        user_prompt="draw a cat", context_task=asyncio.create_task(coro=_ready())
    )

    # The streamed reply is the last create; the user memory block rides in it, then the
    # tone note, mirroring the answer path's order; the server memory never does.
    reply_input = step_dicts(steps=_recorded(cog).responses.create_inputs[-1])
    contents = [block.get("content") for block in reply_input]
    assert "USER_MEM_MARKER" in contents
    assert "TONE_MARKER" in contents
    assert contents.index("USER_MEM_MARKER") < contents.index("TONE_MARKER")
    assert "SERVER_MEM_MARKER" not in contents


async def test_handle_image_reply_retries_the_persona_stream_without_captioning_the_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The persona streamer renders onto the DELIVERED image, which is not a status surface.

    A retry notice there captions a finished image with a promise of more to come, and on the
    path where every attempt is spent the failure is swallowed, so nothing would ever take it
    back. This route is silent to the user by design; only the retry itself belongs to it.
    """
    _no_retry_backoff(monkeypatch=monkeypatch)
    cog = _cog()
    message = FakeMessage(content="畫一隻貓", author=FakeAuthor(user_id=1))
    _recorded(cog).responses.stream_queue = [
        _mid_stream_unavailable(),
        list(_default_turn_events()),
    ]

    await _media_routes(cog=cog, message=as_message(fake=message)).handle_image(
        user_prompt="draw a cat", context_task=_ready_context_task()
    )

    # The retry happened (two streaming dispatches) and the persona reply still landed.
    assert _recorded(cog).responses.create_streams.count(True) == 2
    delivered = message.replies[0]
    assert (delivered.content or "").startswith("done")
    # But nothing announced it: not on the image, and not on the user's message.
    written = [delivered.content or "", *delivered.edits]
    assert all(RETRY_HINT_EMOJI not in text for text in written)
    assert RETRY_HINT_EMOJI not in message.added_reactions


async def test_handle_image_reply_best_effort_when_reply_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure producing the conversational reply still leaves the image delivered."""
    cog = _cog()
    message = FakeMessage(content="畫一隻貓", author=FakeAuthor(user_id=1))

    _install_streamer(monkeypatch=monkeypatch, reply=RuntimeError("stream boom"))

    await _media_routes(cog=cog, message=as_message(fake=message)).handle_image(
        user_prompt="draw a cat", context_task=_ready_context_task()
    )

    # The image is delivered even though the reply stream raised; the error never surfaced.
    assert message.replies[-1].file is not None


async def test_handle_image_reply_hosts_oversized_image_on_separate_message(
    tmp_path: Path,
) -> None:
    """An image too big to upload is hosted as a URL; the persona reply rides a separate message."""
    cog = _cog()
    cog.__dict__["media_delivery"] = hosting_planner(serve_dir=tmp_path)
    message = FakeMessage(content="畫一隻貓", author=FakeAuthor(user_id=1))
    message.guild = FakeGuild(filesize_limit=4)  # tiny ceiling -> the generated PNG is oversized

    await _media_routes(cog=cog, message=as_message(fake=message)).handle_image(
        user_prompt="draw a cat", context_task=_ready_context_task()
    )

    # Two messages: the hosted-URL deliverable (no attachment) and the separate persona reply.
    assert len(message.replies) == 2
    url_msg = message.replies[0]
    assert url_msg.file is None
    assert any(
        line.startswith("https://media.test/") for line in (url_msg.content or "").splitlines()
    )
    # The persona reply streamed onto its own message and never clobbered the URL.
    assert "media.test" not in (message.replies[1].content or "")


async def test_handle_image_reply_hosted_persona_failure_deletes_orphan_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hosted oversize image: a failed persona stream deletes the fresh base, leaving no orphan."""
    cog = _cog()
    cog.__dict__["media_delivery"] = hosting_planner(serve_dir=tmp_path)
    message = FakeMessage(content="畫一隻貓", author=FakeAuthor(user_id=1))
    message.guild = FakeGuild(
        filesize_limit=4
    )  # oversize -> hosted URL deliverable (reply is None)

    _install_streamer(monkeypatch=monkeypatch, reply=RuntimeError("stream boom"))

    await _media_routes(cog=cog, message=as_message(fake=message)).handle_image(
        user_prompt="draw a cat", context_task=_ready_context_task()
    )

    # replies[0] is the hosted-URL deliverable (kept); replies[1] is the bare persona base (deleted).
    assert len(message.replies) == 2
    assert message.replies[0].deleted is False
    assert any(
        line.startswith("https://media.test/")
        for line in (message.replies[0].content or "").splitlines()
    )
    assert message.replies[1].deleted is True  # the orphaned persona base was cleaned up


async def test_handle_image_reply_raises_when_oversized_and_hosting_off() -> None:
    """IMAGE route, hosting off + oversize: the native attach is attempted and its error propagates.

    With no host available the deliverable cannot degrade to a URL, so `MediaReplyRoutes._deliver`
    falls through to the native attach (which Discord 413s on oversize); that error must stay on the
    route's outer hard-fail path, never a silent drop. A FakeMessage models the 413 via reply_error.
    """
    cog = _cog()
    message = FakeMessage(content="畫一隻貓", author=FakeAuthor(user_id=1))
    message.guild = FakeGuild(filesize_limit=4)  # tiny ceiling -> the generated PNG is oversized
    # The native attach of an oversized file 413s on real Discord; the fake raises it on reply.
    message.reply_error = nextcord.HTTPException(
        cast("ClientResponse", SimpleNamespace(status=413, reason="Payload Too Large")),
        {"code": 40005, "message": "Request entity too large"},
    )

    with pytest.raises(nextcord.HTTPException):
        await _media_routes(cog=cog, message=as_message(fake=message)).handle_image(
            user_prompt="draw a cat", context_task=_ready_context_task()
        )


@pytest.mark.parametrize("error", [make_invalid_form_body(), make_not_found()])
@pytest.mark.parametrize(
    argnames=("route", "filename"),
    argvalues=[("IMAGE", INLINE_IMAGE_FILENAME), ("VIDEO", INLINE_VIDEO_FILENAME)],
)
async def test_media_lands_unparented_when_its_source_was_deleted_meanwhile(
    route: Literal["IMAGE", "VIDEO"], filename: str, error: nextcord.HTTPException
) -> None:
    """Media made for a message deleted while it generated still lands, and the reply follows.

    The generation is already paid for and the media message mentions its author, so it means
    something without the message it answered, as the text answer already does.
    """
    cog = _cog()
    message = FakeMessage(content="畫一隻貓", author=FakeAuthor(user_id=1))
    message.reply_error = error
    routes = _media_routes(cog=cog, message=as_message(fake=message))
    handle = routes.handle_image if route == "IMAGE" else routes.handle_video

    await handle(user_prompt="draw a cat", context_task=_ready_context_task())

    assert message.replies == []
    delivered = message.channel.sent[0]
    assert delivered.file is not None
    assert delivered.file.filename == filename
    assert (delivered.content or "").startswith("done")


async def test_a_hosted_image_and_its_reply_land_unparented_when_the_source_was_deleted(
    tmp_path: Path,
) -> None:
    """The hosted URL and the persona reply's own message both land when the source is gone."""
    cog = _cog()
    cog.__dict__["media_delivery"] = hosting_planner(serve_dir=tmp_path)
    message = FakeMessage(content="畫一隻貓", author=FakeAuthor(user_id=1))
    message.guild = FakeGuild(filesize_limit=4)  # tiny ceiling -> the generated PNG is oversized
    message.reply_error = make_not_found()

    await _media_routes(cog=cog, message=as_message(fake=message)).handle_image(
        user_prompt="draw a cat", context_task=_ready_context_task()
    )

    assert message.replies == []
    url_msg, persona = message.channel.sent
    assert any(
        line.startswith("https://media.test/") for line in (url_msg.content or "").splitlines()
    )
    assert (persona.content or "").startswith("done")


async def test_handle_video_reply_oversized_upload_failure_leaves_no_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oversized video hosted as a URL: a failed Files-API upload leaves no empty persona message."""
    cog = _cog()
    cog.__dict__["media_delivery"] = hosting_planner(serve_dir=tmp_path)

    async def _no_upload(**kwargs: object) -> None:
        """Simulates the post-delivery Files-API upload failing."""
        del kwargs

    monkeypatch.setattr("discordbot.cogs.gen_reply.media_reply.upload_as_input_file", _no_upload)
    message = FakeMessage(content="拍一段影片", author=FakeAuthor(user_id=1))
    message.guild = FakeGuild(filesize_limit=1)  # below the 3-byte fake clip -> oversized

    await _media_routes(cog=cog, message=as_message(fake=message)).handle_video(
        user_prompt="video", context_task=_ready_context_task()
    )

    # Only the hosted-URL message exists; the upload failed so no bare persona-base was orphaned.
    assert len(message.replies) == 1
    url_msg = message.replies[0]
    assert url_msg.file is None
    assert any(
        line.startswith("https://media.test/") for line in (url_msg.content or "").splitlines()
    )


async def test_handle_video_reply_refines_prompt_before_render() -> None:
    """The prompt director expands the raw request and the refined prompt reaches omni."""
    cog = _cog()
    _recorded(cog).responses.refine_output_text = "a cat leaping in slow motion, camera pan"

    message = FakeMessage(content="拍一段影片", author=FakeAuthor(user_id=1))

    await _media_routes(cog=cog, message=as_message(fake=message)).handle_video(
        user_prompt="video", context_task=_ready_context_task()
    )

    # The director runs on VIDEO_PROMPT first, then the streaming reply about the video.
    assert _recorded(cog).responses.create_streams == [False, True]
    assert _recorded(cog).responses.create_models == [
        cog.toolkit.runtime_models.fast_model.name,
        cog.toolkit.runtime_models.fast_model.name,
    ]
    assert _recorded(cog).responses.create_instructions[0] == VIDEO_PROMPT
    # The reply (the last create) watches the generated video: referenced as an input_file part.
    reply_parts = step_dicts(steps=_recorded(cog).responses.create_inputs[-1])[-1]["content"]
    assert any(part.get("type") == "input_file" for part in reply_parts)
    # No attachments: the refined prompt reaches omni as input text; the task is omitted so omni
    # infers text_to_video, and the fixed 16:9 aspect ratio is still sent for pure text.
    create_input = _recorded_video(cog).create_inputs[0]
    assert [part["text"] for part in create_input if part["type"] == "text"] == [
        "a cat leaping in slow motion, camera pan"
    ]
    assert not any(part["type"] == "image" for part in create_input)
    assert _recorded_video(cog).create_configs[0] is None
    assert _recorded_video(cog).create_response_formats[0]["aspect_ratio"] == "16:9"
    assert message.replies[-1].file is not None


async def test_handle_video_reply_refine_disabled_sends_raw_prompt() -> None:
    """With VIDEO_REFINE_PROMPT_ENABLED off, the raw request reaches omni with no director call."""
    cog = _cog()
    cog.config.video_refine_prompt_enabled = False

    message = FakeMessage(content="拍一段影片", author=FakeAuthor(user_id=1))

    await _media_routes(cog=cog, message=as_message(fake=message)).handle_video(
        user_prompt="video", context_task=_ready_context_task()
    )

    # The raw prompt reaches omni as input text; the only create is the streaming persona reply
    # (no non-streaming refine call).
    create_input = _recorded_video(cog).create_inputs[0]
    assert [part["text"] for part in create_input if part["type"] == "text"] == ["video"]
    assert _recorded(cog).responses.create_streams == [True]
    assert _recorded(cog).responses.create_models == [cog.toolkit.runtime_models.fast_model.name]


async def _one_source_clip(builder: object, message: object) -> list[LoadedMedia]:
    """Stands in for `get_video_sources`: the message carries one raw source clip."""
    del builder, message
    return [LoadedMedia(data=b"clip", mime_type="video/mp4")]


async def test_handle_video_reply_edits_source_video(monkeypatch: pytest.MonkeyPatch) -> None:
    """A source video is edited in place: uploaded and sent to omni with task=edit, no director."""
    cog = _cog()
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.input.MessageInputBuilder.get_video_sources", _one_source_clip
    )
    message = FakeMessage(content="把這部影片做成新的", author=FakeAuthor(user_id=1))

    await _media_routes(cog=cog, message=as_message(fake=message)).handle_video(
        user_prompt="make it snowy", context_task=_ready_context_task()
    )

    create_input = _recorded_video(cog).create_inputs[0]
    # The actual clip rides as a video part (uploaded to the Files API), edited in place.
    assert any(part["type"] == "video" for part in create_input)
    assert _recorded_video(cog).create_configs[0]["video_config"]["task"] == "edit"
    # The director is skipped for edits, so the raw request reaches omni unchanged and the proxy
    # only ever runs the streaming persona reply (never a non-streaming refine call).
    assert [part["text"] for part in create_input if part["type"] == "text"] == ["make it snowy"]
    assert _recorded(cog).responses.create_streams == [True]
    # An edit keeps the source clip's ratio, so no aspect_ratio is sent (omni 400s it otherwise).
    assert "aspect_ratio" not in _recorded_video(cog).create_response_formats[0]


_SOURCE_UPLOAD_REFUSED = ClientError(403, {"error": {"message": "PERMISSION_DENIED"}}, None)


@pytest.mark.parametrize(
    ("uploaded", "expected"),
    [
        (_SOURCE_UPLOAD_REFUSED, _SOURCE_UPLOAD_REFUSED),
        (
            SimpleNamespace(name=None, uri=None, state=FileState.ACTIVE),
            RuntimeError("Files API upload of source.mp4 returned no resource name"),
        ),
        (
            SimpleNamespace(name="files/vid", uri=None, state=FileState.PROCESSING),
            RuntimeError("Source video did not become ACTIVE before the deadline"),
        ),
        (
            SimpleNamespace(name="files/vid", uri=None, state=FileState.FAILED),
            RuntimeError("Files API upload of source.mp4 failed: state=FileState.FAILED"),
        ),
    ],
    ids=["sdk-error", "no-name", "never-active", "failed-state"],
)
async def test_a_failed_source_video_upload_reaches_the_route_caller_unchanged(
    monkeypatch: pytest.MonkeyPatch, uploaded: object, expected: Exception
) -> None:
    """An edit whose clip never uploads fails the VIDEO route with that very error, before omni.

    The clip is the deliverable's input, so there is nothing to degrade to: the route's caller
    gets the SDK's own exception, or the one naming which step of the upload broke, and it is
    what the user is shown.
    """
    cog = _cog()

    async def upload(file: object, config: dict[str, str]) -> object:
        """Refuses the upload or hands back the file under test."""
        del file, config
        if isinstance(uploaded, Exception):
            raise uploaded
        return uploaded

    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.input.MessageInputBuilder.get_video_sources", _one_source_clip
    )
    monkeypatch.setattr("discordbot.cogs.gen_reply.generation.FILES_READY_TIMEOUT_SECONDS", 0.0)
    _recorded_video(cog).aio.files.upload = upload
    message = FakeMessage(content="把這部影片做成新的", author=FakeAuthor(user_id=1))

    with pytest.raises(type(expected)) as raised:
        await _media_routes(cog=cog, message=as_message(fake=message)).handle_video(
            user_prompt="make it snowy", context_task=_ready_context_task()
        )

    assert type(raised.value) is type(expected)
    assert str(raised.value) == str(expected)
    assert _recorded_video(cog).create_inputs == []


async def test_handle_video_reply_passes_reference_images() -> None:
    """Attached images ride as capped reference images with a real mime; task inferred."""
    cog = _cog()

    message = FakeMessage(content="把這些做成影片", author=FakeAuthor(user_id=1))
    message.attachments = [
        FakeAttachment(filename=f"pic{index}.png", content_type="image/png", payload=_png_bytes())
        for index in range(MAX_VIDEO_REFERENCE_IMAGES + 1)
    ]

    await _media_routes(cog=cog, message=as_message(fake=message)).handle_video(
        user_prompt="video", context_task=_ready_context_task()
    )

    # Images are capped and each MUST carry a real, non-empty image mime (omni 400s an empty mime,
    # the reported bug); the task is omitted so omni infers image_to_video vs reference_to_video.
    create_input = _recorded_video(cog).create_inputs[0]
    image_parts = [part for part in create_input if part["type"] == "image"]
    assert len(image_parts) == MAX_VIDEO_REFERENCE_IMAGES
    assert all(part["data"] for part in image_parts)
    assert all(part.get("mime_type", "").startswith("image/") for part in image_parts)
    assert _recorded_video(cog).create_configs[0] is None


async def test_handle_video_reply_single_image_sends_mime_no_aspect_ratio() -> None:
    """A lone image (the reported crash case) sends a real mime, no aspect ratio, task inferred."""
    cog = _cog()

    message = FakeMessage(content="讓這張動起來", author=FakeAuthor(user_id=1))
    message.attachments = [
        FakeAttachment(filename="pic.png", content_type="image/png", payload=_png_bytes())
    ]

    await _media_routes(cog=cog, message=as_message(fake=message)).handle_video(
        user_prompt="video", context_task=_ready_context_task()
    )

    # The single image carries its mime (this is exactly what was empty before, causing the 400);
    # no aspect_ratio is sent (omni may pick image_to_video, which follows the source frame's ratio),
    # and the task is omitted so omni infers image_to_video.
    create_input = _recorded_video(cog).create_inputs[0]
    image_parts = [part for part in create_input if part["type"] == "image"]
    assert len(image_parts) == 1
    assert image_parts[0].get("mime_type", "").startswith("image/")
    assert "aspect_ratio" not in _recorded_video(cog).create_response_formats[0]
    assert _recorded_video(cog).create_configs[0] is None


class _NeverFinishes:
    """A generator whose render outlasts any window a `/ask` turn could give it."""

    async def render(self, **kwargs: object) -> bytes:
        """Sleeps far past the bound under test rather than returning."""
        del kwargs
        await asyncio.sleep(60)
        return b""


async def test_a_video_outliving_the_ask_window_says_so_instead_of_hanging() -> None:
    """Out of surface before out of work, the route stops while it can still be heard.

    The render is left running rather than made to fail: what is pinned here is that the route
    gives up on it. A clip that lands after the interaction token dies is delivered into a 404,
    and so is the notice that would have explained it, so the turn ends in silence under a
    thinking state that never resolves.
    """
    cog = _cog()
    cog.toolkit.__dict__["video_generator"] = _NeverFinishes()
    message = as_message(fake=FakeMessage(content="拍一段影片", author=FakeAuthor(user_id=1)))

    with pytest.raises(TimeoutError) as raised:
        await _media_routes(
            cog=cog, message=message, surface=_expiring_surface(message=message, seconds_left=0.05)
        ).handle_video(user_prompt="video", context_task=_ready_context_task())

    # The message is the whole point: a bare TimeoutError reaches the user as an empty code block.
    assert str(raised.value) == WINDOW_EXPIRED_NOTICE


async def test_an_image_outliving_the_ask_window_says_so_too() -> None:
    """The IMAGE route shares the failure and the fix: its render carries no bound of its own."""
    cog = _cog()
    cog.toolkit.__dict__["image_generator"] = _NeverFinishes()
    message = as_message(fake=FakeMessage(content="畫一隻貓", author=FakeAuthor(user_id=1)))

    with pytest.raises(TimeoutError) as raised:
        await _media_routes(
            cog=cog, message=message, surface=_expiring_surface(message=message, seconds_left=0.05)
        ).handle_image(user_prompt="draw a cat", context_task=_ready_context_task())

    assert str(raised.value) == WINDOW_EXPIRED_NOTICE


async def test_a_generators_own_timeout_is_not_blamed_on_the_ask_window() -> None:
    """A slow provider inside a healthy window keeps its own failure, so the window is not blamed.

    `VideoGenerator.render` bounds itself with `VIDEO_RENDER_TIMEOUT_SECONDS` and raises the very
    same `TimeoutError`, which is why the route reads `expired()` rather than the exception type.
    Reading the type instead would report every slow render as Discord having closed the command.
    """
    cog = _cog()

    class _TimesOutOnItsOwn:
        """A render that hits its own bound while the surface has plenty of time left."""

        async def render(self, **kwargs: object) -> bytes:
            """Fails the way `render` does past `VIDEO_RENDER_TIMEOUT_SECONDS`."""
            del kwargs
            raise TimeoutError

    cog.toolkit.__dict__["video_generator"] = _TimesOutOnItsOwn()
    message = as_message(fake=FakeMessage(content="拍一段影片", author=FakeAuthor(user_id=1)))

    with pytest.raises(TimeoutError) as raised:
        await _media_routes(
            cog=cog, message=message, surface=_expiring_surface(message=message, seconds_left=300)
        ).handle_video(user_prompt="video", context_task=_ready_context_task())

    assert str(raised.value) != WINDOW_EXPIRED_NOTICE


@pytest.mark.parametrize(
    argnames=("route", "generator"),
    argvalues=[("IMAGE", "image_generator"), ("VIDEO", "video_generator")],
)
async def test_a_failed_media_generation_drains_the_speculative_context(
    route: Literal["IMAGE", "VIDEO"], generator: str
) -> None:
    """A route that fails before consuming the context it was handed cancels and drains it.

    Nothing else will: the pipeline hands the build over to the media route and stops tracking
    it, so a context left pending here keeps reading history and uploading attachments for a
    turn that has already failed.
    """
    cog = _cog()

    class _RefusesToRender:
        """A render that fails outright, well inside the window."""

        async def render(self, **kwargs: object) -> bytes:
            """Fails the way a refused generation does."""
            del kwargs
            raise RuntimeError("render refused")

    cog.toolkit.__dict__[generator] = _RefusesToRender()
    release = asyncio.Event()

    async def pending_context() -> ReplyContext:
        """A context build still in flight when the generation fails."""
        await release.wait()
        return ReplyContext()

    context_task = asyncio.create_task(coro=pending_context())
    message = as_message(fake=FakeMessage(content="畫一隻貓", author=FakeAuthor(user_id=1)))
    routes = _media_routes(cog=cog, message=message)
    handle = routes.handle_image if route == "IMAGE" else routes.handle_video

    with pytest.raises(RuntimeError, match="render refused"):
        await handle(user_prompt="draw a cat", context_task=context_task)

    assert context_task.cancelled()
