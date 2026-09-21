"""Gemini Interactions API answer path for QA turns that must watch a YouTube video.

The runtime answer normally streams through the OpenAI Responses API on the LiteLLM proxy,
but that bridge HTTP-fetches every URL (a YouTube link comes back as HTML) so Gemini never
sees the video. The native Gemini Interactions API forwards a video URI untranslated for
Gemini to fetch server-side, so a YouTube QA turn swaps to it. This module is the whole
swap surface: it translates the already-assembled OpenAI-shaped answer input into the
Interactions step schema, appends the YouTube video to the current message, and adapts the
Interactions stream events back into the shapes `ResponseStreamer._consume` reads, so the
preview / footer / markers / voice / image / reply-edit machinery is reused unchanged. It
calls Gemini DIRECT (the cog's `gemini_client` uses `gemini_api_key`, no proxy): the
Interactions API is inherently Gemini and the swap only fires on a Gemini answer model, so
this is the one runtime answer turn that does not ride the LiteLLM proxy. Importing the
google-genai Interactions types here is the documented carve-out for video ingestion.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, Literal, cast
from collections.abc import AsyncIterator

from google import genai
import logfire
from google.genai.errors import APIError
from openai.types.responses import ResponseStreamEvent
from google.genai.interactions import (
    StepParam,
    URLContext,
    ContentParam,
    GoogleSearch,
    ThinkingLevel,
    AllowlistParam,
    EnvironmentParam,
    TextContentParam,
    VideoContentParam,
    UserInputStepParam,
    AllowlistEntryParam,
    InteractionSSEEvent,
    ModelOutputStepParam,
    GenerationConfigParam,
)
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam
from openai.types.responses.response_input_content_param import ResponseInputContentParam

if TYPE_CHECKING:
    from openai.types.responses.response_input_file_param import ResponseInputFileParam
    from openai.types.responses.response_input_text_param import ResponseInputTextParam
    from openai.types.responses.response_input_image_param import ResponseInputImageParam


_INLINE_PREFIX: Final = "data:"
_BASE64_MARKER: Final = ";base64"


def _kind_from_filename(filename: str) -> Literal["image", "video", "audio", "document"]:
    """Infers the Interactions content kind from a file's extension.

    An `input_file` part carries a Files API URI and no MIME, so the content-param type comes
    from the original filename; an unknown extension falls back to document (best effort, never
    raises). An inlined part does carry its own MIME, and the two agree for everything that
    reaches here because `InlineRenderer` inlines only PDFs as files — but they are two sources
    for one answer, so a new inline file type has to be checked against this list rather than
    assumed to land on the same kind.
    """
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix in {"mp4", "mov", "webm", "avi", "mpeg", "mpg", "flv", "wmv", "3gp", "3gpp", "mkv"}:
        return "video"
    if suffix in {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "heic", "heif"}:
        return "image"
    if suffix in {"mp3", "wav", "ogg", "m4a", "aac", "flac", "opus", "aiff", "weba"}:
        return "audio"
    return "document"


def _media_content(
    *, kind: Literal["image", "video", "audio", "document"], reference: str
) -> ContentParam | None:
    """Builds one media content param from a Files API URI or from inlined bytes.

    Every Interactions content param takes both shapes — `uri` for something the server
    fetches, `data` beside `mime_type` for bytes sent with the request — and which one a
    reference is depends on who rendered it. The Files API path hands over an https URI, while
    `InlineRenderer`, which `file_api_enabled=false` selects for every provider including
    Gemini, hands over a base64 `data:` one. Putting the second in `uri` tells the server to go
    and fetch a URI that is the file (#661).

    A `data:` reference therefore never reaches `uri`, whatever it looks like. Recognising the
    shape and falling through on anything else would reintroduce exactly that bug through a
    narrower door, and two narrow doors are open: the image path hands over whatever MIME
    Discord reported, parameters and all, where `attachment_mime` would have stripped them, and
    an empty attachment inlines to a header with no payload at all. Neither is worth sending, so
    an unusable one is dropped and said out loud instead.
    """
    if not reference.startswith(_INLINE_PREFIX):
        return cast("ContentParam", {"type": kind, "uri": reference})
    header, _, payload = reference[len(_INLINE_PREFIX) :].partition(",")
    mime_type = header.removesuffix(_BASE64_MARKER).split(";")[0].strip()
    if not header.endswith(_BASE64_MARKER) or not mime_type or not payload:
        logfire.warn(
            "dropping an inlined attachment the Interactions turn cannot carry",
            kind=kind,
            header=header[:100],
            payload_length=len(payload),
        )
        return None
    # Each `mime_type` is typed as a Literal of what that kind accepts, and this one came off a
    # Discord attachment, so it is the server that gets to reject it rather than `ty`.
    return cast("ContentParam", {"type": kind, "data": payload, "mime_type": mime_type})


def _translate_part(*, part: ResponseInputContentParam) -> ContentParam | None:
    """Translates one OpenAI input content part into an Interactions content param.

    Returns None for an empty or unmappable part so the caller drops it instead of breaking
    the request.
    """
    part_type = part["type"]
    if part_type == "input_text":
        text = cast("ResponseInputTextParam", part)["text"]
        return TextContentParam(type="text", text=text) if text else None
    if part_type == "input_image":
        image_part = cast("ResponseInputImageParam", part)
        reference = image_part.get("image_url") or image_part.get("file_id")
        return _media_content(kind="image", reference=reference) if reference else None
    if part_type == "input_file":
        file_part = cast("ResponseInputFileParam", part)
        # `file_data` is the inlined half: a PDF rendered without the Files API arrives only
        # there, and reading the other two alone dropped it with no record anywhere (#661).
        reference = (
            file_part.get("file_url") or file_part.get("file_id") or file_part.get("file_data")
        )
        if not reference:
            return None
        return _media_content(
            kind=_kind_from_filename(filename=file_part.get("filename") or ""), reference=reference
        )
    return None


def _translate_content(*, content: "str | object") -> list[ContentParam]:
    """Translates an OpenAI message's content (string shorthand or part list) into params."""
    if isinstance(content, str):
        return [TextContentParam(type="text", text=content)] if content else []
    parts: list[ContentParam] = []
    for part in cast("list[ResponseInputContentParam]", content):
        translated = _translate_part(part=part)
        if translated is not None:
            parts.append(translated)
    return parts


def to_interactions_input(
    answer_input: ResponseInputParam, *, youtube_url: str
) -> list[StepParam]:
    """Translates the assembled OpenAI answer input into Interactions steps.

    Each OpenAI message becomes a user-input or model-output step (system / developer blocks
    fold into a user step, since the Interactions schema has no system step; the developer
    instructions ride the separate `system_instruction` field instead). Consecutive same-role
    steps are coalesced so the request never trips a strict role-alternation check and matches
    how Gemini merges same-role turns. The YouTube video is appended as a `VideoContentParam`
    to the last user step, which is the current message (kept last by the caller), so the video
    sits with the question it is about.
    """
    entries: list[tuple[str, list[ContentParam]]] = []
    for raw in answer_input:
        item = cast("EasyInputMessageParam", raw)
        out_role = "model" if item.get("role", "user") == "assistant" else "user"
        parts = _translate_content(content=item.get("content", ""))
        if not parts:
            continue
        if entries and entries[-1][0] == out_role:
            entries[-1][1].extend(parts)
        else:
            entries.append((out_role, parts))
    video_part = VideoContentParam(type="video", uri=youtube_url)
    if entries and entries[-1][0] == "user":
        entries[-1][1].append(video_part)
    else:
        entries.append(("user", [video_part]))
    steps: list[StepParam] = []
    for out_role, parts in entries:
        if out_role == "user":
            steps.append(UserInputStepParam(type="user_input", content=parts))
        else:
            steps.append(ModelOutputStepParam(type="model_output", content=parts))
    return steps


async def adapt_interactions_stream(
    *, stream: "AsyncIterator[InteractionSSEEvent]"
) -> AsyncIterator[ResponseStreamEvent]:
    """Adapts Interactions stream events into the shapes `ResponseStreamer._consume` reads.

    `_consume` switches on `response.type` and reads `response.delta` /
    `response.response.model` / `response.response.usage.{input,output}_tokens` /
    `response.response.output`. The Interactions stream uses different names (`event_type`,
    `delta.text`, `interaction.model`, `interaction.usage.total_*_tokens`), so each event is
    remapped onto a minimal namespace with the OpenAI-Responses field names. Usage is emitted
    exactly once on `interaction.completed` because `_consume` accumulates it with `+=` over a
    token seed from the earlier selection call; a per-step emit would double-count.
    """
    model_name = ""
    # The fallback for a completed event carrying no usage of its own. google-genai 2.22 moved
    # `total_usage` off that event onto `step.delta`, so it has to be caught on the way past
    # rather than read at the end; without it a stream that reports usage only per step leaves
    # the footer and the turn's token fields reading zero.
    streamed_usage = None
    # Branch on `event.event_type` directly (not a copied local) so the discriminated
    # InteractionSSEEvent union narrows to the member that carries the field being read.
    async for event in stream:
        if event.event_type == "interaction.created":
            model_name = event.interaction.model or ""
            yield cast(
                "ResponseStreamEvent",
                SimpleNamespace(
                    type="response.created",
                    response=SimpleNamespace(model=model_name, usage=None, output=None),
                ),
            )
        elif event.event_type == "step.delta":
            if event.metadata is not None and event.metadata.total_usage is not None:
                streamed_usage = event.metadata.total_usage
            delta = event.delta
            if delta.type == "text":
                yield cast(
                    "ResponseStreamEvent",
                    SimpleNamespace(type="response.output_text.delta", delta=delta.text),
                )
            elif delta.type == "thought_summary":
                text = getattr(delta.content, "text", "") if delta.content is not None else ""
                if text:
                    yield cast(
                        "ResponseStreamEvent",
                        SimpleNamespace(type="response.reasoning_summary_text.delta", delta=text),
                    )
        elif event.event_type == "interaction.completed":
            # Usage rides the completed interaction; `interaction.usage` is optional on a
            # streaming payload, so the last `total_usage` seen above stands in for it.
            usage = event.interaction.usage or streamed_usage
            usage_ns = (
                SimpleNamespace(
                    input_tokens=usage.total_input_tokens or 0,
                    output_tokens=usage.total_output_tokens or 0,
                )
                if usage is not None
                else None
            )
            yield cast(
                "ResponseStreamEvent",
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        model=(event.interaction.model or model_name),
                        usage=usage_ns,
                        # None, not []: this surface reports grounding in a shape the Responses
                        # event cannot carry, and the streamer logs "not reported" rather than a
                        # zero that would read as an ungrounded answer.
                        output=None,
                    ),
                ),
            )
        elif event.event_type == "error":
            # Typed rather than a bare RuntimeError so `extract_friendly_error` can show the
            # user what the provider actually said instead of this event's repr. What it does
            # NOT buy is a retry: the SDK documents `Error.code` as "A URI that identifies the
            # error type", not a status, so an in-band failure here carries no HTTP status at
            # all and `is_retryable_llm_error` leaves it alone. Passing a decimal one through
            # is a hedge against the payload diverging from that doc, not a path anything has
            # been seen to take -- on this backend only a failure `interactions.create` itself
            # raises as a typed genai error is actually retried today.
            failure = event.error
            code = failure.code if failure is not None else None
            raise APIError(
                code=int(code) if code is not None and code.isdigit() else 0,
                response_json={
                    "error": {
                        "message": (failure.message if failure is not None else None)
                        or f"Gemini interactions stream error: {event!r}",
                        "code": code,
                    }
                },
            )


async def create_interactions_answer_stream(
    *,
    client: genai.Client,
    model: str,
    system_instruction: str,
    steps: list[StepParam],
    effort: ReasoningEffort,
) -> AsyncIterator[ResponseStreamEvent]:
    """Streams a YouTube-aware QA answer through the Gemini Interactions API.

    Mirrors the Responses answer call (same model, system instruction, built-in grounding
    tools, effort-as-thinking-level) but lets Gemini watch the linked video, then yields the
    adapted stream so the shared `ResponseStreamer` consumes it unchanged. `extra_body` is
    intentionally omitted (the interactions client does not support it). The call goes direct
    to Google, so no LiteLLM end-user header is sent.
    """
    responses = await client.aio.interactions.create(
        model=model,
        system_instruction=system_instruction,
        input=steps,
        # Ad-hoc remote sandbox, not an environment ID: a bare "remote" string is read as an
        # existing environment's id. The `*` allowlist leaves outbound networking unrestricted so
        # the server-side tools and the YouTube fetch can reach any domain.
        environment=EnvironmentParam(
            type="remote", network=AllowlistParam(allowlist=[AllowlistEntryParam(domain="*")])
        ),
        generation_config=GenerationConfigParam(
            # effort is the route grade copied onto slow_model (always low or high here), and
            # the set Gemini accepts is per-model, looked up rather than assumed (see
            # `ModelSettings.effort`). A level the model does not list is refused outright and
            # loses the whole reply, which is what pinned `slow_model` to snapshots (#459);
            # narrowing ReasoningEffort to the enum here is safe only while the grade stays
            # inside every branch's set.
            thinking_level=cast("ThinkingLevel", effort),
            thinking_summaries="auto",
        ),
        tools=[
            URLContext(type="url_context"),
            GoogleSearch(search_types=["web_search"], type="google_search"),
        ],
        stream=True,
    )
    # `stream=True` returns an async event stream; the overload still types it as a union with
    # the non-streaming Interaction, so narrow to the iterator the adapter consumes.
    stream = cast("AsyncIterator[InteractionSSEEvent]", responses)
    async for event in adapt_interactions_stream(stream=stream):
        yield event
