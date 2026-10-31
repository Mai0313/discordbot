"""Gemini Interactions API answer path for QA turns that must watch a YouTube video.

The runtime answer normally streams through the OpenAI Responses API on the LiteLLM proxy,
but that bridge HTTP-fetches every URL (a YouTube link comes back as HTML) so Gemini never
sees the video. The native Gemini Interactions API forwards a video URI untranslated for
Gemini to fetch server-side, so `RouteClassification.watch_video` swaps the one answer turn's
backend to it rather than running a second pass. This module is the whole swap surface: it
translates the already-assembled OpenAI-shaped answer input into the Interactions step schema,
appends the YouTube video to the current message, and adapts the Interactions stream events
back into the shapes `ResponseStreamer._consume` reads, so the preview / footer / markers /
voice / image / reply-edit machinery is reused unchanged. It calls Gemini direct (the
toolkit's `gemini_client` uses `gemini_api_key`, no proxy): the Interactions API is inherently
Gemini and the swap only fires on a Gemini answer model, so this is the one runtime answer
turn that does not ride the LiteLLM proxy. Importing the google-genai Interactions types here
is the documented carve-out for video ingestion.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, cast
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


def _kind_from_filename(filename: str) -> Literal["image", "video", "audio", "document"]:
    """Infers the Interactions content kind from `filename`'s extension, defaulting to `"document"`.

    Must stay aligned with `InlineRenderer`'s file MIME types so inlined and uploaded files map
    to the same kind.
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
    kind: Literal["image", "video", "audio", "document"], reference: str
) -> ContentParam | None:
    """Builds a media `ContentParam` from a Files API URI or a `data:` base64 string, or None when invalid.

    A `data:` reference is parsed into `data` and `mime_type` (or dropped if malformed) so it is
    never sent as a `uri` for the server to fetch.
    """
    inline_prefix = "data:"
    base64_marker = ";base64"
    if not reference.startswith(inline_prefix):
        return cast("ContentParam", {"type": kind, "uri": reference})
    header, _, payload = reference[len(inline_prefix) :].partition(",")
    mime_type = header.removesuffix(base64_marker).split(";")[0].strip()
    if not header.endswith(base64_marker) or not mime_type or not payload:
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


def _translate_part(part: ResponseInputContentParam) -> ContentParam | None:
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
        # there, and reading the other two alone would drop it without a log line.
        reference = (
            file_part.get("file_url") or file_part.get("file_id") or file_part.get("file_data")
        )
        if not reference:
            return None
        return _media_content(
            kind=_kind_from_filename(filename=file_part.get("filename") or ""), reference=reference
        )
    return None


def _translate_content(content: "str | object") -> list[ContentParam]:
    """Translates an OpenAI message's content (string shorthand or part list) into params."""
    if isinstance(content, str):
        return [TextContentParam(type="text", text=content)] if content else []
    parts: list[ContentParam] = []
    for part in cast("list[ResponseInputContentParam]", content):
        translated = _translate_part(part=part)
        if translated is not None:
            parts.append(translated)
    return parts


def to_interactions_input(answer_input: ResponseInputParam, youtube_url: str) -> list[StepParam]:
    """Returns `answer_input` translated into Interactions steps with `youtube_url` on the final user step.

    Folds system/developer blocks into user steps, coalesces consecutive same-role steps, and
    appends `youtube_url` as a `VideoContentParam` to the trailing user step.

    Args:
        answer_input (ResponseInputParam): Assembled OpenAI Responses input messages.
        youtube_url (str): YouTube URL to attach to the current user message step.
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
    # Agentic lets the model load only the segments or transcript a question needs instead of
    # every frame at 1 FPS; `slow_model`'s model page lists it as supported (read 2026-10-31).
    video_part = VideoContentParam(type="video", uri=youtube_url, processing="agentic")
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
    stream: "AsyncIterator[InteractionSSEEvent]",
) -> AsyncIterator[ResponseStreamEvent]:
    """Yields `ResponseStreamEvent`-shaped namespaces adapted from `stream` for `ResponseStreamer._consume`.

    Remaps text and reasoning deltas, emits token usage once on `interaction.completed` so
    `ResponseStreamer._consume`'s `+=` accumulation never double-counts, and converts an `"error"`
    event into `APIError`.

    Args:
        stream ("AsyncIterator[InteractionSSEEvent]"): Raw Gemini Interactions SSE stream.

    Raises:
        APIError: The Interactions stream emitted an `"error"` event.
    """
    model_name = ""
    # Fallback when `interaction.completed` omits usage and `total_usage` was reported on
    # `step.delta` instead; without it a stream that reports usage only per step leaves the
    # footer and the turn's token fields reading zero.
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
            # not buy is a retry: the SDK documents `Error.code` as "A URI that identifies the
            # error type", not a status, so an in-band failure here carries no HTTP status at
            # all and `is_retryable_llm_error` leaves it alone. Passing a decimal one through
            # is a hedge against the payload diverging from that doc, not a path anything has
            # been seen to take. What this backend does retry is what the SDK itself raises
            # for a transient status or a lost connection.
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
    client: genai.Client,
    model: str,
    system_instruction: str,
    steps: list[StepParam],
    effort: ReasoningEffort,
) -> AsyncIterator[ResponseStreamEvent]:
    """Yields adapted `ResponseStreamEvent` items for a YouTube-aware QA answer on the Gemini Interactions API.

    Calls Gemini directly with `url_context` and `google_search` grounding tools and `effort`
    mapped to `thinking_level`.

    Args:
        client (genai.Client): Direct-to-Google Gemini client.
        model (str): Gemini model name to dispatch.
        system_instruction (str): Runtime developer instructions for the turn.
        steps (list[StepParam]): Translated conversation and YouTube input steps.
        effort (ReasoningEffort): Reasoning effort grade (`"low"` or `"high"`).
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
            # loses the whole reply; narrowing ReasoningEffort to the enum here is safe only
            # while the grade stays inside every branch's set.
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
