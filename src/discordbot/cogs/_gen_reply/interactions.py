"""Gemini Interactions API answer path for QA turns that must watch a YouTube video.

The runtime answer normally streams through the OpenAI Responses API on the LiteLLM proxy,
but that bridge HTTP-fetches every URL (a YouTube link comes back as HTML) so Gemini never
sees the video. The native Gemini Interactions API forwards a video URI untranslated for
Gemini to fetch server-side, so a YouTube QA turn swaps to it. This module is the whole
swap surface: it translates the already-assembled OpenAI-shaped answer input into the
Interactions step schema, appends the YouTube video to the current message, and adapts the
Interactions stream events back into the shapes `ResponseStreamer._consume` reads, so the
preview / footer / markers / voice / image / reply-edit machinery is reused unchanged. It
calls Gemini DIRECT (the cog's `interactions_client` uses `gemini_api_key`, no proxy): the
Interactions API is inherently Gemini and the swap only fires on a Gemini answer model, so
this is the one runtime answer turn that does not ride the LiteLLM proxy. Importing the
google-genai Interactions types here is the documented carve-out for video ingestion.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, cast
from collections.abc import AsyncIterator

from google import genai
from openai.types.responses import ResponseStreamEvent
from google.genai._interactions.types import (
    StepParam,
    ContentParam,
    ThinkingLevel,
    EnvironmentParam,
    TextContentParam,
    AudioContentParam,
    ImageContentParam,
    VideoContentParam,
    UserInputStepParam,
    InteractionSSEEvent,
    DocumentContentParam,
    ModelOutputStepParam,
    GenerationConfigParam,
)
from openai.types.shared.reasoning_effort import ReasoningEffort
from google.genai._interactions.types.tool_param import URLContext, GoogleSearch
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam
from google.genai._interactions.types.environment_param import (
    NetworkAllowlist,
    NetworkAllowlistAllowlist,
)
from openai.types.responses.response_input_content_param import ResponseInputContentParam

if TYPE_CHECKING:
    from openai.types.responses.response_input_file_param import ResponseInputFileParam
    from openai.types.responses.response_input_text_param import ResponseInputTextParam
    from openai.types.responses.response_input_image_param import ResponseInputImageParam


def _kind_from_filename(filename: str) -> Literal["image", "video", "audio", "document"]:
    """Infers the Interactions content kind from a file's extension.

    Gemini-path attachments arrive as `input_file` parts carrying a Files API URI (no MIME),
    so the content-param type is picked from the original filename; an unknown extension falls
    back to document (best effort, never raises).
    """
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix in {"mp4", "mov", "webm", "avi", "mpeg", "mpg", "flv", "wmv", "3gp", "3gpp", "mkv"}:
        return "video"
    if suffix in {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "heic", "heif"}:
        return "image"
    if suffix in {"mp3", "wav", "ogg", "m4a", "aac", "flac", "opus", "aiff", "weba"}:
        return "audio"
    return "document"


def _translate_part(  # noqa: PLR0911 -- one return per OpenAI input content type mapped
    *, part: ResponseInputContentParam
) -> ContentParam | None:
    """Translates one OpenAI input content part into an Interactions content param.

    Media is referenced by its existing URI (Files API `file_id` or a raw `file_url`); the
    Gemini answer model already holds those URIs, so nothing is re-uploaded. Returns None for
    an empty or unmappable part so the caller drops it instead of breaking the request.
    """
    part_type = part["type"]
    if part_type == "input_text":
        text = cast("ResponseInputTextParam", part)["text"]
        return TextContentParam(type="text", text=text) if text else None
    if part_type == "input_image":
        image_part = cast("ResponseInputImageParam", part)
        uri = image_part.get("image_url") or image_part.get("file_id")
        return ImageContentParam(type="image", uri=uri) if uri else None
    if part_type == "input_file":
        file_part = cast("ResponseInputFileParam", part)
        uri = file_part.get("file_url") or file_part.get("file_id")
        if not uri:
            return None
        kind = _kind_from_filename(filename=file_part.get("filename") or "")
        if kind == "video":
            return VideoContentParam(type="video", uri=uri)
        if kind == "audio":
            return AudioContentParam(type="audio", uri=uri)
        if kind == "image":
            return ImageContentParam(type="image", uri=uri)
        return DocumentContentParam(type="document", uri=uri)
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
    `response.response.model` / `response.response.usage.{input,output}_tokens`. The
    Interactions stream uses different names (`event_type`, `delta.text`,
    `interaction.model`, `metadata.usage.total_*_tokens`), so each event is remapped onto a
    minimal namespace with the OpenAI-Responses field names. Usage is emitted exactly once on
    `interaction.completed` because `_consume` accumulates it with `+=` over a token seed from
    the earlier selection call; a per-step emit would double-count.
    """
    model_name = ""
    # Branch on `event.event_type` directly (not a copied local) so the discriminated
    # InteractionSSEEvent union narrows to the member that carries the field being read.
    async for event in stream:
        if event.event_type == "interaction.created":
            model_name = event.interaction.model or ""
            yield cast(
                "ResponseStreamEvent",
                SimpleNamespace(
                    type="response.created", response=SimpleNamespace(model=model_name, usage=None)
                ),
            )
        elif event.event_type == "step.delta":
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
            # Usage rides the completed interaction; `metadata.usage` is optional and often
            # absent, so read the interaction first and only fall back to metadata.
            usage = event.interaction.usage
            if usage is None and event.metadata is not None:
                usage = event.metadata.usage
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
                        model=(event.interaction.model or model_name), usage=usage_ns
                    ),
                ),
            )
        elif event.event_type == "error":
            raise RuntimeError(f"Gemini interactions stream error: {event!r}")


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
            type="remote",
            network=NetworkAllowlist(allowlist=[NetworkAllowlistAllowlist(domain="*")]),
        ),
        generation_config=GenerationConfigParam(
            # effort is the route grade copied onto slow_model (always low / medium / high here,
            # all valid for gemini-3.1-pro), so narrowing ReasoningEffort to the enum is safe.
            thinking_level=cast("ThinkingLevel", effort),
            thinking_summaries="auto",
        ),
        tools=[
            URLContext(type="url_context"),
            GoogleSearch(type="google_search", search_types=["web_search"]),
        ],
        stream=True,
    )
    # `stream=True` returns an async event stream; the overload still types it as a union with
    # the non-streaming Interaction, so narrow to the iterator the adapter consumes.
    stream = cast("AsyncIterator[InteractionSSEEvent]", responses)
    async for event in adapt_interactions_stream(stream=stream):
        yield event
