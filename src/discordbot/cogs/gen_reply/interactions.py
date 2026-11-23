"""Gemini Interactions API answer path for QA turns watching a YouTube video.

Translates OpenAI Responses input into Interactions steps with the YouTube `VideoContentParam`
appended, calls Gemini directly via `genai.Client`, and adapts Interactions SSE events into the
`ResponseStreamEvent` shapes `ResponseStreamer._consume` reads.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, cast
from collections.abc import AsyncIterator

from google import genai
import logfire
from google.genai.errors import APIError
from openai.types.responses import ResponseStreamEvent
from google.genai.interactions import (
    Step,
    StepParam,
    URLContext,
    ContentParam,
    GoogleSearch,
    StepDeltaData,
    ThinkingLevel,
    EnvironmentParam,
    TextContentParam,
    VideoContentParam,
    UserInputStepParam,
    AllowlistEntryParam,
    InteractionSSEEvent,
    ModelOutputStepParam,
    GenerationConfigParam,
    EnvironmentNetworkEgressAllowlistParam,
)
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam
from openai.types.responses.response_input_content_param import ResponseInputContentParam

from discordbot.cogs.gen_reply.tool_use import AnswerToolUse

if TYPE_CHECKING:
    from openai.types.responses.response_input_file_param import ResponseInputFileParam
    from openai.types.responses.response_input_text_param import ResponseInputTextParam
    from openai.types.responses.response_input_image_param import ResponseInputImageParam


def _kind_from_filename(filename: str) -> Literal["image", "video", "audio", "document"]:
    """Infers the Interactions content kind from `filename`'s extension, defaulting to `"document"`.

    Must stay aligned with `InlineRenderer`'s file MIME types.
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
    """Builds a media `ContentParam` from a Files URI or `data:` base64 string, or None."""
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
    # Cast bypasses Literal mime_type checking; invalid attachment MIME types fail server-side.
    return cast("ContentParam", {"type": kind, "data": payload, "mime_type": mime_type})


def _translate_part(part: ResponseInputContentParam) -> ContentParam | None:
    """Translates one OpenAI input part into an Interactions `ContentParam`, or None."""
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
        # `file_data` carries inlined files (e.g. PDFs when the Files API is off).
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
    # Agentic processing loads only needed segments/transcript instead of sampling 1 FPS.
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


def _note_tool_use(
    part: Step | StepDeltaData, queries: set[str], urls: set[str], citations: list[tuple[str, str]]
) -> None:
    """Adds what `part` carries if it is a search or URL context call or a text annotation."""
    if part.type == "google_search_call":
        queries.update(part.arguments.queries or [])
    elif part.type == "url_context_call":
        urls.update(part.arguments.urls or [])
    elif part.type == "text_annotation_delta":
        citations.extend(
            (annotation.url or "", annotation.title or "")
            for annotation in part.annotations or []
            if annotation.type == "url_citation"
        )


async def _record_tool_use(
    stream: "AsyncIterator[InteractionSSEEvent]", tool_use: AnswerToolUse
) -> "AsyncIterator[InteractionSSEEvent]":
    """Yields `stream` unchanged, writing tool calls and citations to `tool_use` on completion."""
    # Tool arguments may arrive on step.start, step.delta, or both, so deduplicate with sets.
    queries: set[str] = set()
    urls: set[str] = set()
    citations: list[tuple[str, str]] = []
    async for event in stream:
        if event.event_type == "step.start":
            _note_tool_use(part=event.step, queries=queries, urls=urls, citations=citations)
        elif event.event_type == "step.delta":
            _note_tool_use(part=event.delta, queries=queries, urls=urls, citations=citations)
        elif (
            event.event_type == "interaction.completed" and event.interaction.status == "completed"
        ):
            tool_use.web_search_queries = len(queries)
            tool_use.url_fetches = len(urls)
            tool_use.record_citations(citations=citations)
        yield event


async def adapt_interactions_stream(
    stream: "AsyncIterator[InteractionSSEEvent]",
) -> AsyncIterator[ResponseStreamEvent]:
    """Yields `ResponseStreamEvent`-shaped namespaces adapted from `stream` for `ResponseStreamer._consume`.

    Remaps text and reasoning deltas, emits token usage once on `interaction.completed`, maps
    non-`completed` terminal statuses to `response.incomplete`, and converts `"error"` events into
    `APIError`.

    Args:
        stream (AsyncIterator[InteractionSSEEvent]): Raw Gemini Interactions SSE stream.

    Raises:
        APIError: The Interactions stream emitted an `"error"` event.
    """
    model_name = ""
    # Fallback when interaction.completed omits usage and only step.delta reported total_usage.
    streamed_usage = None
    # Branch on event.event_type directly so ty narrows the discriminated union.
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
            elif delta.type == "thought_summary" and (
                text := getattr(delta.content, "text", "") if delta.content is not None else ""
            ):
                yield cast(
                    "ResponseStreamEvent",
                    SimpleNamespace(type="response.reasoning_summary_text.delta", delta=text),
                )
        elif event.event_type == "interaction.completed":
            usage = event.interaction.usage or streamed_usage
            usage_ns = (
                SimpleNamespace(
                    input_tokens=usage.total_input_tokens or 0,
                    output_tokens=usage.total_output_tokens or 0,
                )
                if usage is not None
                else None
            )
            # Interactions carries no finish_reason; non-completed status stands in for it.
            completed = event.interaction.status == "completed"
            yield cast(
                "ResponseStreamEvent",
                SimpleNamespace(
                    type="response.completed" if completed else "response.incomplete",
                    response=SimpleNamespace(
                        model=(event.interaction.model or model_name),
                        usage=usage_ns,
                        # None (not []) so ResponseStreamer keeps tool_use from _record_tool_use.
                        output=None,
                        incomplete_details=None
                        if completed
                        else SimpleNamespace(reason=event.interaction.status),
                    ),
                ),
            )
        elif event.event_type == "error":
            # Raise APIError for extract_friendly_error; SDK Error.code is a URI (not HTTP status),
            # so in-band errors carry code=0 (unretried) unless decimal.
            failure = event.error
            code = failure.code if failure is not None else None
            raise APIError(
                code=int(code) if code is not None and code.isdecimal() else 0,
                response_json={
                    "error": {
                        "message": (failure.message if failure is not None else None)
                        or f"Gemini interactions stream error: {event!r}",
                        "code": code,
                    }
                },
            )


async def create_interactions_answer_stream(  # noqa: PLR0913 -- the request inputs plus the streamer's tool-use record
    client: genai.Client,
    model: str,
    system_instruction: str,
    steps: list[StepParam],
    effort: ReasoningEffort,
    tool_use: AnswerToolUse,
) -> AsyncIterator[ResponseStreamEvent]:
    """Yields adapted `ResponseStreamEvent` items for a YouTube-aware QA answer on the Gemini Interactions API.

    Calls Gemini directly with `url_context` and `google_search` grounding tools and `effort`
    mapped to `thinking_level`.

    Args:
        client (genai.Client): Direct-to-Google Gemini client.
        model (str): Gemini model name to dispatch.
        system_instruction (str): Runtime developer instructions for the turn.
        steps (list[StepParam]): Translated conversation and YouTube input steps.
        effort (ReasoningEffort): Reasoning effort grade (`"low"`, `"medium"`, `"high"` or `"xhigh"`).
        tool_use (AnswerToolUse): The streamer's record of the answer's tool use.
    """
    responses = await client.aio.interactions.create(
        model=model,
        system_instruction=system_instruction,
        input=steps,
        # Ad-hoc remote sandbox (a bare "remote" string is treated as an environment ID); "*"
        # allows server-side tools and YouTube fetches to reach any domain.
        environment=EnvironmentParam(
            type="remote",
            network=EnvironmentNetworkEgressAllowlistParam(
                allowlist=[AllowlistEntryParam(domain="*")]
            ),
        ),
        generation_config=GenerationConfigParam(
            # Direct to Google refuses unsupported thinking_level values; effort must stay within
            # slow_model's supported set.
            thinking_level=cast("ThinkingLevel", effort),
            thinking_summaries="auto",
        ),
        tools=[
            URLContext(type="url_context"),
            GoogleSearch(search_types=["web_search"], type="google_search"),
        ],
        stream=True,
    )
    # Overload still returns Interaction | AsyncStream even with stream=True.
    stream = cast("AsyncIterator[InteractionSSEEvent]", responses)
    async for event in adapt_interactions_stream(
        stream=_record_tool_use(stream=stream, tool_use=tool_use)
    ):
        yield event
