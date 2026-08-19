from typing import Literal, cast
from datetime import UTC, datetime

from pydantic import Field, BaseModel, computed_field
from openai.types.responses.tool_param import ToolParam
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.reasoning import Reasoning


class ModelSettings(BaseModel):
    """Model name and reasoning effort that should be used together.

    Attributes:
        name: LiteLLM model string dispatched on the Responses API.
        effort: Reasoning effort passed to the Responses API for this model.
    """

    name: str = Field(
        ...,
        description="LiteLLM model string dispatched on the Responses API.",
        examples=["gemini-3.7-flash", "gemini-3.1-flash-image"],
    )
    # `minimal`, never `none`. Gemini 3 cannot switch thinking off at all, so the API's own
    # vocabulary starts at `minimal` (`thinking_level` accepts minimal / low / medium / high).
    # Which of LiteLLM's two branches translates the effort is decided by the literal substring
    # `gemini-3` in the model string, and every name this project sends through the proxy
    # carries it, so the `thinking_level` branch is the one that runs. There `none` / `disable`
    # are not rejected but rewritten to minimal / low with `includeThoughts: False`, which
    # silently ends the reasoning summary the streaming preview is built on; asking for
    # `minimal` keeps it. On a `*-latest` alias, which carries no `gemini-3`, the same request
    # instead falls through to the pre-3 branch and sends `thinkingBudget: 0`, which a Gemini
    # 3.x model rejects outright.
    effort: ReasoningEffort = Field(
        default="minimal",
        description="Reasoning effort passed to the Responses API for this model.",
    )

    @property
    def reasoning(self) -> Reasoning:
        """Responses API reasoning options for this model.

        Returns:
            Reasoning options using this model's configured effort and an
            automatic reasoning summary.
        """
        return Reasoning(effort=self.effort, summary="auto")

    @property
    def tools(self) -> list[ToolParam]:
        """Built-in tool payloads for this model's provider.

        Code execution is intentionally omitted: Gemini and Claude validate every
        uploaded file part against code execution's narrow MIME allowlist and 400 the
        whole request on video / audio / GIF-as-video attachments, so it cannot coexist
        with the attachment ingestion path. Search / url grounding have no such limit.

        Returns:
            Gemini models receive googleSearch and urlContext tools. Claude models
            receive web_search and web_fetch tools. Grok models receive web_search and
            x_search. Every other model receives the OpenAI web_search tool.
        """
        if "gemini" in self.name:
            return cast("list[ToolParam]", [{"googleSearch": {}}, {"urlContext": {}}])
        if "claude" in self.name:
            return cast(
                "list[ToolParam]",
                [
                    {"type": "web_search_20260209", "name": "web_search"},
                    {"type": "web_fetch_20260209", "name": "web_fetch"},
                ],
            )
        if "grok" in self.name:
            return cast("list[ToolParam]", [{"type": "web_search"}, {"type": "x_search"}])
        return cast("list[ToolParam]", [{"type": "web_search"}])


class RuntimeModelCatalog(BaseModel):
    """Runtime model settings used by Discord bot LLM paths.

    Keep caller lists in sync when moving runtime model usage.
    """

    @computed_field
    @property
    def is_peak(self) -> bool:
        """Whether runtime model selection is in the peak-hour window.

        Returns:
            True during UTC weekdays from 08:00 up to (but excluding) 17:00, otherwise False.
        """
        now = datetime.now(UTC)
        return now.weekday() < 5 and 8 <= now.hour < 17

    @property
    def image_model(self) -> ModelSettings:
        """The model settings for image generation and editing.

        Callers: `ImageGenerator` (its `render` for the IMAGE route `_handle_image_reply`, its
        best-effort `generate` for the QA-route inline `<generate-image>` marker).

        Returns:
            Model settings used with `images.generate` and `images.edit`.
        """
        return ModelSettings(name="gemini-3.1-flash-image")

    @property
    def video_model(self) -> ModelSettings:
        """The model settings for video generation.

        Callers: `VideoGenerator` (its raising `render` for the VIDEO route
        `_handle_video_reply`, its best-effort `generate` for the QA-route inline
        `<generate-video>` marker, which delegates to that same `render`).

        Returns:
            Model settings used with the native Gemini Interactions API (`interactions.create`,
            a bare model name with no provider prefix, since the call goes direct to Google not
            via the proxy). omni unifies text/image/reference/edit video generation, so the same
            model backs plain generation and true source-video editing (`task="edit"`).
        """
        return ModelSettings(name="gemini-omni-flash-preview")

    @property
    def music_model(self) -> ModelSettings:
        """The model settings for music generation.

        Callers: `MusicGenerator.generate` (via the QA-route inline `<generate-music>` marker).

        Returns:
            Model settings used with the native Gemini (Lyria) Interactions API (a bare model
            name, no provider prefix, since the call goes direct to Google not via the proxy).
        """
        return ModelSettings(name="lyria-3-clip-preview")

    @property
    def tts_model(self) -> ModelSettings:
        """The model settings for spoken-reply text-to-speech.

        Callers: `VoiceGenerator` (via `ReplyGeneratorCogs.voice_generator`).

        Returns:
            Model settings whose name is dispatched on the `audio.speech` endpoint. Only
            the reply's `<generate-voice>` segments are synthesized, concatenated into one
            clip; the rest of the reply is never spoken. `effort` is unused for TTS.
        """
        return ModelSettings(name="gemini-3.1-flash-tts-preview")

    @property
    def antigravity_model(self) -> ModelSettings:
        """The deep-research agent: a one-shot Antigravity managed agent.

        Callers: `ResearchCogs` (the only research tier there is).

        Returns:
            The Antigravity managed-agent string dispatched on the Gemini Interactions API
            (direct, not the proxy). `effort` / `tools` are unused on the agent path: the
            agent runs its own internal tool loop.
        """
        return ModelSettings(name="antigravity-preview-05-2026")

    @property
    def triage_model(self) -> ModelSettings:
        """The model settings for the judgments the reply itself waits on.

        Callers: `_route_classify`, `_grade_effort`, `_select_user_memories`.

        What these three share is not their size but where their output goes: each parses
        into a structured field nobody ever reads, and each sits on the critical path
        holding the reply back. Nothing here writes a word a user sees, so the tier is
        bought on latency alone. That is the whole seam against `fast_model` beside it,
        whose output is visible and therefore worth waiting a little longer for.

        Returns:
            Flash-lite at `minimal`, the floor Gemini 3 allows. LiteLLM matches this name as
            a Gemini 3 flash and forwards `thinking_level: minimal` untouched, which is
            exactly what `gemini-3.7-flash` rejects; measured 2026-08-20 end to end, this
            snapshot accepts it and answers under its own name rather than a proxy fallback's.
        """
        return ModelSettings(name="gemini-3.5-flash-lite", effort="minimal")

    @property
    def fast_model(self) -> ModelSettings:
        """The model settings for the text a user reads that is not the answer.

        Callers: `PromptGenerator.refine` (the IMAGE/VIDEO prompt director),
        `_stream_media_persona_reply` (the persona reply that rides generated media),
        `AutoUnmuteCogs._generate_reply`, `StockNewsAI`, the research thread title.

        Every one of them writes prose somebody reads, and none of them is the deliverable:
        a weak line here is visible but costs nothing that was being waited on. That is the
        axis this tier shares with `slow_model`, one difficulty step below it. The structured
        judgments that used to sit here are `triage_model`'s, and they were never on that
        axis at all.

        Returns:
            Flash at `medium`, one snapshot back from `gemini-3.7-flash`, which is popular
            enough to queue behind its own load (observed 2026-08-20). `medium` is what buys
            a prompt or a persona line that reads like it was written rather than filled in;
            `minimal` would be the wrong end of the ladder here even where it is accepted.
        """
        return ModelSettings(name="gemini-3.6-flash", effort="medium")

    @property
    def slow_model(self) -> ModelSettings:
        """The model settings for full text replies and strategic reasoning.

        Dispatched by `_handle_message_reply` (which overrides `effort` with the
        route-decided level) and by `write_up_report` (the background rewrite of a
        `/feedback` report into an issue, which nobody waits on). Three more read only
        the model NAME and dispatch nothing: `_supported_sources` gates attachment
        modalities on it, `ReplyGeneratorCogs.input_builder` picks the attachment handler
        off it through `build_attachment_handler`, and `_run_reply_pipeline` derives each
        link-context builder's `answer_model_is_gemini` from it.

        Returns:
            Slow-path model settings for reply generation and summaries.
        """
        # Both branches, the commented-out one included, are pinned to explicit snapshots and
        # never a `*-latest` alias. This is the
        # one tier whose effort is replaced at runtime by the route's grade, and the YouTube
        # answer turn hands that effort straight to the Interactions API as a `thinking_level`
        # (`gen_reply/interactions.py`), where the enum is per-model: every alias measured
        # (flash / pro / flash-lite) accepts only low / high, while every pinned snapshot takes
        # `medium` as well. The grade is binary since #490, so it no longer reaches the value
        # that lost whole replies through an alias in #459; the pinning stays because it is what
        # keeps the next vocabulary change from doing it again. Note `minimal` is still a 400 on
        # the pro snapshot; it stays legal only because `EffortGrade` never emits it.
        # The peak/off-peak split is commented out rather than deleted, and `is_peak` stays
        # exposed for it. It sent peak hours to the flash snapshot because Gemini Pro used to
        # slow down under load; `gemini-3.7-flash` has since become the popular tier and is
        # now the one that queues (observed 2026-08-20), so the branch was handing the busiest
        # hours the slower of the two. Restore it when that inverts back.
        # if self.is_peak:
        #     return ModelSettings(name="gemini-3.7-flash", effort="high")
        return ModelSettings(name="gemini-3.1-pro-preview", effort="high")

    @property
    def memory_extractor_model(self) -> ModelSettings:
        """The model settings for phase-1 memory extraction.

        Callers: `MemoryExtractorAI.extract` (its `extract_model` field).

        Returns:
            Model settings for the background memory extraction call. Kept apart from the
            writer tier so the recall-oriented first pass can be downgraded on its own if
            the gates behind it prove able to carry the precision.
        """
        return ModelSettings(name="gemini-3.1-pro-preview", effort="high")

    @property
    def memory_writer_model(self) -> ModelSettings:
        """The model settings for everything deciding what reaches long-term memory.

        Callers: the phase-1.5 evaluator (`MemoryExtractorAI.extract`, its optional
        `evaluate_model` field) and phase-2 consolidation (`MemoryExtractorAI.consolidate`,
        its `consolidate_model` field), which also backs `regenerate_main_memory`, plus
        `scripts/regen_memories.py`, which defaults to this tier to drive that rebuild offline.

        Returns:
            Model settings for the background memory write calls. One tier for both because
            both are gates on the same side: the evaluator is the last LLM check before an
            observation is staged, tightening `sharing` and `durability` (downgrade-only),
            deduping by `normalized_key` and stripping personal-attack wording, and the
            consolidator turns that staging into the fact files plus the tone note. A weaker
            model on either loses memory or leaks it, rather than just failing to record it.
        """
        return ModelSettings(name="gemini-3.1-pro-preview", effort="high")


class RouteClassification(BaseModel):
    """Structured reply-mode classification returned by the route model.

    Attributes:
        decision: The reply mode selected for the incoming Discord message.
        watch_video: Whether the QA answer should ingest a linked YouTube video.
        link_context_sources: Linked-post sources whose content the QA answer should ingest.
    """

    decision: Literal["IMAGE", "VIDEO", "QA", "SUMMARY"] = Field(
        ..., description="Reply mode selected for the incoming Discord message."
    )
    watch_video: bool = Field(
        default=False,
        description=(
            "Set true only when the message links a YouTube video AND the user wants its "
            "content analyzed, summarized, or asked about; false when the link is incidental. "
            "Consumed only on the QA route to decide whether to watch the video."
        ),
    )
    link_context_sources: list[Literal["threads", "douyin", "bilibili"]] = Field(
        default_factory=list,
        description=(
            "Registered linked-post sources whose actual content the user wants analyzed, "
            "summarized, or discussed. Empty when every matching link is incidental. Consumed "
            "only on the QA route to decide which source builders may start."
        ),
    )


class EffortGrade(BaseModel):
    """Structured answer-effort grade returned by the effort model.

    Graded by a call that runs in parallel with the route; the answer model's effort is
    overridden with it on the QA and SUMMARY paths.

    Deliberately binary, with `high` as the grade an ordinary message gets and `low` as the
    exception that has to be earned (#490): the grader reads text-only parts, so it never sees
    an attachment's content, a linked post, or the history behind a short message, and every
    one of those blind spots hides work rather than inventing it. `{low, high}` is also the one
    set every model and surface measured in #461 accepts as a `thinking_level`.

    Attributes:
        effort: Reasoning effort the answer model should spend on this message.
    """

    effort: Literal["low", "high"] = Field(
        default="high",
        description=(
            "Reasoning effort the answer model should spend. Use low only for a message "
            "that asks for nothing, looks nothing up and works nothing out — banter and "
            "greetings, even when phrased as a question — and that is fully answerable "
            "from what you were shown; everything else, including anything you are "
            "unsure about, is high."
        ),
    )


__all__ = ["EffortGrade", "ModelSettings", "RouteClassification", "RuntimeModelCatalog"]
