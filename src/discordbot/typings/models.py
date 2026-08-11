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
        examples=["gemini-flash-latest", "gemini-3.1-flash-image"],
    )
    # `minimal`, never `none`. Gemini 3 cannot switch thinking off at all, so the API's own
    # vocabulary starts at `minimal` (`thinking_level` accepts minimal / low / medium / high).
    # LiteLLM does map `none`, but only after recognising the model as Gemini 3 by the literal
    # substring `gemini-3` in the model string, which the `*-latest` aliases this project
    # dispatches on do not contain. It therefore falls through to the pre-3 branch and sends
    # `thinkingBudget: 0`, which a Gemini 3.x model rejects — so `none` silently stopped working
    # the moment `gemini-flash-latest` began resolving to a 3.x snapshot. `minimal` maps to a
    # small positive budget on that same branch and is accepted.
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
            receive web_search and web_fetch tools. Other models receive the OpenAI
            web_search tool.
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

        Callers: `VideoGenerator.render` (via the VIDEO route `_handle_video_reply`).

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
    def prompt_model(self) -> ModelSettings:
        """The model settings for the image/video generation prompt director.

        Callers: `PromptGenerator.refine` (via `_handle_image_reply`, `_handle_video_reply`).

        Returns:
            Flash-with-high-effort settings for the director call that expands a thin user
            request into a rich, self-contained generation prompt before the image/video model
            draws it. Flash (not flash-lite) so it reliably CALLS the grounding tools
            (googleSearch / urlContext) to look up named subjects; effort is the latency lever
            since this call sits serially on the IMAGE/VIDEO critical path before generation.
            Deliberately decoupled from `slow_model` so the refinement tier can be tuned on its
            own (bump to a pro snapshot here if the refined prompts underperform).
        """
        return ModelSettings(name="gemini-flash-latest", effort="high")

    @property
    def media_reply_model(self) -> ModelSettings:
        """The model settings for the conversational reply that rides generated media.

        Callers: `_stream_media_persona_reply` (via `_handle_image_reply` and
        `_handle_video_reply`).

        One tier for both routes because they are the same task on the same shared streamer:
        only the system prompt and the focus part differ per route. Flash, the middle tier
        between the flash-lite caption it replaces and the gemini-pro answer model, and it
        ingests video as well as images: it reads conversation history and the selected user
        memory, then answers in persona while holding the image or clip it just made rather
        than coldly describing it.

        Returns:
            Flash with `effort="low"`, which keeps it snappy yet still emits a reasoning
            summary so the streaming reasoning preview shows. The media is already on screen,
            so this text streams in after with no added generation latency.
        """
        return ModelSettings(name="gemini-flash-latest", effort="low")

    @property
    def tts_model(self) -> ModelSettings:
        """The model settings for spoken-reply text-to-speech.

        Callers: `VoiceGenerator` (via `ReplyGeneratorCogs.voice_generator`).

        Returns:
            Model settings whose name is dispatched on the `audio.speech` endpoint to
            render a fierce QA reply to a voice clip. `effort` is unused for TTS.
        """
        return ModelSettings(name="gemini-3.1-flash-tts-preview")

    @property
    def fast_model(self) -> ModelSettings:
        """The model settings for lightweight, single-shot tasks.

        Callers: `_route_classify`, `_grade_effort`, `AutoUnmuteCogs._generate_reply`,
        `StockNewsAI`, the research thread title.

        This is the difficulty tier, not a purpose: everything routed here is one short call
        whose job is either a narrow classification or a throwaway line of text. The route
        picks the reply mode and the effort grade decides how hard the answer model should
        think; both follow simple rules, so flash-lite is enough and the QA critical path
        stays short (the grade runs in parallel with the route under the same `route_done`
        gate, so its latency hides behind the route entirely).

        Returns:
            Fast minimal-thinking settings.
        """
        return ModelSettings(name="gemini-flash-lite-latest", effort="minimal")

    @property
    def tool_model(self) -> ModelSettings:
        """The model settings for optional oblique-reference memory selection.

        Callers: `_select_user_memories`.

        Returns:
            Fast minimal-thinking settings for matching an obliquely referenced absent
            member to the public nickname table: flash (not flash-lite)
            because matching spoken community nicknames to user ids needs more
            language skill than the lite tier reliably delivers, while staying far
            below answer-model latency.
        """
        return ModelSettings(name="gemini-flash-latest", effort="minimal")

    @property
    def slow_model(self) -> ModelSettings:
        """The model settings for full text replies and strategic reasoning.

        Callers: `_handle_message_reply` (which overrides `effort` with the
        route-decided level), attachment modality gating, `write_up_report` (the
        background rewrite of a `/feedback` report into an issue, which nobody waits on),
        and dev scripts.

        Returns:
            Slow-path model settings for reply generation and summaries.
        """
        # Both branches are pinned to explicit snapshots, never a `*-latest` alias. This is the
        # one tier whose effort is replaced at runtime by the route's grade, and the YouTube
        # answer turn hands that effort straight to the Interactions API as a `thinking_level`
        # (`gen_reply/interactions.py`), where the enum is per-model: every alias measured
        # (flash / pro / flash-lite) accepts only low / high, while every pinned snapshot takes
        # `medium` as well. The grade is binary since #490, so it no longer reaches the value
        # that lost whole replies through an alias in #459; the pinning stays because it is what
        # keeps the next vocabulary change from doing it again, and because `minimal` is still a
        # 400 on the pro snapshot and stays legal only while `EffortGrade` never emits it.
        # The peak/off-peak split is load-bearing again rather than dormant: Gemini Pro has
        # historically slowed down during peak hours, so peak takes the flash snapshot.
        if self.is_peak:
            return ModelSettings(name="gemini-3.6-flash", effort="high")
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
        return ModelSettings(name="gemini-pro-latest", effort="high")

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
        return ModelSettings(name="gemini-pro-latest", effort="high")


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
            "that asks for nothing, looks nothing up and works nothing out, and that is "
            "fully answerable from what you were shown; everything else, including "
            "anything you are unsure about, is high."
        ),
    )


__all__ = ["EffortGrade", "ModelSettings", "RouteClassification", "RuntimeModelCatalog"]
