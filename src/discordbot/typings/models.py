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
        examples=["gemini-flash-latest", "gemini-3-pro-image-preview"],
    )
    effort: ReasoningEffort = Field(
        default="none", description="Reasoning effort passed to the Responses API for this model."
    )

    @property
    def reasoning(self) -> Reasoning:
        """Responses API reasoning options for this model.

        Returns:
            Reasoning options using this model's configured effort and an
            automatic reasoning summary.

        Raises:
            ValueError: The model has no reasoning effort configured.
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

        Callers: `_handle_image_reply`.

        Returns:
            Model settings used with `images.generate` and `images.edit`.
        """
        image_model = ModelSettings(name="gemini-3-pro-image-preview")
        return image_model

    @property
    def video_model(self) -> ModelSettings:
        """The model settings for video generation.

        Callers: `_handle_video_reply`.

        Returns:
            Model settings used with `videos.create`.
        """
        video_model = ModelSettings(name="veo-3.1-fast-generate-preview")
        return video_model

    @property
    def tts_model(self) -> ModelSettings:
        """The model settings for spoken-reply text-to-speech.

        Callers: `VoiceSynthesizer` (via `ReplyGeneratorCogs.voice_synthesizer`).

        Returns:
            Model settings whose name is dispatched on the `audio.speech` endpoint to
            render a fierce QA reply to a voice clip. `effort` is unused for TTS.
        """
        return ModelSettings(name="gemini-3.1-flash-tts-preview")

    @property
    def fast_model(self) -> ModelSettings:
        """The model settings for lightweight reply-generation tasks.

        Callers: `_handle_image_reply`, `_generate_reply`, `SystemNarrator`, `AutoUnmuteCogs._generate_reply`, `StockNewsAI`.

        Returns:
            Fast model settings used for image captions, short Discord
            replies, casino system narrator lines, auto-unmute replies,
            and stock news generation.
        """
        fast_model = ModelSettings(name="gemini-flash-lite-latest", effort="none")
        return fast_model

    @property
    def route_model(self) -> ModelSettings:
        """The model settings for the route classification decision.

        Callers: `_route_message`.

        Returns:
            Fast no-reasoning settings for the route + effort grading call. The route's
            main job is classification and the effort grade follows simple complexity
            rules, so flash-lite is enough here and keeps the QA critical path short.
        """
        return ModelSettings(name="gemini-flash-lite-latest", effort="none")

    @property
    def tool_model(self) -> ModelSettings:
        """The model settings for the phase-1 get_user_memory selection decision.

        Callers: `_select_user_memories`.

        Returns:
            Fast no-reasoning settings for the "whose long-term memory to read"
            tool-call decision on the reply critical path: flash (not flash-lite)
            because matching spoken community nicknames to user ids needs more
            language skill than the lite tier reliably delivers, while staying far
            below answer-model latency.
        """
        return ModelSettings(name="gemini-flash-latest", effort="none")

    @property
    def slow_model(self) -> ModelSettings:
        """The model settings for full text replies and strategic reasoning.

        Callers: `_handle_message_reply` (which overrides `effort` with the
        route-decided level), attachment modality gating, and dev scripts.

        Returns:
            Slow-path model settings for reply generation and summaries.
        """
        # Both branches dispatch the same model today; the peak/off-peak split is
        # kept on purpose because Gemini Pro has historically slowed down during
        # peak hours and the split may be needed again.
        if self.is_peak:
            return ModelSettings(name="gemini-pro-latest", effort="high")
        return ModelSettings(name="gemini-pro-latest", effort="high")

    @property
    def extract_model(self) -> ModelSettings:
        """The model settings for phase-1 per-user memory extraction.

        Callers: `MemoryExtractorAI.extract`.

        Returns:
            Model settings for the background memory extraction call.
        """
        return ModelSettings(name="gemini-pro-latest", effort="high")

    @property
    def memory_evaluator_model(self) -> ModelSettings:
        """The model settings for strict phase-1 memory quality evaluation.

        Callers: `MemoryExtractorAI.extract`.

        Returns:
            Model settings for the background memory evaluator call.
        """
        return ModelSettings(name="gemini-pro-latest", effort="high")

    @property
    def memories_model(self) -> ModelSettings:
        """The model settings for phase-2 memory consolidation.

        Callers: `MemoryExtractorAI.consolidate`.

        Returns:
            Model settings for the background memory consolidation call.
        """
        return ModelSettings(name="gemini-pro-latest", effort="high")

    @property
    def tone_model(self) -> ModelSettings:
        """The model settings for the per-user tone-note updater.

        Callers: `MemoryExtractorAI.update_tone` (via `schedule_tone_update`).

        Returns:
            Fast no-reasoning settings for the single-call background tone refresh.
            Distilling a short tone preference from one conversation is a simple
            summarization, so flash-lite is enough and keeps this frequent
            background call cheap, well below the Pro tier the memory pipeline uses.
        """
        return ModelSettings(name="gemini-flash-lite-latest", effort="none")

    @property
    def player_model(self) -> ModelSettings:
        """The model settings for the casino bot-player AI.

        Pinned to `gemini-flash-latest` regardless of peak hours so bot turns
        between human players stay snappy even if `slow_model` later promotes
        to a heavier Pro tier.

        Callers: `BotPlayerAI`.

        Returns:
            Model settings used by the Blackjack bot player for bet sizing,
            hit/stand, double/split, surrender, and insurance decisions.
        """
        return ModelSettings(name="gemini-flash-latest", effort="minimal")


class RouteDecision(BaseModel):
    """Structured routing decision returned by the model.

    Attributes:
        decision: The reply mode selected for the incoming Discord message.
        effort: Reasoning effort the answer model should spend on this message.
    """

    decision: Literal["IMAGE", "VIDEO", "QA", "SUMMARY"] = Field(
        ..., description="Reply mode selected for the incoming Discord message."
    )
    effort: Literal["low", "medium", "high"] = Field(
        default="high",
        description=(
            "Reasoning effort the answer model should spend: high for any substantive "
            "question or task, medium for trivial lookups or transforms, low only for "
            "pure social chatter."
        ),
    )


__all__ = ["ModelSettings", "RouteDecision", "RuntimeModelCatalog"]
