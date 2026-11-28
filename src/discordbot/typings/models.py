"""The model tiers every runtime call dispatches on, and the shapes a triage call fills in."""

from typing import Literal, cast

from pydantic import Field, BaseModel, ConfigDict
from openai.types.responses.tool_param import ToolParam
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.reasoning import Reasoning

from discordbot.typings.config import ModelsConfig, load_bot_config
from discordbot.typings.emojis import LinkSourceName


class ModelSettings(BaseModel):
    """Model name and reasoning effort that should be used together."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        description="LiteLLM model string dispatched on the Responses API.",
        examples=["gemini-3.8-flash", "gemini-3.1-flash-image"],
    )
    # Default `"minimal"` is only for non-reasoning tiers (image/video/music/TTS/agent);
    # every conversational tier sets `effort` explicitly from the model's supported set.
    effort: ReasoningEffort = Field(
        default="minimal",
        description="Reasoning effort passed to the Responses API for this model.",
    )

    @property
    def reasoning(self) -> Reasoning:
        """Returns Responses API reasoning options with this model's effort and auto summary."""
        return Reasoning(effort=self.effort, summary="auto")

    @property
    def is_gemini(self) -> bool:
        """Returns whether `name` identifies a Gemini model (`"gemini"` in `name`)."""
        return "gemini" in self.name

    @property
    def is_claude(self) -> bool:
        """Returns whether `name` identifies a Claude model (`"claude"` in `name`)."""
        return "claude" in self.name

    @property
    def tools(self) -> list[ToolParam]:
        """Returns the built-in search and URL-grounding tool payloads for this model's provider.

        Omits code execution, which conflicts with non-code attachment MIME types on Gemini and
        Claude.
        """
        if self.is_gemini:
            return cast("list[ToolParam]", [{"googleSearch": {}}, {"urlContext": {}}])
        if self.is_claude:
            return cast(
                "list[ToolParam]",
                [
                    {"type": "web_search_20260209", "name": "web_search"},
                    {"type": "web_fetch_20260209", "name": "web_fetch"},
                ],
            )
        if "grok" in self.name:
            return cast("list[ToolParam]", [{"type": "web_search"}, {"type": "x_search"}])
        return [{"type": "web_search"}]


class SlowModelSettings(ModelSettings):
    """QA answer model settings whose `effort` is restricted to the route's three grades."""

    effort: Literal["low", "medium", "high"] = Field(
        default="low",
        description=(
            "Router-failure fallback reasoning effort for this candidate, or the active grade "
            "chosen by the router for one QA turn."
        ),
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "Relative sampling weight for this candidate; 0 disables it without removing its entry."
        ),
    )


class RuntimeModelCatalog(BaseModel):
    """Runtime model settings used by Discord bot LLM paths, backed by `[global.models]`."""

    @staticmethod
    def _models() -> ModelsConfig:
        """Returns the live `[global.models]` configuration from `config.toml`."""
        return load_bot_config().global_config.models

    @property
    def image_model(self) -> ModelSettings:
        """Returns model settings for `images.generate` and `images.edit`."""
        return ModelSettings(name=self._models().image)

    @property
    def video_model(self) -> ModelSettings:
        """Returns model settings for video generation and editing (`task="edit"`).

        Uses a bare Gemini model name for direct `interactions.create` calls.
        """
        return ModelSettings(name=self._models().video)

    @property
    def music_model(self) -> ModelSettings:
        """Returns model settings for music generation on the native Lyria Interactions API."""
        return ModelSettings(name=self._models().music)

    @property
    def tts_model(self) -> ModelSettings:
        """Returns model settings for spoken-reply synthesis (`audio.speech`; `effort` unused)."""
        return ModelSettings(name=self._models().tts)

    @property
    def antigravity_model(self) -> ModelSettings:
        """Returns model settings for the deep-research Antigravity agent.

        `effort` and `tools` are unused.
        """
        return ModelSettings(name=self._models().antigravity)

    @property
    def antigravity_reasoning_model(self) -> ModelSettings:
        """Returns the reasoning model settings passed as `agent_config.model` for deep research.

        Must name a model accepted by the Antigravity agent; `effort` and `tools` are unused.
        """
        return ModelSettings(name=self._models().antigravity_reasoning)

    @property
    def triage_model(self) -> ModelSettings:
        """Returns model settings for fixed-schema triage calls."""
        triage = self._models().triage
        return ModelSettings(name=triage.name, effort=triage.effort)

    @property
    def fast_model(self) -> ModelSettings:
        """Returns model settings for intermediate prose generation short of the final answer."""
        fast = self._models().fast
        return ModelSettings(name=fast.name, effort=fast.effort)

    @property
    def slow_candidates(self) -> tuple[SlowModelSettings, ...]:
        """Active configured QA candidate models with each model's fallback effort and weight.

        Excludes zero-weight entries; sampled by `weight` per QA turn. Every entry should support
        `RouteClassification.effort` (`low`, `medium`, `high`).
        """
        return tuple(
            SlowModelSettings(
                name=candidate.name, effort=candidate.effort, weight=candidate.weight
            )
            for candidate in self._models().slow.candidates
            if candidate.weight > 0
        )

    @property
    def slow_display_name(self) -> str:
        """Optional visible model label masking the real QA model and effort, or empty."""
        return self._models().slow.display_name

    @property
    def slow_model(self) -> SlowModelSettings:
        """Returns the primary slow model (first active configured QA candidate).

        Callers also inspect `name` to determine provider and supported input modalities.
        """
        return self.slow_candidates[0]

    @property
    def memory_model(self) -> ModelSettings:
        """Returns model settings for background long-term memory evaluation and consolidation."""
        # Keep effort="medium": high exhausts compaction output tokens; low trips deletion guard.
        memory = self._models().memory
        return ModelSettings(name=memory.name, effort=memory.effort)


class RouteClassification(BaseModel):
    """Structured reply-mode classification returned by the route model."""

    decision: Literal["IMAGE", "VIDEO", "QA"] = Field(
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
    link_context_sources: list[LinkSourceName] = Field(
        default_factory=list,
        description=(
            "Registered linked-post sources whose actual content the user wants analyzed, "
            "summarized, or discussed. Empty when every matching link is incidental. Consumed "
            "only on the QA route to decide which source builders may start."
        ),
    )
    # Overrides `slow_model.effort` on QA; every literal must be supported by `slow_model`
    # and its proxy fallback chain.
    effort: Literal["low", "medium", "high"] = Field(
        default="medium",
        description=(
            "Reasoning effort the answer model should spend. medium is the ordinary grade and "
            "the answer whenever you are unsure. Use low only for a message that asks for "
            "nothing, looks nothing up and works nothing out — banter and greetings, even when "
            "phrased as a question — and that is fully answerable from what you were shown. "
            "Use high only for work that needs real reasoning: writing or debugging code, "
            "maths, or a multi-step analysis, comparison or plan, including a follow-up that "
            "asks to fix or extend code or a calculation you were shown."
        ),
    )


# The route's schema on a turn that offers optional recall candidates. No docstring on purpose:
# under `text_format=` a class docstring is sent as the schema's description, and this shape was
# measured without one.
class RecallRouteClassification(RouteClassification):
    recall_user_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Ids from the additional-members block whose long-term memory the latest message "
            "obliquely refers to. Empty is the normal case."
        ),
    )
