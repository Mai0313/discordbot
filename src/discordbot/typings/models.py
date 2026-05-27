from typing import Literal
from datetime import UTC, datetime

from pydantic import Field, BaseModel, computed_field
from openai.types.responses.tool_param import ToolParam
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.reasoning import Reasoning


class ModelSettings(BaseModel):
    """Model name and reasoning effort that should be used together."""

    name: str
    effort: ReasoningEffort = Field(default="none")

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

        Returns:
            Gemini models receive googleSearch and urlContext tools. Claude
            models receive web_search and web_fetch tools. Other models
            receive the OpenAI web_search tool.
        """
        if "gemini" in self.name:
            return [{"googleSearch": {}}, {"urlContext": {}}]
        if "claude" in self.name:
            return [
                {"type": "web_search_20260209", "name": "web_search"},
                {"type": "web_fetch_20260209", "name": "web_fetch"},
            ]
        return [{"type": "web_search"}]


class RuntimeModelCatalog(BaseModel):
    """Runtime model settings used by Discord bot LLM paths.

    Keep caller lists in sync when moving runtime model usage.
    """

    @computed_field
    @property
    def is_peak(self) -> bool:
        """Whether runtime model selection is in the peak-hour window.

        Returns:
            True during UTC weekdays from 08:00 to 17:00, otherwise False.
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
    def fast_model(self) -> ModelSettings:
        """The model settings for lightweight reply-generation tasks.

        Callers: `_handle_image_reply`, `_route_message`, `_generate_reply`, `dealer`.

        Returns:
            Fast model settings used for routing, image captions, and short
            Discord replies.
        """
        fast_model = ModelSettings(name="gemini-flash-lite-latest", effort="none")
        return fast_model

    @property
    def slow_model(self) -> ModelSettings:
        """The model settings for full text replies and summaries.

        Uses `gemini-flash-latest` during UTC weekday 08:00 to 17:00 peak hours and `gemini-3.5-flash` outside that peak window.

        Callers: `_get_attachment_parts`, `_handle_message_reply`.

        Returns:
            Slow-path model settings for reply and summary generation.
        """
        if self.is_peak:
            return ModelSettings(name="gemini-flash-latest", effort="high")
        return ModelSettings(name="gemini-3.5-flash", effort="high")


class RouteDecision(BaseModel):
    """Structured routing decision returned by the model.

    Attributes:
        decision: The reply mode selected for the incoming Discord message.
    """

    decision: Literal["IMAGE", "VIDEO", "QA", "SUMMARY"]
