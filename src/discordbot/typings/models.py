from typing import Literal

from pydantic import BaseModel
from openai.types.responses.tool_param import ToolParam
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.reasoning import Reasoning

from discordbot.utils.model_pricing import get_supported_modalities


class ModelSettings(BaseModel):
    """Model name and reasoning effort that should be used together."""

    name: str
    effort: ReasoningEffort | None

    @property
    def reasoning(self) -> Reasoning:
        """Builds Responses API reasoning options for this model."""
        if self.effort is None:
            raise ValueError("Model effort must be set to build reasoning options.")
        return Reasoning(effort=self.effort, summary="auto")

    @property
    def input_modalities(self) -> set[str]:
        """Input modalities this model accepts, derived from the LiteLLM price table."""
        return get_supported_modalities(model_name=self.name)

    @property
    def tools(self) -> list[ToolParam]:
        """Built-in tools (web search / URL context / fetch) wired to this model's provider."""
        if "gemini" in self.name:
            return [{"googleSearch": {}}, {"urlContext": {}}]
        if "claude" in self.name:
            return [
                {"type": "web_search_20260209", "name": "web_search"},
                {"type": "web_fetch_20260209", "name": "web_fetch"},
            ]
        return [{"type": "web_search"}]


class RouteDecision(BaseModel):
    """Structured routing decision returned by the model.

    Attributes:
        decision: The reply mode selected for the incoming Discord message.
    """

    decision: Literal["IMAGE", "VIDEO", "QA", "SUMMARY"]
