from typing import Literal

from pydantic import BaseModel
from openai.types.responses.tool_param import ToolParam
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.reasoning import Reasoning


class ModelSettings(BaseModel):
    """Model name and reasoning effort that should be used together."""

    name: str
    effort: ReasoningEffort | None

    @property
    def reasoning(self) -> Reasoning:
        """Responses API reasoning options for this model.

        Returns:
            Reasoning options using this model's configured effort and an
            automatic reasoning summary.

        Raises:
            ValueError: The model has no reasoning effort configured.
        """
        if self.effort is None:
            raise ValueError("Model effort must be set to build reasoning options.")
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


class RouteDecision(BaseModel):
    """Structured routing decision returned by the model.

    Attributes:
        decision: The reply mode selected for the incoming Discord message.
    """

    decision: Literal["IMAGE", "VIDEO", "QA", "SUMMARY"]
