from typing import Literal

from pydantic import BaseModel
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.reasoning import Reasoning


class ModelSettings(BaseModel):
    """Model name and reasoning effort that should be used together."""

    model_name: str
    model_effort: ReasoningEffort

    @property
    def reasoning(self) -> Reasoning:
        """Builds Responses API reasoning options for this model."""
        return Reasoning(effort=self.model_effort, summary="auto")


class RouteDecision(BaseModel):
    """Structured routing decision returned by the model.

    Attributes:
        decision: The reply mode selected for the incoming Discord message.
    """

    decision: Literal["IMAGE", "VIDEO", "QA", "SUMMARY"]
