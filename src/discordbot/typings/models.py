from typing import Literal

from pydantic import BaseModel
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.reasoning import Reasoning


class ModelSettings(BaseModel):
    """Model name and reasoning effort that should be used together."""

    model_name: str
    model_effort: ReasoningEffort | None

    @property
    def reasoning(self) -> Reasoning:
        """Builds Responses API reasoning options for this model."""
        if self.model_effort is None:
            raise ValueError("Model effort must be set to build reasoning options.")
        return Reasoning(effort=self.model_effort, summary="auto")


class RouteDecision(BaseModel):
    """Structured routing decision returned by the model.

    Attributes:
        decision: The reply mode selected for the incoming Discord message.
    """

    decision: Literal["IMAGE", "VIDEO", "QA", "SUMMARY"]
