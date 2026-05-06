from typing import Literal

from pydantic import BaseModel
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


class RouteDecision(BaseModel):
    """Structured routing decision returned by the model.

    Attributes:
        decision: The reply mode selected for the incoming Discord message.
    """

    decision: Literal["IMAGE", "VIDEO", "QA", "SUMMARY"]
