"""Pre-built reply context shared by routing, memory selection, and the answer."""

from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_param import EasyInputMessageParam


class ReplyContext(BaseModel):
    """Reply inputs built once per message and shared across pipeline phases.

    Built speculatively by `_prepare_reply_context` while the route decision is
    still in flight; it carries everything `_handle_message_reply` needs so the
    answer phase adds no further context work.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hist_messages: SkipValidation[list[EasyInputMessageParam]] = Field(
        default_factory=list, description="Rendered channel-history context blocks."
    )
    reference_messages: SkipValidation[list[EasyInputMessageParam]] = Field(
        default_factory=list, description="Rendered reference-chain context blocks (depth <= 3)."
    )
    current_message: SkipValidation[list[EasyInputMessageParam]] = Field(
        default_factory=list,
        description="Header plus the processed current message; stays last in the answer input.",
    )
    server_memory_block: SkipValidation[EasyInputMessageParam | None] = Field(
        default=None, description="Rendered server-memory context block, if any."
    )
    memory_block: SkipValidation[EasyInputMessageParam | None] = Field(
        default=None, description="Rendered selected-user-memory context block, if any."
    )
    tone_block: SkipValidation[EasyInputMessageParam | None] = Field(
        default=None, description="Rendered tone-preference block for the message author, if any."
    )
    threads_block: SkipValidation[list[EasyInputMessageParam]] = Field(
        default_factory=list,
        description="Rendered Threads-post context blocks, injected before the current message.",
    )
    memory_labels: list[str] = Field(
        default_factory=list, description="Footer labels of users whose memory was injected."
    )
    selection_input_tokens: int = Field(
        default=0, description="Input tokens spent by the memory selection request."
    )
    selection_output_tokens: int = Field(
        default=0, description="Output tokens spent by the memory selection request."
    )

    @property
    def message_list(self) -> list[EasyInputMessageParam]:
        """History, reference, and current blocks in transcript order."""
        return [*self.hist_messages, *self.reference_messages, *self.current_message]
