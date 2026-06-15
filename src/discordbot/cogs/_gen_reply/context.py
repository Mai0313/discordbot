"""Pre-built reply context shared by routing, memory selection, and the answer."""

from nextcord import Message
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_param import EasyInputMessageParam


class RenderedHistory(BaseModel):
    """One channel-history fetch shared by the rendered context and the memory allowlist.

    History used to be fetched twice (once to render Responses input, once for the raw
    authors and mentions the allowlist needs); carrying both views of a single fetch
    removes that duplicate Discord API call from the reply critical path.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rendered: SkipValidation[list[EasyInputMessageParam]] = Field(
        default_factory=list, description="History context blocks rendered for the answer input."
    )
    rendered_text_only: SkipValidation[list[EasyInputMessageParam]] = Field(
        default_factory=list,
        description="Same history blocks rendered with attachment markers, for memory selection.",
    )
    raw: SkipValidation[list[Message]] = Field(
        default_factory=list,
        description="Raw history messages backing the rendered blocks, for the allowlist.",
    )


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
