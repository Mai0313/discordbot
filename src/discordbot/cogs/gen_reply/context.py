"""The per-message reply payload `gen_reply` builds once and every later phase reads back.

`ReplyContext` is the seam between the pipeline's phases, which is why it sits in its own module
instead of inside the cog that fills it. `_prepare_reply_context` builds one speculatively, as its
own task running concurrently with routing: the channel history render, the bot's per-server
memory, the deterministic user memories, the author's tone note and the optional memory-selection
call all land in it before the route decision is known. Everything that goes in was read, never
written, so a route that does not want it can drop the whole object.

The blocks are kept apart rather than pre-ordered, because each consumer assembles a different
input from them. The answer (`_handle_message_reply`) emits history, then server memory / user
memory / tone, then the reference chain, then the linked-post blocks, then the current message.
The IMAGE and VIDEO persona replies (`_stream_media_persona_reply`) reuse the same object but
leave out the server memory and the linked-post blocks. The per-server memory update takes
`message_list`, the plain transcript view. `link_blocks` is the one field written after
construction: the pipeline splices the resolved linked-post blocks in with `model_copy` once
routing has said which sources to read.

Every block is an OpenAI Responses SDK TypedDict handed straight to the request, so the fields are
wrapped in `SkipValidation` — pydantic validation would rebuild each block into a fresh dict on
construction. Nothing here touches Discord or an LLM; it is inert data passed between phases.
"""

from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_param import EasyInputMessageParam


class ReplyContext(BaseModel):
    """Reply inputs built once per message and shared across pipeline phases.

    Built speculatively by `_prepare_reply_context` while the route decision is
    still in flight; it carries everything `_handle_message_reply` needs so the
    answer phase adds no further context work.

    How much of it is populated depends on the route that asked for it: the speculative build
    runs at `history_limit=30` with memory on, SUMMARY rebuilds one at `history_limit=200` with
    memory off, and only a QA route ever fills `link_blocks`. Every field has a default, so an
    empty instance is a valid context and an optional block that was never built is simply
    absent from the assembled input rather than an error.
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
        default=None, description="Rendered deterministic and optional user-memory block."
    )
    tone_block: SkipValidation[EasyInputMessageParam | None] = Field(
        default=None, description="Rendered tone-preference block for the message author, if any."
    )
    link_blocks: SkipValidation[list[EasyInputMessageParam]] = Field(
        default_factory=list,
        description=(
            "Rendered linked-post context blocks in LINK_CONTEXT_SOURCES order, "
            "injected before the current message."
        ),
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
        """History, reference, and current blocks in transcript order.

        The memory, tone, and linked-post blocks are deliberately left out: this view feeds the
        per-server memory update, and injected memory fed back in would be re-ingested as if the
        conversation had said it.

        Returns:
            A fresh list of the three transcript block groups; mutating it does not touch the
            context.
        """
        return [*self.hist_messages, *self.reference_messages, *self.current_message]
