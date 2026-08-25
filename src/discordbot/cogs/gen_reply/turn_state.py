"""The two things a failing turn needs that the frame it fails in cannot see.

Both are `ContextVar`s for the same reason: a pipeline failure surfaces in `on_message`, several
frames above the code that picked a model or opened a reply, and nextcord dispatches each
`on_message` as its own task, which copies the context — so a write here can never be read by
another user's turn.
"""

from typing import TYPE_CHECKING
from contextvars import ContextVar

if TYPE_CHECKING:
    from discordbot.cogs.gen_reply.streaming import ResponseStreamer


# The model this turn most recently dispatched on, so `gen_reply failed` can name it: a provider
# error rarely says which model it refused. Set only where the turn itself dispatches (route,
# answer, image, video); a generator that swallows its own failure logs its own model and never
# reaches the reader, and None means the turn failed before any model was asked for anything.
dispatched_model: ContextVar[str | None] = ContextVar("gen_reply_dispatched_model", default=None)


# The turn's UNFINISHED answer, so the pipeline's failure path can land its error on the reply
# already on screen. It holds the streamer rather than the message because what the error path
# needs is decided at failure time -- whether a reply exists at all (`withdraw_retry_notice` and a
# mid-stream delete each drop it) and what it is showing. Published by `ResponseStreamer.stream`
# for a `carries_turn_notices` streamer, and taken back the moment its answer is written in full.
current_answer_streamer: "ContextVar[ResponseStreamer | None]" = ContextVar(
    "gen_reply_answer_streamer", default=None
)
