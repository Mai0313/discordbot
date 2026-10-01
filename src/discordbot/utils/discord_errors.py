"""What a Discord error raised by a send means."""

from nextcord import NotFound, HTTPException


def is_reply_target_gone(error: HTTPException) -> bool:
    """Whether a reply failed because the message it replies to no longer exists.

    Discord answers that with NotFound or with 50035. 50035 is the generic invalid-form-body code,
    so it means this only on a send that carries a message reference; on an edit it is a
    rejected body.
    """
    return isinstance(error, NotFound) or error.code == 50035
