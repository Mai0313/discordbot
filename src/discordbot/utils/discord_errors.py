"""What a Discord error raised by a send means."""

from nextcord import NotFound, HTTPException


def is_reply_target_gone(error: HTTPException) -> bool:
    """Returns whether a reply send failed for a missing reference (`NotFound` or `50035`).

    Valid only for sends carrying a message reference; on edits, `50035` indicates a rejected body.

    Args:
        error (HTTPException): Exception raised by the reply send.
    """
    return isinstance(error, NotFound) or error.code == 50035
