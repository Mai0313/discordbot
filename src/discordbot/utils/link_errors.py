"""What a failed read of a linked post means, shared by every platform that reads one.

The four expansion cogs answer with a reaction and nothing else, so the reaction has to carry
the difference between "the platform refused us, the link is fine" and "there is no post here".
Douyin could already tell those apart because it has its own error tree; Threads, Facebook and
Instagram wrapped every fetch failure in a bare `RuntimeError`, so a 429 reached the reader as
the cross that means the bot broke. This module is the vocabulary the other three were missing,
and `services/platforms/douyin.py` re-parents its own tree onto it rather than keeping a second one.

Every class here is a `RuntimeError`, which is what keeps it a drop-in: the callers that already
catch `RuntimeError` around a parse — `/clean_threads_url` among them — keep catching these.

`link_fetch_error` classifies only what HTTP says unambiguously and hands everything else back
as today's bare `RuntimeError`. A 403 is the case that argument is really about: logged out, it
could be a post we may not read or a wall that lifts in ten minutes, and guessing either way
writes a reaction that lies half the time. Reclassifying one needs evidence about that platform,
not a rule invented here.
"""

import requests

# What HTTP says to try the same request again for, with no judgement of our own on top: 429 is
# the server asking for exactly that, and a 5xx is the server saying the failure is its own.
_TOO_MANY_REQUESTS = 429
# What the server answers when it has looked and there is nothing to serve.
_UNAVAILABLE_STATUS = frozenset({404, 410})


class LinkReadError(RuntimeError):
    """A linked post could not be read. The base every platform's own errors sit under."""


class LinkRetryableError(LinkReadError):
    """The platform refused the request or the transport failed; the post itself is fine."""


class LinkUnavailableError(LinkReadError):
    """The platform answered and there is no readable post in it (deleted, private, gone)."""


def is_retryable_fetch_failure(*, error: requests.RequestException) -> bool:
    """Reports whether a failed request is worth making again, exactly as HTTP frames it.

    Split out from `link_fetch_error` because Douyin raises its own error classes rather than
    these, and a transport failure has to mean the same thing on all four platforms or the
    reaction stops meaning anything.

    Args:
        error: What `requests` raised.

    Returns:
        True for a request that never got an answer, a 429, or a 5xx.
    """
    # `Timeout` subclasses `ConnectionError` for `ConnectTimeout` only, so both are named.
    if isinstance(error, requests.Timeout | requests.ConnectionError):
        return True
    # `Response.__bool__` is `self.ok`, so a 429 and a 503 are both FALSY: testing the response
    # for truth here would classify nothing and raise nothing while looking correct.
    if error.response is None:
        return False
    return error.response.status_code == _TOO_MANY_REQUESTS or error.response.status_code >= 500


def link_fetch_error(*, error: requests.RequestException, url: str) -> RuntimeError:
    """Maps a failed page fetch to the error whose class says what the reader should do.

    Args:
        error: What `requests` raised.
        url: The page being fetched, for the message.

    Returns:
        The exception to raise, carrying the same message every caller wrote before this
        existed: a `LinkRetryableError` or `LinkUnavailableError` where the status or the
        transport failure is unambiguous, and a plain `RuntimeError` everywhere else.
    """
    message = f"Failed to fetch HTML from {url}: {error}"
    if is_retryable_fetch_failure(error=error):
        return LinkRetryableError(message)
    if error.response is not None and error.response.status_code in _UNAVAILABLE_STATUS:
        return LinkUnavailableError(message)
    return RuntimeError(message)
