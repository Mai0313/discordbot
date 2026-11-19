"""Shared exception hierarchy and HTTP failure classifiers for linked-post readers.

All classes inherit from `RuntimeError`. `link_fetch_error` classifies only unambiguous HTTP
statuses (429, 5xx, or transport failures as retryable; 404 and 410 as unavailable) and returns
bare `RuntimeError` for all others, including 403.
"""

import requests

_TOO_MANY_REQUESTS = 429
_UNAVAILABLE_STATUS = frozenset({404, 410})


class LinkReadError(RuntimeError):
    """A linked post could not be read. The base every platform's own errors sit under."""


class LinkRetryableError(LinkReadError):
    """The platform refused the request or the transport failed; the post itself is fine."""


class LinkUnavailableError(LinkReadError):
    """The platform answered and there is no readable post in it (deleted, private, gone)."""


def is_retryable_fetch_failure(error: requests.RequestException) -> bool:
    """Returns whether `error` is a transport failure, HTTP 429, or HTTP 5xx worth retrying.

    Args:
        error (requests.RequestException): What `requests` raised.
    """
    # ConnectTimeout subclasses ConnectionError, ReadTimeout does not; ChunkedEncodingError neither
    if isinstance(
        error,
        requests.Timeout | requests.ConnectionError | requests.exceptions.ChunkedEncodingError,
    ):
        return True
    # Response.__bool__ is self.ok (429/503 are falsy); check `is None` explicitly.
    if error.response is None:
        return False
    return error.response.status_code == _TOO_MANY_REQUESTS or error.response.status_code >= 500


def link_fetch_error(error: requests.RequestException, url: str) -> RuntimeError:
    """Returns `LinkRetryableError`, `LinkUnavailableError`, or `RuntimeError` for `error`.

    Args:
        error (requests.RequestException): What `requests` raised.
        url (str): Page URL being fetched, included in the error message.
    """
    message = f"Failed to fetch HTML from {url}: {error}"
    if is_retryable_fetch_failure(error=error):
        return LinkRetryableError(message)
    if error.response is not None and error.response.status_code in _UNAVAILABLE_STATUS:
        return LinkUnavailableError(message)
    return RuntimeError(message)
