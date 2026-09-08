"""Tests for what a failed read of a linked post is classified as.

The whole point of the classifier is the reaction it decides, and getting a refusal and a
missing post the wrong way round is the worst answer the expansion feature can give: it tells
someone a working link is dead. So these pin the mapping status by status rather than trusting
a range check to keep meaning what it meant.
"""

from collections.abc import Callable

import pytest
import requests

from discordbot.utils.threads import ThreadsDownloader
from discordbot.utils.facebook import FacebookDownloader
from discordbot.utils.instagram import InstagramDownloader
from discordbot.utils.link_errors import (
    LinkReadError,
    LinkRetryableError,
    LinkUnavailableError,
    link_fetch_error,
)


def _http_error(*, status: int) -> requests.HTTPError:
    """Builds the error `raise_for_status` raises for one status code."""
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"{status} error", response=response)


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_a_platform_asking_us_to_come_back_is_retryable(status: int) -> None:
    """429 says exactly that and a 5xx says the failure is the server's own."""
    error = link_fetch_error(error=_http_error(status=status), url="https://example.test/p/1")

    assert isinstance(error, LinkRetryableError)


@pytest.mark.parametrize("status", [404, 410])
def test_a_post_the_platform_says_is_gone_is_unavailable(status: int) -> None:
    """The server looked and there is nothing to serve, which no retry changes."""
    error = link_fetch_error(error=_http_error(status=status), url="https://example.test/p/1")

    assert isinstance(error, LinkUnavailableError)


@pytest.mark.parametrize(
    "failure",
    [requests.Timeout("too slow"), requests.ConnectionError("reset"), requests.ConnectTimeout()],
)
def test_a_request_that_never_got_an_answer_is_retryable(
    failure: requests.RequestException,
) -> None:
    """No response at all is about the network, never about the post."""
    error = link_fetch_error(error=failure, url="https://example.test/p/1")

    assert isinstance(error, LinkRetryableError)


@pytest.mark.parametrize("status", [400, 401, 403])
def test_an_ambiguous_refusal_keeps_the_plain_error(status: int) -> None:
    """A 403 logged out could be a post we may not read or a wall that lifts in minutes.

    Guessing writes a reaction that lies half the time, so it stays the generic failure it
    already was until somebody measures that platform.
    """
    error = link_fetch_error(error=_http_error(status=status), url="https://example.test/p/1")

    assert type(error) is RuntimeError


def test_every_classified_error_is_still_a_runtime_error() -> None:
    """Callers already catch `RuntimeError` around a parse, `/clean_threads_url` among them."""
    assert issubclass(LinkReadError, RuntimeError)
    assert issubclass(LinkRetryableError, LinkReadError)
    assert issubclass(LinkUnavailableError, LinkReadError)


def test_the_message_still_names_the_url_it_could_not_read() -> None:
    """Unchanged from before the classifier existed; the logs are grepped on it."""
    error = link_fetch_error(error=_http_error(status=503), url="https://example.test/p/1")

    assert "https://example.test/p/1" in str(error)


@pytest.mark.parametrize(
    "build_downloader",
    [FacebookDownloader, InstagramDownloader, lambda: ThreadsDownloader(output_folder="")],
    ids=["facebook", "instagram", "threads"],
)
def test_a_refused_page_leaves_each_reader_as_a_retryable_error(
    build_downloader: Callable[[], object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three readers wrapped every fetch failure in a bare `RuntimeError` before this.

    Which meant a 429 reached the channel as the cross that says the bot broke. Stubbing
    `requests.get` rather than `_fetch_page` is the point: `_fetch_page` is the seam every
    other test in the suite replaces, so it is the one thing nothing else exercises.
    """

    def refuse(**kwargs: object) -> requests.Response:
        """Answers the way a platform under load does."""
        response = requests.Response()
        response.status_code = 429
        response.url = str(kwargs["url"])
        return response

    monkeypatch.setattr(target=requests, name="get", value=refuse)
    downloader = build_downloader()

    with pytest.raises(LinkRetryableError):
        downloader._fetch_page(url="https://example.test/p/1")  # ty: ignore[unresolved-attribute]
