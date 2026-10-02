"""Tests for what a failed read of a linked post is classified as.

The whole point of the classifier is the reaction it decides, and getting a refusal and a
missing post the wrong way round is the worst answer the expansion feature can give: it tells
someone a working link is dead. So these pin the mapping status by status rather than trusting
a range check to keep meaning what it meant.
"""

import io
from collections.abc import Callable

import pytest
import urllib3
import requests

from discordbot.utils.link_errors import (
    LinkReadError,
    LinkRetryableError,
    LinkUnavailableError,
    link_fetch_error,
)
from discordbot.utils.expansion_cog import expansion_failure_emoji
from discordbot.services.platforms.douyin import (
    DouyinError,
    DouyinBlockedError,
    DouyinUnavailableError,
)
from discordbot.services.platforms.threads import ThreadsDownloader
from discordbot.services.platforms.facebook import FacebookDownloader
from discordbot.utils.expansion_placeholder import (
    EXPANSION_UNREADABLE_EMOJI,
    EXPANSION_RETRY_LATER_EMOJI,
)
from discordbot.services.platforms.instagram import InstagramDownloader


def _http_error(status: int) -> requests.HTTPError:
    """Builds the error `raise_for_status` raises for one status code."""
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"{status} error", response=response)


@pytest.mark.parametrize(argnames="status", argvalues=[429, 500, 502, 503, 504])
def test_a_platform_asking_us_to_come_back_is_retryable(status: int) -> None:
    """429 says exactly that and a 5xx says the failure is the server's own."""
    error = link_fetch_error(error=_http_error(status=status), url="https://example.test/p/1")

    assert isinstance(error, LinkRetryableError)


@pytest.mark.parametrize(argnames="status", argvalues=[404, 410])
def test_a_post_the_platform_says_is_gone_is_unavailable(status: int) -> None:
    """The server looked and there is nothing to serve, which no retry changes."""
    error = link_fetch_error(error=_http_error(status=status), url="https://example.test/p/1")

    assert isinstance(error, LinkUnavailableError)


@pytest.mark.parametrize(
    argnames="failure",
    argvalues=[
        requests.Timeout("too slow"),
        requests.ConnectionError("reset"),
        requests.ConnectTimeout(),
    ],
)
def test_a_request_that_never_got_an_answer_is_retryable(
    failure: requests.RequestException,
) -> None:
    """No response at all is about the network, never about the post."""
    error = link_fetch_error(error=failure, url="https://example.test/p/1")

    assert isinstance(error, LinkRetryableError)


def test_a_body_cut_off_mid_transfer_is_retryable() -> None:
    """A connection lost part-way through the body is the same network failure, only later.

    The error is the one `requests` itself raises over a truncated body rather than a class
    picked by hand: it subclasses neither `ConnectionError` nor `Timeout` and carries no
    response, so only reading a real cut-off body pins what the classifier has to accept.
    """
    response = requests.Response()
    response.status_code = 200
    response.raw = urllib3.HTTPResponse(
        body=io.BytesIO(b"x" * 500), headers={"Content-Length": "100000"}, preload_content=False
    )
    with pytest.raises(requests.RequestException) as dropped:
        _ = response.content

    error = link_fetch_error(error=dropped.value, url="https://example.test/p/1")

    assert isinstance(error, LinkRetryableError)


@pytest.mark.parametrize(argnames="status", argvalues=[400, 401, 403])
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
    argnames="build_downloader",
    argvalues=[
        FacebookDownloader,
        InstagramDownloader,
        lambda: ThreadsDownloader(output_folder=""),
    ],
    ids=["facebook", "instagram", "threads"],
)
def test_a_refused_page_leaves_each_reader_as_a_retryable_error(
    build_downloader: Callable[[], object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused page is retryable, never the bare `RuntimeError` that paints the failure cross.

    Stubbing `requests.get` rather than `_fetch_page` is the point: `_fetch_page` is the seam
    every other test in the suite replaces, so it is the one thing nothing else exercises.
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


@pytest.mark.parametrize(
    argnames=("error", "expected"),
    argvalues=[
        (DouyinBlockedError("bot wall"), EXPANSION_RETRY_LATER_EMOJI),
        (DouyinUnavailableError("filtered"), EXPANSION_UNREADABLE_EMOJI),
        (DouyinError("unreadable"), EXPANSION_UNREADABLE_EMOJI),
    ],
    ids=["blocked", "gone", "unreadable"],
)
def test_each_douyin_error_earns_the_mark_of_the_shared_class_it_sits_under(
    error: DouyinError, expected: str
) -> None:
    """Douyin raises its own classes, so where each sits in the shared tree is its reaction."""
    assert expansion_failure_emoji(error=error) == expected
