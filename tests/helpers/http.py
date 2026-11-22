"""Shared HTTP response and session doubles backed by real `requests.Response` objects."""

import json
from typing import Any, Self
from collections.abc import Callable, Iterator

import pytest
import requests
from requests.cookies import cookiejar_from_dict

type RequestHandler = Callable[[str, dict[str, object]], requests.Response]


class StreamedResponse(requests.Response):
    """Real `requests.Response` with optional chunk streaming, stall, and body-read guards."""

    def __init__(  # noqa: PLR0913 -- optional HTTP response knobs for test doubles
        self,
        status_code: int = 200,
        text: str = "",
        body: bytes | None = None,
        json_body: object | None = None,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        url: str = "",
        stall: bool = False,
        chunks: Callable[[int], Iterator[bytes]] | None = None,
        forbid_body_read: bool = False,
    ) -> None:
        """Initializes status, headers, cookies, body bytes, and streaming test hooks."""
        super().__init__()
        self.status_code = status_code
        self.url = url
        self.encoding = "utf-8"
        if headers:
            self.headers.update(headers)
        if cookies:
            self.cookies.update(cookiejar_from_dict(cookie_dict=cookies))
        if json_body is not None:
            raw_bytes = json.dumps(obj=json_body).encode("utf-8")
        elif body is not None:
            raw_bytes = body
        else:
            raw_bytes = text.encode("utf-8")
        self._body = raw_bytes
        self._content = raw_bytes
        self._content_consumed = True
        self._stall = stall
        self._chunks = chunks
        self._forbid_body_read = forbid_body_read
        self.body_reads = 0
        self.closed = False

    @property
    def text(self) -> str:
        """Returns decoded text, or fails the test when body reads are forbidden."""
        if self._forbid_body_read:
            raise AssertionError("response body must not be read")
        return super().text

    @property
    def content(self) -> bytes:
        """Returns raw body bytes, or fails the test when body reads are forbidden."""
        if self._forbid_body_read:
            raise AssertionError("response body must not be read")
        return super().content

    def iter_content(
        self, chunk_size: int | None = 1, decode_unicode: bool = False
    ) -> Iterator[Any]:
        """Yields configured chunks or body bytes, counting reads and stalling when requested."""
        del decode_unicode
        if self._forbid_body_read:
            raise AssertionError("response body must not be read")
        self.body_reads += 1
        if self._chunks is not None:
            yield from self._chunks(chunk_size or 1)
        else:
            yield self._body
        if self._stall:
            raise requests.ConnectionError("read timed out")

    def close(self) -> None:
        """Marks the response closed and releases underlying response resources."""
        self.closed = True
        super().close()


def make_response(  # noqa: PLR0913 -- mirrors StreamedResponse optional knobs
    status_code: int = 200,
    text: str = "",
    body: bytes | None = None,
    json_body: object | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    url: str = "",
    stall: bool = False,
    chunks: Callable[[int], Iterator[bytes]] | None = None,
    forbid_body_read: bool = False,
) -> StreamedResponse:
    """Builds a real `requests.Response` double with the requested HTTP fields and hooks."""
    return StreamedResponse(
        status_code=status_code,
        text=text,
        body=body,
        json_body=json_body,
        headers=headers,
        cookies=cookies,
        url=url,
        stall=stall,
        chunks=chunks,
        forbid_body_read=forbid_body_read,
    )


def _resolve_handler(
    handler: RequestHandler | None, response: requests.Response | None
) -> RequestHandler:
    """Returns a request handler from either `handler` or a fixed `response`."""
    if handler is not None:
        return handler
    assert response is not None, "pass either handler or response"
    return lambda _url, _kwargs: response


def install_session(
    monkeypatch: pytest.MonkeyPatch,
    target: object,
    handler: RequestHandler | None = None,
    response: requests.Response | None = None,
) -> list[dict[str, Any]]:
    """Replaces `target.Session` with a recording stub, returning captured GET call dicts."""
    resolved = _resolve_handler(handler=handler, response=response)
    calls: list[dict[str, Any]] = []

    class _FakeSession:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def get(self, url: str, **kwargs: object) -> requests.Response:
            calls.append({"url": url, **kwargs})
            return resolved(url, kwargs)

    monkeypatch.setattr(target=target, name="Session", value=_FakeSession)
    return calls


def install_get(
    monkeypatch: pytest.MonkeyPatch,
    target: object,
    handler: RequestHandler | None = None,
    response: requests.Response | None = None,
) -> list[dict[str, Any]]:
    """Replaces `target.get` with a recording stub, returning captured GET call dicts."""
    resolved = _resolve_handler(handler=handler, response=response)
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, **kwargs: object) -> requests.Response:
        calls.append({"url": url, **kwargs})
        return resolved(url, kwargs)

    monkeypatch.setattr(target=target, name="get", value=fake_get)
    return calls
