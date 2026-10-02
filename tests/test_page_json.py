"""Tests for the shared page-reading primitives in `page_json`.

Each platform's own tests stub the fetch and serve canned payloads, so what the shared pieces
promise is pinned here once.
"""

import pytest

from discordbot.services.platforms import page_json
from discordbot.services.platforms.page_json import JsonValue, count_of, fetch_page


def test_fetch_page_reports_where_the_request_landed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A share link names its post only where the redirect ended, and a login wall is one too.

    The parser tests all stub the fetch, so nothing else exercises what the real one reports: a
    fetch that stopped following redirects, or handed back the URL it asked for, would leave
    every share link unreadable with all of them still green.
    """
    asked = "https://www.threads.com/share/DfX81RWN8"
    landed = "https://www.threads.com/@target_author/post/TARGET?xmt=AQF0p6Ufiuvt"

    class _Response:
        """A response that came back from somewhere other than it was asked for."""

        text = "<html>the post page</html>"
        url = landed

        def raise_for_status(self) -> None:
            """Accepts the transfer."""

    requested: list[str] = []

    def fake_get(url: str, **kwargs: object) -> _Response:
        """Records what was asked for and answers as the redirect chain's last hop."""
        del kwargs
        requested.append(url)
        return _Response()

    monkeypatch.setattr(target=page_json.requests, name="get", value=fake_get)

    fetched = fetch_page(url=asked, headers={}, timeout=1.0)

    assert requested == [asked]
    assert fetched.html == "<html>the post page</html>"
    assert fetched.final_url == landed


@pytest.mark.parametrize(
    argnames=("served", "expected"),
    argvalues=[("8,855", 8855), ({"count": 8855}, 8855), (8855, 8855), (True, 0), (None, 0)],
)
def test_a_count_reads_the_same_in_any_shape_the_page_serves(
    served: JsonValue, expected: int
) -> None:
    """A formatted or wrapped count is still the number, and a flag is not a count."""
    assert count_of(value=served) == expected
