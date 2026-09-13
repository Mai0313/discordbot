"""Covers the Twitter reader against payloads built the way `tests/test_facebook.py` builds pages.

The fixtures mirror the real syndication payload's shape rather than reproducing one: every key
here was read off a live response, and the ones that carry a trap have a test of their own. Nothing
in this file reaches the network: `_downloader` replaces the one request the module makes, and the
tests that are ABOUT that request replace `requests.get` under it instead, so the params, the
headers and the error classification are exercised rather than stubbed past.
"""

from typing import Any

import pytest
import requests

from discordbot.typings.timeouts import TWITTER_PAGE_TIMEOUT_SECONDS
from discordbot.utils.link_errors import LinkReadError, LinkRetryableError, LinkUnavailableError
from discordbot.services.platforms import twitter as twitter_module
from discordbot.services.platforms.twitter import (
    TWITTER_URL_RE,
    TwitterURL,
    TwitterDownloader,
    TwitterConversation,
)

_STATUS_ID = "1628549742539194368"
_URL = f"https://x.com/Dbacks/status/{_STATUS_ID}"


def _user(*, screen_name: str = "Dbacks") -> dict[str, Any]:
    """The author block, present on every live post and on every embedded one."""
    return {
        "id_str": "31164229",
        "name": "Arizona Diamondbacks",
        "screen_name": screen_name,
        "profile_image_url_https": "https://pbs.twimg.com/profile_images/1/a_normal.jpg",
    }


def _photo(*, name: str = "FpnFWuRaMAEGX5e") -> dict[str, Any]:
    """One still, with the `sizes` block that reports the original while the URL serves medium."""
    return {
        "type": "photo",
        "media_url_https": f"https://pbs.twimg.com/media/{name}.jpg",
        "original_info": {"height": 1500, "width": 1200},
        "sizes": {"large": {"h": 1500, "w": 1200, "resize": "fit"}},
    }


def _video(*, poster: str = "poster") -> dict[str, Any]:
    """One clip, with the rendition list in the order the endpoint serves it.

    The HLS manifest comes first and carries no bitrate, which is what makes "the first variant"
    the wrong way to pick a rendition and "the highest bitrate mp4" the right one.
    """
    return {
        "type": "video",
        "media_url_https": f"https://pbs.twimg.com/amplify_video_thumb/{poster}.jpg",
        "video_info": {
            "variants": [
                {"content_type": "application/x-mpegURL", "url": "https://video.twimg.com/x.m3u8"},
                {"bitrate": 256000, "content_type": "video/mp4", "url": "https://v.tw/480.mp4"},
                {"bitrate": 2176000, "content_type": "video/mp4", "url": "https://v.tw/1280.mp4"},
                {"bitrate": 832000, "content_type": "video/mp4", "url": "https://v.tw/640.mp4"},
            ]
        },
    }


def _tweet(
    *,
    text: str = "Play good.",
    screen_name: str = "Dbacks",
    media: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One live post, as the endpoint serves it.

    Everything past the three commonest knobs goes through `extra` under the payload's OWN key
    names (`parent`, `quoted_tweet`, `note_tweet`, `display_text_range`, ...), so a test reads as
    the response it is standing in for rather than as this helper's vocabulary.
    """
    payload: dict[str, Any] = {
        "__typename": "Tweet",
        "id_str": _STATUS_ID,
        "text": text,
        "lang": "en",
        "created_at": "2023-02-23T00:18:10.000Z",
        "favorite_count": 125,
        "conversation_count": 2,
        "user": _user(screen_name=screen_name),
    }
    if media is not None:
        payload["mediaDetails"] = media
        # The lossy convenience view the module deliberately does not read: it drops videos and
        # loses their position, so a fixture that populated it faithfully would hide a regression.
        payload["photos"] = [{"url": "wrong"} for item in media if item["type"] == "photo"]
    payload.update(extra or {})
    return payload


def _tombstone(
    *, text: str | None = "This Post was deleted by the Post author. Learn more"
) -> dict[str, Any]:
    """A refusal. `text` of None is the wordless variant 4 of 133 sampled refusals carried."""
    if text is None:
        return {"__typename": "TweetTombstone", "tombstone": {}}
    return {
        "__typename": "TweetTombstone",
        "tombstone": {"text": {"entities": [], "rtl": False, "text": text}},
    }


def _downloader(monkeypatch: pytest.MonkeyPatch, *, payload: dict[str, Any]) -> TwitterDownloader:
    """A downloader whose one request answers with `payload`."""
    downloader = TwitterDownloader()

    def fake_fetch(self: TwitterDownloader, *, status_id: str) -> dict[str, Any]:
        del self, status_id
        return payload

    monkeypatch.setattr(TwitterDownloader, "_fetch_tweet", fake_fetch)
    return downloader


def _parse(
    monkeypatch: pytest.MonkeyPatch, *, payload: dict[str, Any], url: str = _URL
) -> TwitterConversation:
    """Parses one payload into a conversation."""
    return _downloader(monkeypatch, payload=payload).parse_metadata(url=url)


@pytest.mark.parametrize(
    "url",
    [
        f"https://x.com/Dbacks/status/{_STATUS_ID}",
        f"https://twitter.com/Dbacks/status/{_STATUS_ID}",
        f"https://mobile.twitter.com/Dbacks/statuses/{_STATUS_ID}",
        f"https://m.x.com/Dbacks/status/{_STATUS_ID}",
        f"https://www.x.com/Dbacks/status/{_STATUS_ID}",
        f"https://x.com/Dbacks/status/{_STATUS_ID}?s=46&t=f4h6W3bSqlanXhWACchAYQ",
        f"https://x.com/Dbacks/status/{_STATUS_ID}/photo/1",
        f"https://x.com/Dbacks/status/{_STATUS_ID}/video/1",
        f"https://x.com/i/web/status/{_STATUS_ID}",
        f"https://x.com/Dbacks/status/{_STATUS_ID}/",
    ],
)
def test_every_form_of_a_post_url_names_the_same_post(url: str) -> None:
    """One post is spelled ten ways, and only the id in the middle of it means anything."""
    assert TWITTER_URL_RE.search(string=url)
    assert TwitterURL(raw_url=url).status_id == _STATUS_ID


def test_a_handle_that_is_not_the_author_still_names_the_post() -> None:
    """x.com serves the same post under any handle, so the id is the only load-bearing part.

    Measured: a URL carrying a handle that does not exist answers 307 to the real author's. So the
    parse must not reject it, and must not publish it back either — the author comes off the
    payload rather than out of the URL.
    """
    url = f"https://x.com/nobody_at_all_here/status/{_STATUS_ID}"

    assert TwitterURL(raw_url=url).status_id == _STATUS_ID


@pytest.mark.parametrize(
    "url",
    [
        "https://x.com/Dbacks",
        "https://x.com/",
        "https://x.com/i/lists/12345",
        "https://x.com/home",
        "https://x.com.attacker.com/a/status/123",
        "https://notx.com/a/status/123",
        "https://example.com/x.com/a/status/123",
    ],
)
def test_a_url_that_names_no_post_is_refused(url: str) -> None:
    """The pattern is path-anchored, so the registry needs no `url_filter` on top of it.

    The last two are the host-confusion pair every pattern in this package is anchored against:
    a lookalike domain and the real one buried in another site's path.
    """
    assert not TWITTER_URL_RE.search(string=url)


def test_a_text_post_carries_its_author_counters_and_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """The floor: everything a card shows for a post with no media."""
    conversation = _parse(monkeypatch, payload=_tweet())
    target = conversation.target

    assert target is not None
    assert target.text == "Play good."
    assert target.author_name == "Dbacks"
    assert target.like_count == 125
    assert target.comment_count == 2
    assert target.taken_at is not None
    assert target.url == f"https://x.com/Dbacks/status/{_STATUS_ID}"
    assert target.is_readable


def test_the_post_url_names_the_real_author_not_the_pasted_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Republishing the pasted handle would carry a stranger's name into the channel."""
    conversation = _parse(
        monkeypatch,
        payload=_tweet(screen_name="RealAuthor"),
        url=f"https://x.com/someone_else/status/{_STATUS_ID}",
    )
    target = conversation.target

    assert target is not None
    assert target.url == f"https://x.com/RealAuthor/status/{_STATUS_ID}"


def test_images_are_taken_at_original_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bare URL serves `medium` while the payload reports the original's dimensions beside it.

    Measured: bare 57,759 bytes at 900x1200, `?name=orig` 978,797 bytes at 3024x4032, for a payload
    that called it 3024x4032 either way. Dropping the suffix silently posts the small one.
    """
    conversation = _parse(monkeypatch, payload=_tweet(media=[_photo(), _photo(name="second")]))
    target = conversation.target

    assert target is not None
    assert target.image_urls == [
        "https://pbs.twimg.com/media/FpnFWuRaMAEGX5e.jpg?name=orig",
        "https://pbs.twimg.com/media/second.jpg?name=orig",
    ]


def test_a_video_takes_the_highest_bitrate_mp4_and_keeps_its_poster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first variant is the HLS manifest, which is not a file and carries no bitrate."""
    conversation = _parse(monkeypatch, payload=_tweet(media=[_video()]))
    target = conversation.target

    assert target is not None
    assert target.video_urls == ["https://v.tw/1280.mp4"]
    assert target.video_poster_urls == ["https://pbs.twimg.com/amplify_video_thumb/poster.jpg"]
    assert target.image_urls == []


def test_an_animated_gif_is_read_by_the_same_rendition_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Twitter models a GIF as an mp4, and it is the case the content-type filter exists for.

    A video leads with an HLS manifest to skip past; a GIF has none, and its one rendition sits at
    `bitrate: 0`. So "skip the first entry" reads a GIF as having no video at all, while filtering
    on `video/mp4` serves both — which is why the payload's own `type` is not branched on here.
    """
    gif = {
        "type": "animated_gif",
        "media_url_https": "https://pbs.twimg.com/tweet_video_thumb/gif.jpg",
        "video_info": {
            "variants": [
                {"bitrate": 0, "content_type": "video/mp4", "url": "https://v.tw/gif.mp4"}
            ]
        },
    }
    conversation = _parse(monkeypatch, payload=_tweet(media=[gif]))
    target = conversation.target

    assert target is not None
    assert target.video_urls == ["https://v.tw/gif.mp4"]
    assert target.video_poster_urls == ["https://pbs.twimg.com/tweet_video_thumb/gif.jpg"]


def test_mixed_media_keeps_every_item_the_payload_carried(monkeypatch: pytest.MonkeyPatch) -> None:
    """`mediaDetails` is the authoritative list; `photos` drops the video and loses its place.

    The fixture's `photos` is deliberately wrong, so a reader that took it would fail here.
    """
    conversation = _parse(
        monkeypatch, payload=_tweet(media=[_photo(name="one"), _video(), _photo(name="two")])
    )
    target = conversation.target

    assert target is not None
    assert target.image_urls == [
        "https://pbs.twimg.com/media/one.jpg?name=orig",
        "https://pbs.twimg.com/media/two.jpg?name=orig",
    ]
    assert target.video_urls == ["https://v.tw/1280.mp4"]


def test_the_trailing_media_link_is_cut_on_utf16_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two index bases live in one payload, and only one of them indexes a Python string.

    `display_text_range` counts UTF-16 CODE UNITS while `entities.*.indices` count codepoints, so
    a post carrying anything outside the BMP — an emoji — slices short when the range is used
    directly, leaking the leading characters of the trailing `t.co` into the body.

    TWO emoji, deliberately: with one the bases diverge by a single character, that character is
    the space before the link, and `_body_text`'s closing `.strip()` removes it — so the naive
    slice and the correct one produce the same string and the test proves nothing. The second
    emoji is what pushes the divergence past the space and onto the link itself.
    """
    body = "Look at this 🎉🎉 https://t.co/abcdefghij"
    # Each 🎉 is one codepoint and TWO UTF-16 units: "Look at this " is 13 either way, so the body
    # ends at 17 in code units and at 15 by codepoint, and slicing by the wrong one takes " h".
    conversation = _parse(
        monkeypatch,
        payload=_tweet(text=body, media=[_photo()], extra={"display_text_range": [0, 17]}),
    )
    target = conversation.target

    assert target is not None
    assert target.text == "Look at this 🎉🎉"


def test_a_link_in_the_body_is_expanded_out_of_its_t_co_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every URL in `text` is a `t.co`, which tells a reader and the answer model nothing."""
    conversation = _parse(
        monkeypatch,
        payload=_tweet(
            text="Read this https://t.co/vBYe2L9FN8 now",
            extra={
                "entities": {
                    "urls": [
                        {
                            "url": "https://t.co/vBYe2L9FN8",
                            "expanded_url": "https://example.com/article",
                        }
                    ]
                }
            },
        ),
    )
    target = conversation.target

    assert target is not None
    assert target.text == "Read this https://example.com/article now"


def test_a_reply_carries_the_post_above_it_as_the_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ancestor arrives inside the same response, so one costs nothing and two cost a request."""
    conversation = _parse(monkeypatch, payload=_tweet(extra={"parent": _tweet(text="Feel good.")}))

    assert len(conversation.chain) == 2
    assert conversation.chain[0].text == "Feel good."
    assert conversation.target is not None
    assert conversation.target.text == "Play good."


def test_an_embedded_post_reports_its_replies_under_the_other_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reply figure changes key by POSITION: `conversation_count` on the post asked for,
    `reply_count` on an embedded parent or quote. Reading one renders the other as having none.
    """
    conversation = _parse(
        monkeypatch,
        payload=_tweet(
            extra={
                "quoted_tweet": _tweet(
                    text="quoted", extra={"conversation_count": 0, "reply_count": 729}
                )
            }
        ),
    )
    target = conversation.target

    assert target is not None
    assert target.comment_count == 2
    assert target.quoted is not None
    assert target.quoted.comment_count == 729


def test_a_quoted_post_is_carried_one_level_and_no_deeper(monkeypatch: pytest.MonkeyPatch) -> None:
    """46 sampled quotes carried no nested quote; the bound holds the payload to that anyway."""
    conversation = _parse(
        monkeypatch,
        payload=_tweet(
            extra={
                "quoted_tweet": _tweet(text="quoted", extra={"quoted_tweet": _tweet(text="deep")})
            }
        ),
    )
    target = conversation.target

    assert target is not None
    assert target.quoted is not None
    assert target.quoted.text == "quoted"
    assert target.quoted.quoted is None


def test_a_truncated_post_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """The body is cut at ~275 characters and the rest sits behind an id the endpoint will not
    resolve. Nothing in the text marks it, so a card that does not say so passes a fragment off
    as the post.
    """
    conversation = _parse(
        monkeypatch, payload=_tweet(extra={"note_tweet": {"id": "Tm90ZVR3ZWV0UmVzdWx0czox"}})
    )
    target = conversation.target

    assert target is not None
    assert target.is_truncated


def test_an_ordinary_post_is_not_marked_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flag rides on a key's presence, so its absence has to mean what it says."""
    conversation = _parse(monkeypatch, payload=_tweet())
    target = conversation.target

    assert target is not None
    assert not target.is_truncated


@pytest.mark.parametrize(
    "reason",
    [
        "This Post was deleted by the Post author. Learn more",
        # Twitter's own curly apostrophe, kept because this is a real response rather than prose.
        "You’re unable to view this Post because this account owner limits who can view it.",  # noqa: RUF001
        "This Post is from a suspended account. Learn more",
        "此貼文來自遭停權的帳戶。了解更多",
        None,
    ],
)
def test_a_refused_post_reads_as_unreadable_whatever_it_says(
    monkeypatch: pytest.MonkeyPatch, reason: str | None
) -> None:
    """A refusal is HTTP 200 with a well-formed body, so the type is the only signal.

    The wording is localized by `lang` and one variant carries none at all, which is why nothing
    matches on it. The zh-TW entry is a real response, not a translation made up here.
    """
    conversation = _parse(monkeypatch, payload=_tombstone(text=reason))

    assert conversation.target is None
    assert conversation.posts == []
    assert conversation.comments == []


def test_a_conversation_never_carries_replies(monkeypatch: pytest.MonkeyPatch) -> None:
    """The endpoint serves no reply content at all, so the field is carried and stays empty.

    Only the count survives, and it rides on the target rather than on a branch list nothing can
    fill — which is what stops a caller presenting `comments` as the thread.
    """
    conversation = _parse(monkeypatch, payload=_tweet(extra={"conversation_count": 540}))

    assert conversation.reply_branches == []
    assert conversation.comments == []
    assert conversation.selected_comment is None
    assert conversation.selected_comment_id == ""
    assert conversation.target is not None
    assert conversation.target.comment_count == 540


def test_a_url_naming_no_post_costs_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """A profile URL that reached the parser anyway must not spend a request to say so."""
    calls: list[str] = []

    def fake_fetch(self: TwitterDownloader, *, status_id: str) -> dict[str, Any]:
        del self
        calls.append(status_id)
        return _tweet()

    monkeypatch.setattr(TwitterDownloader, "_fetch_tweet", fake_fetch)
    conversation = TwitterDownloader().parse_metadata(url="https://x.com/Dbacks")

    assert conversation.target is None
    assert calls == []


def test_a_post_with_no_author_block_still_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """The payload is not ours, and a missing author must cost the byline rather than the post."""
    payload = _tweet()
    del payload["user"]

    conversation = _parse(monkeypatch, payload=payload)
    target = conversation.target

    assert target is not None
    assert target.author_name == ""
    assert target.text == "Play good."
    assert target.is_readable


class _FakeResponse:
    """The slice of `requests.Response` `_fetch_tweet` touches."""

    def __init__(
        self, *, payload: dict[str, Any] | Exception | None = None, status_code: int = 200
    ) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        """Raises the `HTTPError` carrying a response, exactly as `requests` does."""
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)

    def json(self) -> dict[str, Any]:
        """The decoded body, or the decoder's own error when the body is not JSON."""
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload or {}


def _install_get(
    monkeypatch: pytest.MonkeyPatch, *, response: _FakeResponse
) -> list[dict[str, Any]]:
    """Replaces the one `requests.get` the module makes; returns the captured call kwargs.

    A captured call is the request kwargs bag, so `Any` is what lets an assertion reach into a
    nested value such as `calls[0]["headers"]["User-Agent"]`.
    """
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        calls.append({"url": url, **kwargs})
        return response

    monkeypatch.setattr(twitter_module.requests, "get", fake_get)
    return calls


def test_the_request_carries_what_the_endpoint_refuses_to_answer_without(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token of any value and any User-Agent, neither of which fails loudly when dropped.

    Every other test here replaces `_fetch_tweet` wholesale, so without this one the request shape
    is unguarded: dropping the token answers HTTP 200 with a body of `{}`, which reads downstream
    as a post that cannot be read, and dropping the User-Agent answers 400 with an empty body.
    Both are total, silent outages of the feature.
    """
    calls = _install_get(monkeypatch, response=_FakeResponse(payload=_tweet()))

    TwitterDownloader().parse_metadata(url=_URL)

    assert calls[0]["url"] == "https://cdn.syndication.twimg.com/tweet-result"
    assert calls[0]["params"]["id"] == _STATUS_ID
    assert calls[0]["params"]["token"]
    assert calls[0]["headers"]["User-Agent"]
    assert calls[0]["timeout"] == TWITTER_PAGE_TIMEOUT_SECONDS


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(429, LinkRetryableError), (503, LinkRetryableError), (404, LinkUnavailableError)],
)
def test_a_failed_request_raises_the_class_that_decides_the_reaction(
    monkeypatch: pytest.MonkeyPatch, status_code: int, expected: type[Exception]
) -> None:
    """The expansion's mark is read off this class, so a bare RuntimeError would paint ❌.

    Telling a reader a working link is dead is the worst outcome the feature has, and a 429 is
    exactly when it would: the post is fine and the same link works in a minute.
    """
    _install_get(monkeypatch, response=_FakeResponse(status_code=status_code))

    with pytest.raises(expected):
        TwitterDownloader().parse_metadata(url=_URL)


def test_an_id_too_large_to_be_a_post_is_refused_before_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the snowflake range the endpoint answers 400, which classifies as nothing at all.

    `link_fetch_error` names 429 / 5xx and 404 / 410 and hands everything else back bare, so a
    mistyped 21-digit id would reach the channel as the cross that means the bot broke, plus an
    error-level log with a traceback. The id cannot be a post, so no request is spent on it.
    """
    calls = _install_get(monkeypatch, response=_FakeResponse(payload=_tweet()))

    conversation = TwitterDownloader().parse_metadata(
        url="https://x.com/Dbacks/status/9999999999999999999999"
    )

    assert conversation.target is None
    assert calls == []


def test_a_body_that_is_not_json_is_not_reported_as_a_failed_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`requests.JSONDecodeError` is a `RequestException`, so the decode sits outside that clause.

    Inside it, an interstitial served behind a 200 would be classified as a failed FETCH of a
    request that landed fine, and the operator would go looking at the network.
    """
    _install_get(
        monkeypatch,
        response=_FakeResponse(payload=requests.exceptions.JSONDecodeError("no", "<html>", 0)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        TwitterDownloader().parse_metadata(url=_URL)

    assert not isinstance(excinfo.value, LinkReadError)
    assert "non-JSON" in str(excinfo.value)
