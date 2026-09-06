"""Tests for Instagram URL parsing and post extraction.

Every test replaces `InstagramDownloader._fetch_page`, the one seam that touches the network,
the same way the Threads and Facebook tests do. The fixture page mirrors the real one's shape
where it matters: the wanted post sits among the author's OTHER posts, which carry the same
keys and would be picked up by anything positional.
"""

import json
from typing import Any

import pytest

from discordbot.utils.instagram import (
    FetchedPage,
    InstagramURL,
    InstagramDownloader,
    is_instagram_post_url,
)

_CODE = "Dc5eNjYkoZE"
_COMMENT_ID = "17946527169275440"
_URL = f"https://www.instagram.com/p/{_CODE}/"


def _image(*, name: str) -> dict[str, Any]:
    """One carousel child, with its candidates widest-first like the real payload."""
    return {
        "code": f"child{name}",
        "media_type": 1,
        "image_versions2": {
            "candidates": [
                {"url": f"https://instagram.example/{name}-original.jpg"},
                {"url": f"https://instagram.example/{name}-p720x720.jpg"},
            ]
        },
    }


def _media(
    *, code: str = _CODE, caption: str = "post body", children: int = 2, video_url: str = ""
) -> dict[str, Any]:
    """The post node, in the shape the logged-out page serialises."""
    node: dict[str, Any] = {
        "code": code,
        "pk": "3979344618500294212",
        "media_type": 8,
        "product_type": "carousel_container",
        "taken_at": 1788594866,
        "like_count": 8855,
        "comment_count": 11,
        "caption": {"text": caption, "pk": "1", "created_at": 1788594866},
        "user": {
            "pk": "5553326911",
            "username": "c_cylynn",
            "full_name": "晏凌",
            "profile_pic_url": "https://instagram.example/avatar.jpg",
        },
    }
    if video_url:
        node["media_type"] = 2
        node["video_versions"] = [{"url": video_url}]
        node["image_versions2"] = {"candidates": [{"url": "https://instagram.example/thumb.jpg"}]}
    else:
        node["carousel_media"] = [_image(name=str(index)) for index in range(children)]
    return node


def _other_post(*, code: str, caption: str) -> dict[str, Any]:
    """A node from the "more posts by this author" rail: same keys, no media list."""
    return {
        "code": code,
        "pk": f"pk-{code}",
        "media_type": 8,
        "carousel_media_count": 3,
        "caption": {"text": caption},
        "user": {"username": "c_cylynn"},
    }


def _comment(*, pk: str, text: str, author: str = "someone", parent: str = "") -> dict[str, Any]:
    """One comment node, recognised by carrying both a like count and a body."""
    node: dict[str, Any] = {
        "pk": pk,
        "text": text,
        "created_at": 1788595447,
        "comment_like_count": 0,
        "child_comment_count": 0,
        "user": {"username": author, "profile_pic_url": "https://instagram.example/c.jpg"},
    }
    if parent:
        node["parent_comment_id"] = parent
    return node


def _page(
    *,
    media: dict[str, Any] | None = None,
    comments: list[dict[str, Any]] | None = None,
    others: list[dict[str, Any]] | None = None,
) -> str:
    """Wraps the payloads into the script blocks the parser scans, plus one it must skip."""
    blocks = [
        json.dumps(
            obj={
                "require": [
                    {
                        "__bbox": {
                            "result": {
                                "data": {"xdt_media": media if media is not None else _media()}
                            }
                        }
                    }
                ]
            }
        )
    ]
    if others:
        blocks.append(json.dumps(obj={"data": {"more_posts": {"edges": others}}}))
    if comments:
        blocks.append(json.dumps(obj={"data": {"comments": comments}}))
    scripts = "".join(
        f'<script type="application/json" data-sjs>{block}</script>' for block in blocks
    )
    return f'<html><script type="application/json">{{"broken"</script>{scripts}</html>'


def _downloader(
    monkeypatch: pytest.MonkeyPatch, *, html: str, final_url: str = _URL
) -> InstagramDownloader:
    """A downloader whose only network call is replaced with canned HTML."""

    def fake_fetch_page(self: InstagramDownloader, *, url: str) -> FetchedPage:
        """Serves the canned page regardless of the URL asked for."""
        del self, url
        return FetchedPage(html=html, final_url=final_url)

    monkeypatch.setattr(target=InstagramDownloader, name="_fetch_page", value=fake_fetch_page)
    return InstagramDownloader()


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.instagram.com/p/{_CODE}",
        f"https://www.instagram.com/p/{_CODE}/",
        f"https://www.instagram.com/reel/{_CODE}/",
        f"https://www.instagram.com/tv/{_CODE}/",
        f"https://www.instagram.com/c_cylynn/p/{_CODE}/",
        f"https://instagram.com/p/{_CODE}/",
    ],
)
def test_every_accepted_spelling_names_the_same_post(url: str) -> None:
    """Instagram serves one post under several paths, and all of them are pasted in practice."""
    assert InstagramURL(raw_url=url).shortcode == _CODE
    assert is_instagram_post_url(url=url)


@pytest.mark.parametrize(
    "url",
    [
        "https://www.instagram.com/",
        "https://www.instagram.com/c_cylynn/",
        "https://www.instagram.com/explore/tags/cat/",
        "https://www.instagram.com/accounts/login/",
        # The sound page a reel's audio links to. It sits under `/reels/` like a post does, so
        # without the exception it parses with `audio` as the shortcode.
        "https://www.instagram.com/reels/audio/1234567890123456/",
    ],
)
def test_a_url_that_names_no_post_is_refused(url: str) -> None:
    """A profile link would otherwise cost a full page fetch to learn there is no post."""
    assert not is_instagram_post_url(url=url)


def test_the_comment_permalink_is_parsed_but_never_fetched() -> None:
    """That URL answers with a page carrying no post payload, so the fetch must not use it."""
    parsed = InstagramURL(raw_url=f"{_URL}c/{_COMMENT_ID}/?img_index=1")

    assert parsed.shortcode == _CODE
    assert parsed.comment_id == _COMMENT_ID
    assert parsed.clean_url == _URL
    assert "/c/" not in parsed.clean_url


def test_clean_url_drops_the_share_token_and_the_image_index() -> None:
    """`stkn` is minted per share, so echoing it names whoever passed the link on."""
    parsed = InstagramURL(
        raw_url=f"{_URL}?utm_source=ig_web_copy_link&stkn=NTc4MTIwNjQ2YQ==&img_index=1"
    )

    assert parsed.clean_url == _URL
    assert "stkn" not in parsed.clean_url
    assert "img_index" not in parsed.clean_url


def test_a_reel_keeps_its_own_path() -> None:
    """`/reel/` and `/p/` both resolve, and keeping the pasted one cannot be wrong."""
    assert InstagramURL(raw_url=f"https://www.instagram.com/reels/{_CODE}/").clean_url == (
        f"https://www.instagram.com/reel/{_CODE}/"
    )


def test_a_post_is_read_with_its_caption_images_and_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point: the caption, every carousel image, and the counters."""
    downloader = _downloader(monkeypatch, html=_page())

    conversation = downloader.parse_metadata(url=_URL)

    post = conversation.target
    assert post is not None
    assert post.is_readable
    assert post.text == "post body"
    assert post.author_name == "c_cylynn"
    assert post.author_full_name == "晏凌"
    assert post.image_urls == [
        "https://instagram.example/0-original.jpg",
        "https://instagram.example/1-original.jpg",
    ]
    assert post.like_count == 8855
    assert post.comment_count == 11
    assert post.taken_at is not None


def test_the_first_candidate_is_the_original(monkeypatch: pytest.MonkeyPatch) -> None:
    """The candidates are widest-first and carry no dimensions, so position is the only signal."""
    downloader = _downloader(monkeypatch, html=_page())

    conversation = downloader.parse_metadata(url=_URL)

    post = conversation.target
    assert post is not None
    assert all("original" in url for url in post.image_urls)


def test_the_chain_is_the_post_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instagram has no ancestor posts, but the shape matches Threads so callers agree."""
    downloader = _downloader(monkeypatch, html=_page())

    conversation = downloader.parse_metadata(url=_URL)

    assert len(conversation.chain) == 1
    assert conversation.target is conversation.chain[0]


def test_another_post_by_the_same_author_is_not_mistaken_for_this_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The page carries a "more posts" rail whose nodes look just like the target's."""
    others = [
        _other_post(code="DVigY57EpeH", caption="somebody else's caption"),
        _other_post(code="DcawIpFEj46", caption="another one"),
    ]
    downloader = _downloader(monkeypatch, html=_page(others=others))

    conversation = downloader.parse_metadata(url=_URL)

    post = conversation.target
    assert post is not None
    assert post.text == "post body"


def test_the_comments_come_back_as_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shaped like Threads' reply branches, so one renderer walks either platform."""
    comments = [
        _comment(pk="111", text="first", author="a"),
        _comment(pk="222", text="a reply to the first", author="b", parent="111"),
        _comment(pk="333", text="second top-level", author="c"),
    ]
    downloader = _downloader(monkeypatch, html=_page(comments=comments))

    conversation = downloader.parse_metadata(url=_URL)

    assert [[c.text for c in branch] for branch in conversation.reply_branches] == [
        ["first", "a reply to the first"],
        ["second top-level"],
    ]
    assert [c.text for c in conversation.comments] == [
        "first",
        "a reply to the first",
        "second top-level",
    ]


def test_a_comment_permalink_selects_that_comment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The id is matched exactly, so the highlighted comment is never a guess."""
    comments = [
        _comment(pk="999", text="another", author="a"),
        _comment(pk=_COMMENT_ID, text="the one linked", author="xiao_pang0704"),
    ]
    downloader = _downloader(monkeypatch, html=_page(comments=comments))

    conversation = downloader.parse_metadata(url=f"{_URL}c/{_COMMENT_ID}/")

    selected = conversation.selected_comment
    assert selected is not None
    assert selected.text == "the one linked"
    assert selected.author_name == "xiao_pang0704"


def test_a_comment_that_is_not_on_the_page_selects_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A very old comment on a viral post falls outside what the page serves."""
    downloader = _downloader(
        monkeypatch, html=_page(comments=[_comment(pk="999", text="another")])
    )

    conversation = downloader.parse_metadata(url=f"{_URL}c/{_COMMENT_ID}/")

    assert conversation.target is not None
    assert conversation.selected_comment is None


def test_a_comment_serialised_twice_is_read_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The page repeats a comment across blocks, and a duplicate would render twice."""
    comment = _comment(pk="111", text="only once", author="a")
    downloader = _downloader(monkeypatch, html=_page(comments=[comment, dict(comment)]))

    conversation = downloader.parse_metadata(url=_URL)

    assert [c.text for c in conversation.comments] == ["only once"]


def test_a_video_post_yields_its_playable_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Reel carries a playable url, unlike Facebook's logged-out video node."""
    downloader = _downloader(
        monkeypatch, html=_page(media=_media(video_url="https://instagram.example/clip.mp4"))
    )

    conversation = downloader.parse_metadata(url=_URL)

    post = conversation.target
    assert post is not None
    assert post.video_urls == ["https://instagram.example/clip.mp4"]
    assert post.is_readable


def test_a_login_wall_reads_as_an_empty_conversation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A private account redirects to login, which is a normal outcome rather than a failure."""
    downloader = _downloader(
        monkeypatch,
        html="<html>login</html>",
        final_url="https://www.instagram.com/accounts/login/?next=x",
    )

    conversation = downloader.parse_metadata(url=_URL)

    assert conversation.chain == []
    assert conversation.target is None


def test_a_page_with_no_post_payload_reads_as_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deleted post answers 200 with a page carrying no media node at all."""
    downloader = _downloader(monkeypatch, html="<html><body>nothing here</body></html>")

    conversation = downloader.parse_metadata(url=_URL)

    assert conversation.target is None


def test_a_url_naming_no_post_is_never_fetched(monkeypatch: pytest.MonkeyPatch) -> None:
    """A profile link must cost no request at all, not a request that comes back empty."""
    fetched: list[str] = []

    def fake_fetch_page(self: InstagramDownloader, *, url: str) -> FetchedPage:
        """Records that it was called, which this test asserts never happens."""
        del self
        fetched.append(url)
        return FetchedPage(html=_page(), final_url=url)

    monkeypatch.setattr(target=InstagramDownloader, name="_fetch_page", value=fake_fetch_page)

    conversation = InstagramDownloader().parse_metadata(url="https://www.instagram.com/c_cylynn/")

    assert fetched == []
    assert conversation.target is None
