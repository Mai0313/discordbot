"""Tests for Facebook URL parsing and post extraction.

Every test replaces `FacebookDownloader._fetch_page`, the one seam that touches the network,
the same way `tests/test_threads.py` does. The fixture HTML mirrors the real page's shape
closely enough to exercise the walk: the story is buried under a route-dependent key, repeated,
and surrounded by the module-loader blocks the real page is mostly made of.
"""

import json
import base64
from typing import Any

import pytest

from discordbot.utils.facebook import (
    FacebookURL,
    FetchedPage,
    FacebookOutput,
    FacebookDownloader,
    is_facebook_post_url,
)

_POST_ID = "1730774811333135"
_GROUP_ID = "1176671326743489"
_COMMENT_ID = "1730777104666239"
_PERMALINK = f"https://www.facebook.com/groups/{_GROUP_ID}/posts/{_POST_ID}/"
_SHARE_URL = "https://www.facebook.com/share/p/1MQuL1qHQ4/"


def _story(
    *,
    post_id: str = _POST_ID,
    text: str = "post body",
    images: tuple[str, ...] = ("https://scontent.example/a.jpg", "https://scontent.example/b.jpg"),
    video_permalink: str = "",
) -> dict[str, Any]:
    """One story node shaped like the real payload's."""
    if video_permalink:
        nodes = [{"media": {"__typename": "Video", "permalink_url": video_permalink}}]
    else:
        nodes = [
            {"media": {"viewer_image": {"uri": uri, "width": 1536, "height": 2048}}}
            for uri in images
        ]
    return {
        "post_id": post_id,
        "creation_time": 1788598024,
        "permalink_url": f"https://www.facebook.com/groups/{_GROUP_ID}/posts/{post_id}/",
        "attachments": [{"styles": {"attachment": {"all_subattachments": {"nodes": nodes}}}}],
        "feedback": {
            "i18n_reaction_count": "1,017",
            "reaction_count": 1017,
            "share_count": {"count": 37, "is_empty": False},
            "total_comment_count": 40,
        },
        "comet_sections": {
            "content": {
                "story": {
                    "actors": [
                        {
                            "name": "Somebody",
                            "profile_picture": {"uri": "https://scontent.example/avatar.jpg"},
                        }
                    ],
                    "comet_sections": {
                        "message_container": {"story": {"message": {"text": text}}}
                    },
                }
            }
        },
    }


def _encoded_comment_id(*, post_id: str, comment_id: str) -> str:
    """A comment node's own id, the way the page serves it: unpadded base64."""
    return base64.b64encode(f"comment:{post_id}_{comment_id}".encode()).decode().rstrip("=")


def _comment(
    *,
    comment_id: str,
    text: str,
    author: str = "Someone",
    post_id: str = _POST_ID,
    parent_id: str = "",
) -> dict[str, Any]:
    """One comment node, in the fuller of the two shapes the page serialises.

    It carries both ids the real node does: `legacy_fbid` for its own, and the encoded `id` that
    also names the post it hangs off. `comment_direct_parent` is present on every real comment
    and null on a top-level one, which is what the parent fixture mirrors.
    """
    return {
        "__typename": "Comment",
        "legacy_fbid": comment_id,
        "id": _encoded_comment_id(post_id=post_id, comment_id=comment_id),
        "body": {"text": text},
        "author": {"name": author, "profile_picture": {"uri": "https://scontent.example/c.jpg"}},
        "created_time": 1788600000,
        "depth": 1 if parent_id else 0,
        "comment_direct_parent": {"legacy_fbid": parent_id} if parent_id else None,
    }


def _page(
    *,
    stories: list[dict[str, Any]] | None = None,
    comments: list[dict[str, Any]] | None = None,
    groups: list[dict[str, Any]] | None = None,
) -> str:
    """Wraps payloads into the script blocks the parser scans, plus one it must skip."""
    story_nodes = _story() if stories is None else stories
    payload = {
        "require": [
            [
                "ScheduledServerJS",
                "handle",
                None,
                [
                    {
                        "__bbox": {
                            "result": {
                                "data": {
                                    "node_v2": story_nodes[0]
                                    if isinstance(story_nodes, list)
                                    else story_nodes
                                }
                            }
                        }
                    }
                ],
            ]
        ]
    }
    blocks = [json.dumps(obj=payload)]
    if isinstance(story_nodes, list) and len(story_nodes) > 1:
        for extra in story_nodes[1:]:
            blocks.append(json.dumps(obj={"require": [{"__bbox": {"data": {"node": extra}}}]}))
    if comments:
        blocks.append(json.dumps(obj={"comment_rendering_instance": {"comments": comments}}))
    if groups:
        blocks.append(json.dumps(obj={"data": {"groups": groups}}))
    scripts = "".join(
        f'<script type="application/json" data-sjs>{block}</script>' for block in blocks
    )
    # A block that does not parse, which the walk must skip rather than fail on.
    return f'<html><script type="application/json">{{"broken"</script>{scripts}</html>'


def _downloader(
    monkeypatch: pytest.MonkeyPatch, *, html: str, final_url: str = _PERMALINK
) -> FacebookDownloader:
    """A downloader whose only network call is replaced with canned HTML."""

    def fake_fetch_page(self: FacebookDownloader, *, url: str) -> FetchedPage:
        """Serves the canned page regardless of the URL asked for."""
        del self, url
        return FetchedPage(html=html, final_url=final_url)

    monkeypatch.setattr(target=FacebookDownloader, name="_fetch_page", value=fake_fetch_page)
    return FacebookDownloader()


def _parse(downloader: FacebookDownloader, url: str) -> FacebookOutput:
    """The post out of a parsed conversation, or an empty one when it carried none."""
    return downloader.parse_metadata(url=url).target or FacebookOutput()


def test_a_group_post_url_names_its_post_and_group() -> None:
    """The canonical group form carries both ids in its path."""
    parsed = FacebookURL(raw_url=_PERMALINK)

    assert parsed.post_id == _POST_ID
    assert parsed.group_id == _GROUP_ID
    assert parsed.comment_id == ""


def test_a_permalink_form_names_its_post() -> None:
    """The `/groups/<id>/permalink/<id>` spelling is the same post as `/posts/<id>`."""
    parsed = FacebookURL(
        raw_url=f"https://www.facebook.com/groups/{_GROUP_ID}/permalink/{_POST_ID}/"
    )

    assert parsed.post_id == _POST_ID


def test_a_story_fbid_query_names_its_post() -> None:
    """`permalink.php` carries the post id in the query rather than the path."""
    parsed = FacebookURL(
        raw_url=f"https://www.facebook.com/permalink.php?story_fbid={_POST_ID}&id=100077759593577"
    )

    assert parsed.post_id == _POST_ID


def test_a_group_feed_url_names_the_post_it_highlights() -> None:
    """A group feed link with `multi_permalinks` points at one post, so it is readable."""
    url = f"https://www.facebook.com/groups/{_GROUP_ID}?multi_permalinks={_POST_ID}"
    parsed = FacebookURL(raw_url=url)

    assert parsed.post_id == _POST_ID
    assert parsed.group_id == _GROUP_ID
    assert is_facebook_post_url(url=url)


def test_a_share_link_names_no_post_until_it_redirects() -> None:
    """The share form's own code is unrelated to the post's, so only the redirect names it."""
    parsed = FacebookURL(raw_url=_SHARE_URL)

    assert parsed.post_id == ""
    assert parsed.is_share_link
    assert is_facebook_post_url(url=_SHARE_URL)


def test_clean_url_drops_the_tokens_that_name_whoever_shared_it() -> None:
    """`rdid` and `share_url` are minted per share, so echoing them names the sharer."""
    parsed = FacebookURL(
        raw_url=f"{_PERMALINK}?rdid=OznEaPh5P9lgFfNX&share_url=https%3A%2F%2Ffb.me%2Fx"
    )

    assert "rdid" not in parsed.clean_url
    assert "share_url" not in parsed.clean_url
    assert parsed.clean_url == _PERMALINK.rstrip("/")


def test_clean_url_keeps_the_comment_the_url_singles_out() -> None:
    """The comment id is content, not provenance, so it survives the strip."""
    parsed = FacebookURL(raw_url=f"{_PERMALINK}?comment_id={_COMMENT_ID}&rdid=abc")

    assert parsed.comment_id == _COMMENT_ID
    assert f"comment_id={_COMMENT_ID}" in parsed.clean_url
    assert "rdid" not in parsed.clean_url


@pytest.mark.parametrize(
    "url",
    [
        "https://www.facebook.com/NASA",
        f"https://www.facebook.com/groups/{_GROUP_ID}",
        "https://www.facebook.com/",
        "https://www.facebook.com/marketplace/item/123456/",
    ],
)
def test_a_url_that_names_no_post_is_refused(url: str) -> None:
    """A profile or group home would otherwise cost a full page fetch to learn nothing."""
    assert not is_facebook_post_url(url=url)


def test_a_post_is_read_with_its_text_images_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point: the full body, not the ~190 characters the Open Graph tag carries."""
    downloader = _downloader(monkeypatch, html=_page())

    post = _parse(downloader, _PERMALINK)

    assert post.is_readable
    assert post.text == "post body"
    assert post.author_name == "Somebody"
    assert post.author_icon_url == "https://scontent.example/avatar.jpg"
    assert post.image_urls == ["https://scontent.example/a.jpg", "https://scontent.example/b.jpg"]
    assert post.like_count == 1017
    assert post.share_count == 37
    assert post.comment_count == 40
    assert post.taken_at is not None


def test_the_permalink_is_preferred_over_the_pasted_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """What gets published back into the channel must never carry the sharer's tokens."""
    downloader = _downloader(monkeypatch, html=_page())

    post = _parse(downloader, f"{_SHARE_URL}?rdid=abc")

    assert post.url == _PERMALINK
    assert "rdid" not in post.url


def test_a_comment_id_url_selects_that_comment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The id is matched exactly, so the highlighted comment is never a guess."""
    comments = [
        _comment(comment_id="999", text="another comment", author="Bystander"),
        _comment(comment_id=_COMMENT_ID, text="the one linked", author="Commenter"),
    ]
    downloader = _downloader(monkeypatch, html=_page(comments=comments))

    conversation = downloader.parse_metadata(url=f"{_PERMALINK}?comment_id={_COMMENT_ID}")

    selected = conversation.selected_comment
    assert selected is not None
    assert selected.text == "the one linked"
    assert selected.author_name == "Commenter"
    assert selected.taken_at is not None


def test_a_comment_the_page_did_not_preload_selects_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a handful of comments are preloaded, so an old one resolves to no comment at all."""
    downloader = _downloader(
        monkeypatch, html=_page(comments=[_comment(comment_id="999", text="another")])
    )

    conversation = downloader.parse_metadata(url=f"{_PERMALINK}?comment_id={_COMMENT_ID}")

    assert conversation.target is not None
    assert conversation.selected_comment is None


def test_a_comment_id_is_recovered_from_the_base64_node_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node without `legacy_fbid` still identifies itself through its encoded id."""
    node = _comment(comment_id=_COMMENT_ID, text="decoded me")
    del node["legacy_fbid"]
    downloader = _downloader(monkeypatch, html=_page(comments=[node]))

    conversation = downloader.parse_metadata(url=f"{_PERMALINK}?comment_id={_COMMENT_ID}")

    selected = conversation.selected_comment
    assert selected is not None
    assert selected.text == "decoded me"


def test_a_comment_serialised_twice_is_read_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The page repeats each comment as a stub for its reply expander."""
    stub = {"__typename": "Comment", "legacy_fbid": _COMMENT_ID, "body": {"text": ""}}
    full = _comment(comment_id=_COMMENT_ID, text="the real body")
    downloader = _downloader(monkeypatch, html=_page(comments=[stub, full, dict(full)]))

    conversation = downloader.parse_metadata(url=_PERMALINK)

    assert [comment.text for comment in conversation.comments] == ["the real body"]


def test_a_reply_threads_behind_the_comment_it_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """The page serialises the top-level comments first and the replies after all of them.

    So the branch a reply belongs to is the one its `comment_direct_parent` names, never the one
    that happens to be open when the walk reaches it.
    """
    comments = [
        _comment(comment_id="11", text="top one"),
        _comment(comment_id="22", text="top two"),
        _comment(comment_id="111", text="reply to top one", parent_id="11"),
    ]
    downloader = _downloader(monkeypatch, html=_page(comments=comments))

    conversation = downloader.parse_metadata(url=_PERMALINK)

    assert [[comment.text for comment in branch] for branch in conversation.reply_branches] == [
        ["top one", "reply to top one"],
        ["top two"],
    ]


def test_a_reply_whose_parent_is_absent_opens_its_own_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a handful of comments are preloaded, so a reply can outlive the comment it answers."""
    comments = [
        _comment(comment_id="22", text="top two"),
        _comment(comment_id="111", text="reply to a comment nobody can see", parent_id="11"),
    ]
    downloader = _downloader(monkeypatch, html=_page(comments=comments))

    conversation = downloader.parse_metadata(url=_PERMALINK)

    assert [[comment.text for comment in branch] for branch in conversation.reply_branches] == [
        ["top two"],
        ["reply to a comment nobody can see"],
    ]


def test_a_neighbouring_posts_comments_are_not_attributed_to_this_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comments sit outside the story node, so the walk that finds them is page-wide.

    On a group feed that means walking past the other posts' comments, and each comment's own
    encoded id is what says which post it belongs to.
    """
    comments = [
        _comment(comment_id="11", text="on the linked post"),
        _comment(comment_id="22", text="on the post below it", post_id="999"),
    ]
    downloader = _downloader(
        monkeypatch, html=_page(stories=[_story(), _story(post_id="999")], comments=comments)
    )

    conversation = downloader.parse_metadata(url=_PERMALINK)

    assert [comment.text for comment in conversation.comments] == ["on the linked post"]


def test_a_group_feed_yields_the_post_the_url_named(monkeypatch: pytest.MonkeyPatch) -> None:
    """A feed serialises several posts, and only one of them is the one that was linked."""
    other = _story(post_id="999", text="somebody else's post")
    wanted = _story(post_id=_POST_ID, text="the linked post")
    downloader = _downloader(monkeypatch, html=_page(stories=[other, wanted]))

    post = _parse(
        downloader, f"https://www.facebook.com/groups/{_GROUP_ID}?multi_permalinks={_POST_ID}"
    )

    assert post.text == "the linked post"


def test_a_share_link_takes_its_post_id_from_where_it_landed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The share form names its post only through the redirect, exactly as Threads does."""
    other = _story(post_id="999", text="somebody else's post")
    wanted = _story(post_id=_POST_ID, text="the shared post")
    downloader = _downloader(monkeypatch, html=_page(stories=[other, wanted]))

    post = _parse(downloader, _SHARE_URL)

    assert post.text == "the shared post"


def test_a_login_wall_reads_as_an_unreadable_post(monkeypatch: pytest.MonkeyPatch) -> None:
    """A private post redirects to login, which is a normal outcome rather than a failure."""
    downloader = _downloader(
        monkeypatch,
        html="<html>login</html>",
        final_url="https://www.facebook.com/login.php?next=x",
    )

    post = _parse(downloader, _PERMALINK)

    assert not post.is_readable
    assert post.text == ""


def test_a_page_with_no_story_payload_reads_as_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deleted post answers 200 with a page carrying no story at all."""
    downloader = _downloader(monkeypatch, html="<html><body>nothing here</body></html>")

    post = _parse(downloader, _PERMALINK)

    assert not post.is_readable


def test_a_video_post_yields_a_permalink_and_no_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Logged out there is no `playable_url`, so a video is a link and never a download."""
    downloader = _downloader(
        monkeypatch,
        html=_page(stories=[_story(video_permalink="https://www.facebook.com/watch/?v=1")]),
    )

    post = _parse(downloader, _PERMALINK)

    assert post.video_urls == ["https://www.facebook.com/watch/?v=1"]
    assert post.image_urls == []
    assert post.is_readable


def test_the_group_name_comes_from_the_group_the_url_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A group page also serialises the groups it recommends beside the one being read."""
    groups = [
        {"__typename": "Group", "id": "111", "name": "A recommended group"},
        {"__typename": "Group", "id": _GROUP_ID, "name": "The real group"},
    ]
    downloader = _downloader(monkeypatch, html=_page(groups=groups))

    post = _parse(downloader, _PERMALINK)

    assert post.group_name == "The real group"


def test_a_page_post_carries_no_group_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A page post has no group, and must not borrow a recommendation's name."""
    groups = [{"__typename": "Group", "id": "111", "name": "A recommended group"}]
    downloader = _downloader(monkeypatch, html=_page(groups=groups))

    post = _parse(downloader, f"https://www.facebook.com/NASA/posts/{_POST_ID}")

    assert post.group_name == ""


def test_an_image_only_post_is_still_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A post with pictures and no words is legitimate, so it must not read as empty."""
    downloader = _downloader(monkeypatch, html=_page(stories=[_story(text="")]))

    post = _parse(downloader, _PERMALINK)

    assert post.is_readable
    assert post.text == ""
    assert len(post.image_urls) == 2


def test_clean_url_keeps_the_owner_id_a_permalink_needs() -> None:
    """`permalink.php` resolves nothing without the `id` naming the page that owns the post."""
    parsed = FacebookURL(
        raw_url=f"https://www.facebook.com/permalink.php?story_fbid={_POST_ID}&id=100077759593577"
    )

    assert "id=100077759593577" in parsed.clean_url
    assert parsed.post_id == _POST_ID


def test_a_profile_url_is_not_a_post_despite_carrying_an_id() -> None:
    """`id` is kept but never READ as a post id, or every profile link would look like a post."""
    assert not is_facebook_post_url(url="https://www.facebook.com/profile.php?id=100077759593577")
