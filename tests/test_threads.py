"""Tests for Threads URL parsing and media extraction."""

import json
import shutil
from pathlib import Path

import pytest

from discordbot.utils import threads as threads_module
from discordbot.utils.threads import (
    THREADS_URL_RE,
    Post,
    ThreadData,
    ThreadItem,
    ThreadsURL,
    ThreadsOutput,
    ThreadsDownloader,
)


@pytest.fixture
def downloader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ThreadsDownloader:
    """Provides a ThreadsDownloader that fakes media downloads."""

    def fake_download_media(self: ThreadsDownloader, url: str, filename: str) -> Path:
        """Writes fake media bytes and returns the expected output path."""
        assert url, "download url should not be empty"
        filepath = Path(self.output_folder) / filename
        filepath.write_bytes(data=b"fake media")
        return filepath

    monkeypatch.setattr(target=ThreadsDownloader, name="download_media", value=fake_download_media)
    return ThreadsDownloader(output_folder=str(tmp_path))


def test_find_post_with_parents_returns_chain_in_order() -> None:
    """Verifies that finding a post returns the correct chain of parent posts."""
    chain = ThreadData(
        thread_items=[
            ThreadItem(post=Post(code="ROOT")),
            ThreadItem(post=Post(code="LVL1")),
            ThreadItem(post=Post(code="LVL2")),
            ThreadItem(post=Post(code="TARGET")),
        ]
    )
    post, parents = chain.find_post_with_parents(post_code="TARGET")
    assert post is not None
    assert post.code == "TARGET"
    assert [p.code for p in parents] == ["ROOT", "LVL1", "LVL2"]


def test_find_post_with_parents_no_match() -> None:
    """Verifies that finding a non-existent post returns None and empty parents."""
    chain = ThreadData(thread_items=[ThreadItem(post=Post(code="A"))])
    post, parents = chain.find_post_with_parents(post_code="MISSING")
    assert post is None
    assert parents == []


def test_find_post_with_parents_root_has_no_parents() -> None:
    """Verifies that the root post of a thread has no parents."""
    chain = ThreadData(thread_items=[ThreadItem(post=Post(code="ROOT"))])
    post, parents = chain.find_post_with_parents(post_code="ROOT")
    assert post is not None
    assert parents == []


def test_threads_output_mutable_defaults_are_isolated(tmp_path: Path) -> None:
    """Threads output image and local video path defaults are isolated."""
    first = ThreadsOutput()
    second = ThreadsOutput()

    first.image_urls.append("https://cdn.example/image.jpg")
    first.video_paths.append(tmp_path / "clip.mp4")

    assert second.image_urls == []
    assert second.video_paths == []
    assert second.reply_to_username == ""


def _thread_post_payload(
    code: str, username: str, text: str, reply_to_username: str = "", video_url: str = ""
) -> dict[str, object]:
    """Returns a minimal Threads post payload with parser-relevant fields."""
    media: dict[str, object] = (
        {"video_versions": [{"url": video_url}]}
        if video_url
        else {"image_versions2": {"candidates": [{"url": f"https://cdn.example/{code}.jpg"}]}}
    )
    return {
        "post": {
            "code": code,
            "caption": {"text": text},
            "user": {
                "username": username,
                "profile_pic_url": f"https://cdn.example/{username}.jpg",
            },
            **media,
            "text_post_app_info": {
                "direct_reply_count": 1,
                "repost_count": 2,
                "quote_count": 3,
                "reshare_count": 4,
                "is_reply": bool(reply_to_username),
                "reply_to_author": (
                    {"username": reply_to_username, "profile_pic_url": ""}
                    if reply_to_username
                    else None
                ),
            },
            "like_count": 5,
            "taken_at": 1_735_689_600,
        }
    }


def _section_header(label: str) -> dict[str, object]:
    """Returns the node Threads inserts between a post's own replies and the filler below."""
    return {"header": label, "thread_items": [], "thread_type": "header", "id": "0"}


def _sjs_html(*threads: list[object] | dict[str, object]) -> str:
    """Builds deterministic Threads SJS HTML holding one node per thread, in page order.

    Mirrors how a real post page serialises itself: the chain ending at the target, every reply
    branch under it, and any section header between them are sibling `edges[].node` entries in
    one script block. A plain list becomes a thread node; a dict rides through as-is, which is
    how `_section_header` gets in.
    """
    payload = {
        "require": [
            {
                "__bbox": {
                    "result": {
                        "data": {
                            "data": {
                                "edges": [
                                    {
                                        "node": (
                                            entry
                                            if isinstance(entry, dict)
                                            else {"thread_type": "thread", "thread_items": entry}
                                        )
                                    }
                                    for entry in threads
                                ]
                            }
                        }
                    }
                }
            }
        ]
    }
    return (
        f'<html><script type="application/json" data-sjs>{json.dumps(obj=payload)}</script></html>'
    )


def _thread_html(post_code: str) -> str:
    """Builds deterministic Threads SJS HTML for parser tests."""
    return _sjs_html([
        _thread_post_payload(code="ROOT", username="root_author", text="Root post"),
        _thread_post_payload(
            code=post_code,
            username="target_author",
            text=f"Target post {post_code}",
            reply_to_username="root_author",
        ),
    ])


@pytest.mark.parametrize(
    "url",
    [
        "https://www.threads.com/@tpp_taiwan/post/DWWIhcQktP_",
        "https://www.threads.com/@wilson.sup/post/DWWMzCNkUUM",
        "https://www.threads.com/@yu0030722025/post/DWZmc8OkSx8",
        "https://www.threads.com/@show4653/post/DWYp35uGh4l",
        "https://www.threads.com/@cyj308/post/DVn6dqzjzQf",
        "https://www.threads.com/@tpp_taiwan/post/DWWIhcQktP_?hl=zh-tw",
        "https://www.threads.com/@tpp_taiwan/post/DWWIhcQktP_?xmt=AQF0p6UfiuvtlPVEKZ36kqN7JVKUzuMJUhGDOwfkJK6Rsw",
        "https://www.threads.com/@babe.0530/post/DXyk3qXGT6o",
    ],
)
def test_parse(downloader: ThreadsDownloader, url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that parsing valid Threads URLs returns expected post data."""
    threads_url = ThreadsURL(raw_url=url)
    fetched_urls: list[str] = []

    def fake_fetch_html(self: ThreadsDownloader, url: str) -> str:
        """Returns deterministic HTML for the requested Threads URL."""
        fetched_urls.append(url)
        return _thread_html(post_code=threads_url.post_code)

    monkeypatch.setattr(target=ThreadsDownloader, name="_fetch_html", value=fake_fetch_html)

    with downloader.parse(url=url) as conversation:
        assert conversation.chain, "should yield at least one post"
        target = conversation.target
        assert target is not None
        assert target.text or target.image_urls or target.video_urls, "post should have content"
        assert target.author_name, "author_name should not be empty"
        assert target.taken_at is not None, "taken_at should not be None"
    assert fetched_urls == [threads_url.clean_url]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "see https://www.threads.com/@a/post/ABC123 now",
            "https://www.threads.com/@a/post/ABC123",
        ),
        ("https://threads.net/@b/post/XYZ", "https://threads.net/@b/post/XYZ"),
        (
            "<@1> look https://www.threads.net/@c.d/post/Q_-1?hl=zh",
            "https://www.threads.net/@c.d/post/Q_-1?hl=zh",
        ),
        (
            "check this https://www.threads.com/@a/post/ABC123. it is wild",
            "https://www.threads.com/@a/post/ABC123",
        ),
        (
            "see https://www.threads.com/@a/post/ABC123, then reply",
            "https://www.threads.com/@a/post/ABC123",
        ),
        (
            "看這篇 https://www.threads.com/@a/post/ABC123。很扯",
            "https://www.threads.com/@a/post/ABC123",
        ),
        (
            "貼文【https://www.threads.com/@a/post/ABC123】super",
            "https://www.threads.com/@a/post/ABC123",
        ),
    ],
)
def test_threads_url_re_matches_post_links(text: str, expected: str) -> None:
    """The shared regex extracts a Threads post URL from surrounding text, query included."""
    match = THREADS_URL_RE.search(string=text)
    assert match is not None
    assert match.group(0) == expected


@pytest.mark.parametrize(
    "text",
    [
        "no url here at all",
        "https://www.threads.com/@profile_only",
        "https://www.instagram.com/p/ABC/",
    ],
)
def test_threads_url_re_rejects_non_posts(text: str) -> None:
    """Profile URLs, Instagram URLs, and plain text are not matched as posts."""
    assert THREADS_URL_RE.search(string=text) is None


def _thread_html_with_video(post_code: str) -> str:
    """Builds Threads SJS HTML whose single target post carries a video rendition."""
    return _sjs_html([
        _thread_post_payload(
            code=post_code,
            username="vid_author",
            text=f"Video post {post_code}",
            video_url=f"https://cdn.example/{post_code}.mp4",
        )
    ])


def test_parse_metadata_returns_chain_without_downloading(
    downloader: ThreadsDownloader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """parse_metadata yields the ordered chain with video URLs but no local files or downloads."""
    url = "https://www.threads.com/@root_author/post/TARGET"
    threads_url = ThreadsURL(raw_url=url)

    def fake_fetch_html(self: ThreadsDownloader, url: str) -> str:
        """Returns deterministic HTML with a video target for the requested URL."""
        del url
        return _thread_html_with_video(post_code=threads_url.post_code)

    def fail_download(self: ThreadsDownloader, url: str, filename: str) -> None:
        """Fails loudly if the metadata path ever tries to download media."""
        raise AssertionError("parse_metadata must not download media")

    monkeypatch.setattr(target=ThreadsDownloader, name="_fetch_html", value=fake_fetch_html)
    monkeypatch.setattr(target=ThreadsDownloader, name="download_media", value=fail_download)

    conversation = downloader.parse_metadata(url=url)

    assert conversation.chain, "should return at least the target post"
    target = conversation.target
    assert target is not None
    assert target.video_urls == [f"https://cdn.example/{threads_url.post_code}.mp4"]
    assert all(post.video_paths == [] for post in conversation.posts)


_REPLIES_TARGET_URL = "https://www.threads.com/@target_author/post/TARGET"


def _thread_html_with_replies() -> str:
    """Builds SJS HTML mixing the target's chain, its reply branches, and posts that are neither.

    A real post page serialises all of these into one block, so the parser has to tell them
    apart by who each thread's first post answers.
    """
    return _sjs_html(
        # The chain ending at the target.
        [
            _thread_post_payload(code="ROOT", username="root_author", text="Root post"),
            _thread_post_payload(
                code="TARGET",
                username="target_author",
                text="Target post",
                reply_to_username="root_author",
            ),
        ],
        # One branch: a direct comment plus the exchange nested under it.
        [
            _thread_post_payload(
                code="R1",
                username="commenter",
                text="First comment",
                reply_to_username="target_author",
            ),
            _thread_post_payload(
                code="R1A",
                username="target_author",
                text="Author answers",
                reply_to_username="commenter",
            ),
            _thread_post_payload(
                code="R1B",
                username="commenter",
                text="Commenter again",
                reply_to_username="target_author",
            ),
        ],
        # A second direct comment, with nothing nested under it.
        [
            _thread_post_payload(
                code="R2",
                username="other",
                text="Second comment",
                reply_to_username="target_author",
            )
        ],
        # A sibling: it answers the target's own parent, so it belongs to the root, not here.
        [
            _thread_post_payload(
                code="SIBLING",
                username="stranger",
                text="Answering the root instead",
                reply_to_username="root_author",
            )
        ],
        # A post answering nobody: a recommendation, or a reply whose parent was deleted.
        [_thread_post_payload(code="RECO", username="unrelated", text="Recommended post")],
    )


def _stub_html(monkeypatch: pytest.MonkeyPatch, html: str) -> None:
    """Serves one canned page for every fetch, so no test touches the network."""

    def fake_fetch_html(self: ThreadsDownloader, url: str) -> str:
        """Returns the canned HTML regardless of url."""
        del url
        return html

    monkeypatch.setattr(target=ThreadsDownloader, name="_fetch_html", value=fake_fetch_html)


def test_parse_metadata_collects_the_reply_branches(
    downloader: ThreadsDownloader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The comments under the target come back as branches, nesting preserved."""
    _stub_html(monkeypatch, _thread_html_with_replies())

    conversation = downloader.parse_metadata(url=_REPLIES_TARGET_URL)

    assert [post.text for post in conversation.chain] == ["Root post", "Target post"]
    assert [[post.author_name for post in branch] for branch in conversation.reply_branches] == [
        ["commenter", "target_author", "commenter"],
        ["other"],
    ]
    # A nested comment carries who it answers, which is what renders its place in the branch.
    nested = conversation.reply_branches[0][1]
    assert nested.reply_to_username == "commenter"
    assert nested.url == "https://www.threads.com/@target_author/post/R1A"


def test_parse_metadata_drops_threads_that_do_not_answer_the_target(
    downloader: ThreadsDownloader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sibling reply and a recommended post share the block but are not comments on the target."""
    _stub_html(monkeypatch, _thread_html_with_replies())

    conversation = downloader.parse_metadata(url=_REPLIES_TARGET_URL)

    codes = {post.url.rsplit("/", 1)[-1] for post in conversation.posts}
    assert "SIBLING" not in codes
    assert "RECO" not in codes


def test_a_section_header_ends_the_targets_own_replies(
    downloader: ThreadsDownloader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Threads pads a thin post with replies to the ROOT, and only the header separates them.

    On a post that is its author's own reply to their own post, the padding answers the same
    username the target does, so the author test alone would hand every one of those unrelated
    comments to the wrong post.
    """
    html = _sjs_html(
        [
            _thread_post_payload(code="ROOT", username="target_author", text="Root post"),
            _thread_post_payload(
                code="TARGET",
                username="target_author",
                text="The author's own follow-up",
                reply_to_username="target_author",
            ),
        ],
        [
            _thread_post_payload(
                code="MINE",
                username="commenter",
                text="A comment on the follow-up",
                reply_to_username="target_author",
            )
        ],
        _section_header(label="More replies to target_author"),
        [
            _thread_post_payload(
                code="FILLER",
                username="stranger",
                text="Actually a comment on the root",
                reply_to_username="target_author",
            )
        ],
    )
    _stub_html(monkeypatch, html)

    conversation = downloader.parse_metadata(url=_REPLIES_TARGET_URL)

    assert [[post.text for post in branch] for branch in conversation.reply_branches] == [
        ["A comment on the follow-up"]
    ]


def test_a_target_with_no_author_collects_no_comments(
    downloader: ThreadsDownloader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty author matches every post whose `reply_to_author` is missing, so it matches none."""
    html = _sjs_html(
        [{"post": {"code": "TARGET", "caption": {"text": "Author data missing"}}}],
        [_thread_post_payload(code="R1", username="commenter", text="Comment")],
    )
    _stub_html(monkeypatch, html)

    conversation = downloader.parse_metadata(url=_REPLIES_TARGET_URL)

    assert [post.text for post in conversation.chain] == ["Author data missing"]
    assert conversation.reply_branches == []


def test_a_page_without_the_post_yields_an_empty_conversation(
    downloader: ThreadsDownloader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page carrying no such post reads as empty, which is the callers' "unavailable" signal."""
    _stub_html(
        monkeypatch,
        _sjs_html([
            _thread_post_payload(code="SOMEONE_ELSE", username="other", text="Other post")
        ]),
    )

    conversation = downloader.parse_metadata(url=_REPLIES_TARGET_URL)

    assert conversation.chain == []
    assert conversation.target is None
    assert conversation.reply_branches == []


def test_a_malformed_thread_does_not_cost_the_target(
    downloader: ThreadsDownloader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One unparsable thread costs that thread only, never the target sharing its block."""
    html = _sjs_html(
        ["not a thread item at all"],
        [_thread_post_payload(code="TARGET", username="target_author", text="Target post")],
        [
            _thread_post_payload(
                code="R1", username="commenter", text="Comment", reply_to_username="target_author"
            )
        ],
    )
    _stub_html(monkeypatch, html)

    conversation = downloader.parse_metadata(url=_REPLIES_TARGET_URL)

    assert [post.text for post in conversation.chain] == ["Target post"]
    assert [[post.text for post in branch] for branch in conversation.reply_branches] == [
        ["Comment"]
    ]


def test_parse_downloads_the_target_video_and_no_others(
    downloader: ThreadsDownloader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A comment's clip is never written to disk; only the linked post's is."""
    html = _sjs_html(
        [
            _thread_post_payload(
                code="TARGET",
                username="target_author",
                text="Target post",
                video_url="https://cdn.example/target.mp4",
            )
        ],
        [
            _thread_post_payload(
                code="R1",
                username="commenter",
                text="Comment with a clip",
                reply_to_username="target_author",
                video_url="https://cdn.example/reply.mp4",
            )
        ],
    )
    _stub_html(monkeypatch, html)

    with downloader.parse(url=_REPLIES_TARGET_URL) as conversation:
        target = conversation.target
        assert target is not None
        assert len(target.video_paths) == 1
        assert target.video_paths[0].exists()
        reply = conversation.reply_branches[0][0]
        assert reply.video_urls == ["https://cdn.example/reply.mp4"]
        assert reply.video_paths == []
        downloaded = target.video_paths[0]

    assert not downloaded.exists()  # the context manager cleans up what it downloaded


def test_parse_and_parse_metadata_agree_when_nothing_downloads(
    downloader: ThreadsDownloader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two entry points are one walk: with no video to download they are indistinguishable."""
    _stub_html(monkeypatch, _thread_html_with_replies())

    metadata = downloader.parse_metadata(url=_REPLIES_TARGET_URL)
    with downloader.parse(url=_REPLIES_TARGET_URL) as parsed:
        assert parsed == metadata


def test_post_tolerates_null_string_fields() -> None:
    """A post whose link preview serialises image_url as null still parses."""
    post = Post.model_validate(
        obj={
            "code": "NULLPREV",
            "caption": {"text": "shared a link"},
            "user": {"username": "author", "profile_pic_url": ""},
            "text_post_app_info": {
                "link_preview_attachment": {
                    "title": "instagram.com",
                    "image_url": None,
                    "url": "https://www.instagram.com/reel/abc/",
                }
            },
        }
    )
    assert post.code == "NULLPREV"
    assert post.text_post_app_info is not None
    assert post.text_post_app_info.link_preview_attachment is not None
    assert post.text_post_app_info.link_preview_attachment.image_url == ""


def test_download_media_does_not_rebuild_a_removed_scratch_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scratch dir removed mid-download stops the worker instead of being recreated.

    The reply pipeline runs this in a thread it cannot cancel, so the caller's `rmtree` is the
    only stop signal it has. Recreating the directory here would strand the clip in a temp dir
    nobody will clean up.
    """
    scratch = tmp_path / "threads-scratch"
    scratch.mkdir()

    class _Response:
        """A body that never has to stream, because the open should fail first."""

        def raise_for_status(self) -> None:
            """Accepts the transfer."""

        def iter_content(self, chunk_size: int) -> list[bytes]:
            """Yields one chunk, which the test expects never to be written."""
            del chunk_size
            return [b"clip"]

    def fake_get(**kwargs: object) -> _Response:
        """Removes the scratch dir the way a cancelled caller's rmtree would."""
        del kwargs
        shutil.rmtree(path=scratch)
        return _Response()

    monkeypatch.setattr(target=threads_module.requests, name="get", value=fake_get)
    downloader = ThreadsDownloader(output_folder=str(scratch))

    with pytest.raises(FileNotFoundError):
        downloader.download_media(url="https://cdn.test/v.mp4", filename="clip.mp4")

    assert not scratch.exists()
