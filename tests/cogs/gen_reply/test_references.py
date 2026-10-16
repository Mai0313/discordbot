"""Tests for finding the links a reply should read."""

from __future__ import annotations

import pytest
from nextcord import Embed

from discordbot.cogs.gen_reply.references import find_youtube_url, link_url_for_source

from tests.helpers.casting import as_message
from tests.helpers.reply_fakes import FakeMessage, FakeSnapshot, FakeReference, _link_source
from tests.helpers.link_sources import SAMPLE_POST_URLS


def test_find_youtube_url_searches_the_replied_to_message() -> None:
    """A YouTube link in the replied-to message is found even when the reply omits it."""
    url = "https://youtu.be/jNQXAC9IVRw"
    referenced = FakeMessage(content=f"look at this {url}")
    message = FakeMessage(content="<@999> 總結這影片")
    message.reference = FakeReference(resolved=referenced)

    assert find_youtube_url(message=as_message(fake=message)) == url


def test_find_youtube_url_ignores_url_inside_replied_to_usage_footer() -> None:
    """A memory label in the bot's footer cannot choose the next watched video."""
    url = "https://youtu.be/jNQXAC9IVRw"
    footer = f"\n\n-# model · ⬆ 1 ⬇ 2 · $0.00000000\n-# 📖 讀了 {url} 的記憶"
    answer = FakeMessage(content=f"這是我的回答{footer}")
    message = FakeMessage(content="<@999> 再說清楚一點")
    message.reference = FakeReference(resolved=answer)

    assert find_youtube_url(message=as_message(fake=message)) is None
    answer.content = footer
    answer.embeds = [Embed(url=url)]
    assert find_youtube_url(message=as_message(fake=message)) is None
    answer.content = f"這是我的回答 {url}{footer}"
    assert find_youtube_url(message=as_message(fake=message)) == url


def test_find_youtube_url_keeps_footer_shaped_text_in_the_current_message() -> None:
    """The triggering author's complete text still selects its own YouTube link."""
    url = "https://youtu.be/jNQXAC9IVRw"
    message = FakeMessage(
        content=(
            f"<@999> 再說清楚一點\n\n-# model · ⬆ 1 ⬇ 2 · $0.00000000\n-# 📖 讀了 {url} 的記憶"
        )
    )

    assert find_youtube_url(message=as_message(fake=message)) == url


def test_find_youtube_url_reads_embed_card_in_replied_to_message() -> None:
    """Footer stripping keeps the wider replied-to scan used for YouTube cards."""
    url = "https://youtu.be/jNQXAC9IVRw"
    referenced = FakeMessage(content="")
    referenced.embeds = [Embed(url=url)]
    message = FakeMessage(content="<@999> 總結這影片")
    message.reference = FakeReference(resolved=referenced)

    assert find_youtube_url(message=as_message(fake=message)) == url


def test_find_youtube_url_none_without_link() -> None:
    """No YouTube link in the message or the one it replies to returns None."""
    message = FakeMessage(content="<@999> hi")
    message.reference = FakeReference(resolved=FakeMessage(content="just chatting"))

    assert find_youtube_url(message=as_message(fake=message)) is None


_FORWARDED_URL = "https://youtu.be/jNQXAC9IVRw"


@pytest.mark.parametrize(
    ("content", "embed", "expected"),
    [
        (f"summarize this {_FORWARDED_URL}", None, _FORWARDED_URL),
        ("", Embed(title=f"watch {_FORWARDED_URL}"), _FORWARDED_URL),
        ("", Embed(url=_FORWARDED_URL), _FORWARDED_URL),
        ("lol look at this", Embed(url=_FORWARDED_URL), None),
    ],
    ids=["content", "embed-title", "bare-link-card", "captioned-forward-skips-its-embed"],
)
def test_find_youtube_url_in_a_forwarded_snapshot(
    content: str, embed: Embed | None, expected: str | None
) -> None:
    """A pure forward's link (in message.snapshots) is found wherever routing sees it, and only there.

    A captioned forward renders only its caption, so an embed-only URL there is not scanned.
    """
    message = FakeMessage(content="")
    message.snapshots = [FakeSnapshot(content=content, embeds=[embed] if embed else None)]

    assert find_youtube_url(message=as_message(fake=message)) == expected


def test_link_url_for_source_finds_the_threads_share_form() -> None:
    """The share button copies `/share/<code>`, which the registry has to select like any post.

    It resolves to the same post as the canonical form, and it is what the mobile app offers,
    so a pattern that missed it would leave the answer turn with no post context at all.
    """
    share_url = "https://www.threads.com/share/DfX81RWN8"
    message = FakeMessage(content=f"<@999> 這篇在說什麼 {share_url}")

    found = link_url_for_source(
        source=_link_source(name="threads"), message=as_message(fake=message)
    )
    assert found == share_url


def test_link_url_for_source_prefers_the_current_message() -> None:
    """With a Threads link on both, the one the user typed wins over the replied-to one."""
    referenced = FakeMessage(content=f"看看這篇 {SAMPLE_POST_URLS['threads']}")
    own_url = "https://www.threads.com/@b/post/XYZ789"
    message = FakeMessage(content=f"<@999> 跟這篇比 {own_url}")
    message.reference = FakeReference(resolved=referenced)

    found = link_url_for_source(
        source=_link_source(name="threads"), message=as_message(fake=message)
    )
    assert found == own_url


@pytest.mark.parametrize(
    ("name", "refused"),
    [
        ("facebook", "https://www.facebook.com/NASA"),
        ("instagram", "https://www.instagram.com/nasa/"),
        ("douyin", "https://live.douyin.com/123456"),
    ],
)
def test_link_url_for_source_skips_a_link_the_source_refuses(name: str, refused: str) -> None:
    """A profile or live-room link ahead of a post must not cost the answer the post (#854)."""
    source = _link_source(name=name)
    assert source.url_pattern.search(string=refused) is not None
    message = FakeMessage(content=f"<@999> {refused} 跟這篇 {SAMPLE_POST_URLS[name]}")

    found = link_url_for_source(source=source, message=as_message(fake=message))
    assert found == SAMPLE_POST_URLS[name]


def test_link_url_for_source_falls_back_past_a_link_the_source_refuses() -> None:
    """A message holding only refused links still reads the replied-to post (#854)."""
    referenced = FakeMessage(content=f"看看這篇 {SAMPLE_POST_URLS['facebook']}")
    message = FakeMessage(content="<@999> 這個粉專 https://www.facebook.com/NASA 發的這篇在講什麼")
    message.reference = FakeReference(resolved=referenced)

    found = link_url_for_source(
        source=_link_source(name="facebook"), message=as_message(fake=message)
    )
    assert found == SAMPLE_POST_URLS["facebook"]


@pytest.mark.parametrize(
    ("name", "refused"),
    [
        ("facebook", "https://www.facebook.com/NASA"),
        ("instagram", "https://www.instagram.com/nasa/"),
    ],
)
def test_link_url_for_source_skips_a_refused_link_in_the_replied_to_message(
    name: str, refused: str
) -> None:
    """One hop up, a refused link must not hide the post after it either (#854)."""
    referenced = FakeMessage(content=f"{refused} 跟這篇 {SAMPLE_POST_URLS[name]}")
    message = FakeMessage(content="<@999> 這篇在講什麼")
    message.reference = FakeReference(resolved=referenced)

    found = link_url_for_source(source=_link_source(name=name), message=as_message(fake=message))
    assert found == SAMPLE_POST_URLS[name]


def test_link_url_for_source_ignores_an_embed_card_in_the_replied_to_message() -> None:
    """The bot's own Threads expansion is not a trigger, because its first permalink is wrong.

    The Threads expansion renders the reply chain root-first with one permalink per post, so a
    first-match scan of that message would fetch the thread's top post rather than the one the
    human linked. One hop out only what the author actually typed counts.
    """
    root_url = "https://www.threads.com/@a/post/ROOT111"
    expansion = FakeMessage(content="")  # an expansion posts embeds with no content of its own
    expansion.embeds = [
        Embed(description="the thread's top post", url=root_url),
        Embed(description="the post the human linked", url=SAMPLE_POST_URLS["threads"]),
    ]
    message = FakeMessage(content="<@999> 留言在說什麼")
    message.reference = FakeReference(resolved=expansion)

    threads = _link_source(name="threads")
    assert link_url_for_source(source=threads, message=as_message(fake=message)) is None
    # The hazard itself, so this test fails if the narrow scan is ever widened: the same embeds
    # scanned in full hand back the ROOT, not the post the human linked. On the triggering
    # message that is still the behavior, since there the user chose to send that card.
    assert link_url_for_source(source=threads, message=as_message(fake=expansion)) == root_url


def test_link_url_for_source_ignores_a_url_inside_the_replied_to_usage_footer() -> None:
    """A display name in the bot's own footer cannot choose the post the next reply fetches.

    The footer credits looked-up memory owners by display name, and a name is user-chosen and
    long enough to hold a whole Threads permalink, so the span has to go before the scan.
    """
    footer = (
        f"\n\n-# model · ⬆ 1 ⬇ 2 · $0.00000000\n-# 📖 讀了 {SAMPLE_POST_URLS['threads']} 的記憶"
    )
    answer = FakeMessage(content=f"這是我的回答{footer}")
    message = FakeMessage(content="<@999> 再說清楚一點")
    message.reference = FakeReference(resolved=answer)

    threads = _link_source(name="threads")
    assert link_url_for_source(source=threads, message=as_message(fake=message)) is None
    # The body above the footer is still scanned, so the strip is what did the work here.
    answer.content = f"這是我的回答 {SAMPLE_POST_URLS['threads']}{footer}"
    assert (
        link_url_for_source(source=threads, message=as_message(fake=message))
        == SAMPLE_POST_URLS["threads"]
    )


def test_link_url_for_source_reads_a_forwarded_link_in_the_replied_to_message() -> None:
    """A forward counts for what its author wrote, on the same terms as a typed link."""
    forward = FakeMessage(content="")  # a pure forward puts its payload in snapshots
    forward.snapshots = [FakeSnapshot(content=f"看看這篇 {SAMPLE_POST_URLS['threads']}")]
    message = FakeMessage(content="<@999> 這篇底下在吵什麼")
    message.reference = FakeReference(resolved=forward)

    threads = _link_source(name="threads")
    assert (
        link_url_for_source(source=threads, message=as_message(fake=message))
        == SAMPLE_POST_URLS["threads"]
    )
    # A forwarded link CARD is not: it carries the same root-first hazard as the message's own
    # embeds, and forwarding the bot's expansion is exactly how one would arrive here.
    forward.snapshots = [FakeSnapshot(embeds=[Embed(url=SAMPLE_POST_URLS["threads"])])]
    assert link_url_for_source(source=threads, message=as_message(fake=message)) is None
