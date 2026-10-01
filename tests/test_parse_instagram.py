"""Tests for how the Instagram-context builder renders a linked post for the answer model."""

import pytest

from discordbot.typings.context_budgets import MAX_INSTAGRAM_COMMENTS
from discordbot.services.platforms.instagram import InstagramOutput, InstagramDownloader
from discordbot.cogs.gen_reply.link_sources.instagram import build_instagram_context_messages

from tests.helpers.casting import make_stub_gemini_client
from tests.helpers.link_sources import (
    INSTAGRAM_URL,
    block_body,
    instagram_post,
    block_separator,
    serve_conversation,
    accept_image_uploads,
)


async def test_the_post_names_both_the_handle_and_the_display_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instagram identifies people by handle, but the display name is what a reader knows."""
    serve_conversation(monkeypatch, downloader=InstagramDownloader, post=instagram_post())

    blocks = await build_instagram_context_messages(
        url=INSTAGRAM_URL,
        answer_model_is_gemini=False,
        gemini_client=None,
        allow_media_ingest=True,
    )

    body = block_body(blocks=blocks)
    assert "@c_cylynn" in body
    assert "晏凌" in body


async def test_the_comment_cap_bounds_what_rides_and_the_header_says_how_many(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instagram serves the whole list; `MAX_INSTAGRAM_COMMENTS` is what the bot chooses to send.

    The header prints what actually rode, which is the only place a model can see 20 against a
    comment count of 60 and know the rest was left behind.
    """
    comments = [
        InstagramOutput(comment_id=str(index), text=f"comment {index}", author_name="a")
        for index in range(MAX_INSTAGRAM_COMMENTS + 5)
    ]
    serve_conversation(
        monkeypatch,
        downloader=InstagramDownloader,
        post=instagram_post(comments=comments, comment_count=len(comments)),
    )

    blocks = await build_instagram_context_messages(
        url=INSTAGRAM_URL,
        answer_model_is_gemini=False,
        gemini_client=None,
        allow_media_ingest=True,
    )

    body = block_body(blocks=blocks)
    assert f"[{MAX_INSTAGRAM_COMMENTS} of the post's comments" in body
    assert f"comment {MAX_INSTAGRAM_COMMENTS - 1}" in body
    assert f"comment {MAX_INSTAGRAM_COMMENTS}" not in body


async def test_both_separators_stop_short_of_promising_the_whole_comment_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instagram serves the real list, not Facebook's preload — but not on a viral post.

    Pinned on BOTH because the text-only one is the path every non-Gemini model, every disabled
    media ingest and every failed image fetch lands on, and it carried no caveat at all.
    """
    serve_conversation(monkeypatch, downloader=InstagramDownloader, post=instagram_post())
    accept_image_uploads(monkeypatch, uploaded=[])

    with_media = await build_instagram_context_messages(
        url=INSTAGRAM_URL,
        answer_model_is_gemini=True,
        gemini_client=make_stub_gemini_client(),
        allow_media_ingest=True,
    )
    text_only = await build_instagram_context_messages(
        url=INSTAGRAM_URL,
        answer_model_is_gemini=True,
        gemini_client=make_stub_gemini_client(),
        allow_media_ingest=False,
    )

    caveat = "the first page rather than every reply"
    assert caveat in block_separator(blocks=with_media)
    assert caveat in block_separator(blocks=text_only)
