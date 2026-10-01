"""Tests for how the Facebook-context builder renders a linked post for the answer model."""

import pytest

from discordbot.services.platforms.facebook import FacebookOutput, FacebookDownloader
from discordbot.cogs.gen_reply.link_sources.facebook import build_facebook_context_messages

from tests.helpers.link_sources import (
    FACEBOOK_URL,
    block_body,
    facebook_post,
    block_separator,
    serve_conversation,
)


async def test_the_post_names_its_author_and_its_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """The group is where a post lives, and a reader cannot get it from the post itself."""
    serve_conversation(monkeypatch, downloader=FacebookDownloader, post=facebook_post())

    blocks = await build_facebook_context_messages(
        url=FACEBOOK_URL, answer_model_is_gemini=False, gemini_client=None, allow_media_ingest=True
    )

    body = block_body(blocks=blocks)
    assert "Somebody" in body
    assert "Some Group" in body


async def test_the_block_says_the_comments_are_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    """The page preloads a handful of a much longer thread, and the model must be told."""
    comments = [FacebookOutput(comment_id="111", text="only one shown", author_name="A")]
    serve_conversation(
        monkeypatch, downloader=FacebookDownloader, post=facebook_post(comments=comments)
    )

    blocks = await build_facebook_context_messages(
        url=FACEBOOK_URL, answer_model_is_gemini=False, gemini_client=None, allow_media_ingest=True
    )

    assert "not the whole discussion" in block_body(blocks=blocks)
    assert "never the whole discussion" in block_separator(blocks=blocks)
