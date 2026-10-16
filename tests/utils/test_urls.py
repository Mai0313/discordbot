"""Tests for pulling a URL out of pasted text and checking whose host it names."""

from discordbot.utils.urls import normalized_host, extract_first_url, host_matches_domain
from discordbot.services.platforms.douyin import DOUYIN_URL_RE
from discordbot.services.platforms.youtube import YOUTUBE_URL_RE
from discordbot.cogs.gen_reply.link_sources.registry import LINK_CONTEXT_SOURCES

from tests.helpers.link_sources import SAMPLE_POST_URLS


def test_host_matches_domain_refuses_a_lookalike_host() -> None:
    """The exact-label rule every site checks its host through."""
    assert host_matches_domain(host="douyin.com", domain="douyin.com")
    assert host_matches_domain(host="v.douyin.com", domain="douyin.com")
    assert not host_matches_domain(host="douyin.com.attacker.com", domain="douyin.com")
    assert not host_matches_domain(host="notdouyin.com", domain="douyin.com")
    assert not host_matches_domain(host="", domain="douyin.com")


def test_normalized_host_reads_a_scheme_less_paste_and_never_raises() -> None:
    """A pasted host with no scheme still parses, and malformed input answers rather than raising.

    This runs in routing checks that sit ahead of any error handling, so a `ValueError` out of
    urlparse would escape into a command handler that has nothing to say about it.
    """
    assert normalized_host(url="https://V.Douyin.com/abc") == "v.douyin.com"
    assert normalized_host(url="v.douyin.com/abc") == "v.douyin.com"
    assert normalized_host(url="https://[abc/x") == ""


def test_every_url_pattern_shares_the_generic_start_anchor() -> None:
    """A link glued to the end of an ASCII word is not a link to ANY of the scanners.

    The site patterns used to carry no start anchor at all, so `xhttps://v.douyin.com/abc` was
    refused by the generic scanner and matched by every site one. CJK in front is not an ASCII
    word character, so those still match (#492). The patterns are read off the registry, plus
    YouTube's, which gates the answer turn rather than a source.
    """
    # The shared sample table's one coverage guard: a new source fails here until it has a URL.
    assert set(SAMPLE_POST_URLS) == {source.name for source in LINK_CONTEXT_SOURCES}
    patterns = [
        (source.url_pattern, SAMPLE_POST_URLS[source.name]) for source in LINK_CONTEXT_SOURCES
    ]
    patterns.append((YOUTUBE_URL_RE, "https://www.youtube.com/watch?v=dQw4w9WgXcQ"))

    for pattern, url in patterns:
        assert pattern.search(string=url) is not None, url
        assert pattern.search(string=f"x{url}") is None, url
        assert pattern.search(string=f"看這個{url}") is not None, url


def test_a_bare_url_is_left_untouched() -> None:
    """The common case must be unchanged: a bare URL passes through as-is."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert extract_first_url(text=url, patterns=(DOUYIN_URL_RE,)) == url


def test_sentence_punctuation_after_a_link_is_dropped() -> None:
    """A link written mid-sentence must not carry the full stop into the request."""
    assert extract_first_url(text="see https://example.com/a/b.", patterns=()) == (
        "https://example.com/a/b"
    )


def test_a_link_typed_flush_against_chinese_is_found() -> None:
    r"""A link with no space in front of it is still a link, but only past a non-ASCII word.

    The generic pattern used to head on `\b`, which counts CJK as a word character, so
    `這個https://...` found nothing at all (#492). A link glued to the end of an ASCII word is
    still refused, and falls through to the unchanged-passthrough behaviour below.
    """
    assert extract_first_url(text="幫我下載這個https://example.com/a/b", patterns=()) == (
        "https://example.com/a/b"
    )
    assert extract_first_url(text="xhttps://example.com/a/b", patterns=()) == (
        "xhttps://example.com/a/b"
    )


def test_unparseable_input_is_passed_through() -> None:
    """Text with no URL is handed on unchanged, so it fails downstream as it always did."""
    assert extract_first_url(text="  not a url  ", patterns=()) == "not a url"
