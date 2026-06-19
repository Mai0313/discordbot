"""Tests for the inline `<voice>` / `<image>` control-tag parser."""

from __future__ import annotations

from discordbot.cogs._gen_reply.markup import extract_reply_segments, strip_tags_for_preview


def test_extract_plain_text_has_no_spans() -> None:
    """A reply with no control tags is returned untouched with both spans empty."""
    segments = extract_reply_segments(text="just a normal reply")

    assert segments.display_text == "just a normal reply"
    assert segments.voice_text == ""
    assert segments.image_prompt == ""
    assert segments.voice_requested is False
    assert segments.image_requested is False


def test_extract_voice_span_keeps_content_drops_tags() -> None:
    """A <voice> span stays visible in the text; only the tags are removed and only it is spoken."""
    segments = extract_reply_segments(text="開頭 <voice>朗讀這段</voice> 結尾")

    assert segments.display_text == "開頭 朗讀這段 結尾"
    assert segments.voice_text == "朗讀這段"
    assert segments.voice_requested is True
    assert segments.image_requested is False


def test_extract_multiple_voice_spans_concatenate() -> None:
    """Every <voice> span contributes to the spoken text, joined by a newline; all stay visible."""
    segments = extract_reply_segments(text="a <voice>one</voice> b <voice>two</voice> c")

    assert segments.display_text == "a one b two c"
    assert segments.voice_text == "one\ntwo"
    assert segments.voice_requested is True


def test_extract_image_block_removed_entirely() -> None:
    """An <image> block is removed whole; its description never leaks into the visible text."""
    segments = extract_reply_segments(text="這是你要的 <image>a red cat</image> 圖")

    assert "<image>" not in segments.display_text
    assert "a red cat" not in segments.display_text
    assert segments.display_text == "這是你要的 圖"
    assert segments.image_prompt == "a red cat"
    assert segments.image_requested is True
    assert segments.voice_requested is False


def test_extract_first_image_block_is_the_prompt() -> None:
    """Only the first image block drives the prompt, but every block is stripped from the text."""
    segments = extract_reply_segments(text="<image>first</image> mid <image>second</image>")

    assert segments.image_prompt == "first"
    assert "first" not in segments.display_text
    assert "second" not in segments.display_text
    assert segments.display_text == "mid"


def test_extract_voice_and_image_together() -> None:
    """A reply with both spans keeps the voice text and removes the image block."""
    segments = extract_reply_segments(text="看 <image>a cat</image> 也 <voice>聽我說</voice>")

    assert segments.display_text == "看 也 聽我說"
    assert segments.voice_text == "聽我說"
    assert segments.image_prompt == "a cat"
    assert segments.voice_requested is True
    assert segments.image_requested is True


def test_extract_unclosed_image_drops_to_end() -> None:
    """An unclosed <image> (model forgot the closer) drops from the tag to the end of the reply."""
    segments = extract_reply_segments(text="前言 <image>a cat with no closer")

    assert segments.display_text == "前言"
    assert segments.image_prompt == "a cat with no closer"
    assert segments.image_requested is True


def test_extract_unclosed_voice_keeps_content() -> None:
    """An unclosed <voice> keeps its content visible and still feeds the spoken clip."""
    segments = extract_reply_segments(text="嗆你 <voice>大聲說出來")

    assert segments.display_text == "嗆你 大聲說出來"
    assert segments.voice_text == "大聲說出來"
    assert segments.voice_requested is True


def test_extract_tolerates_whitespace_in_bare_tags() -> None:
    """A bare tag with stray inner whitespace is still recognised as a control block."""
    segments = extract_reply_segments(text="y < image >draw< / image >")

    assert segments.image_prompt == "draw"
    assert "draw" not in segments.display_text
    assert segments.display_text == "y"


def test_extract_skips_inline_code_tags() -> None:
    """A backticked tag is inline code, not a control tag, so it stays in the reply verbatim."""
    segments = extract_reply_segments(text="use `<voice>say it</voice>` to speak a span")

    assert segments.voice_requested is False
    assert "`<voice>say it</voice>`" in segments.display_text


def test_extract_skips_fenced_code_tags() -> None:
    """An <image> block inside a fenced code example is left intact, not sent to generation."""
    reply = "Here is the markup:\n```\n<image>not a real request</image>\n```\ndone"

    segments = extract_reply_segments(text=reply)

    assert segments.image_requested is False
    assert segments.display_text == reply


def test_extract_empty_span_is_not_requested() -> None:
    """An empty span carries no work, so it is not flagged as requested."""
    segments = extract_reply_segments(text="hi <voice></voice> <image></image>")

    assert segments.voice_requested is False
    assert segments.image_requested is False


def test_extract_preserves_whitespace_when_no_tags() -> None:
    """A reply with no control tags is returned byte-for-byte (code/tables/art survive)."""
    reply = "Here:\n```\nx   =   1\ncol1    col2\n```\nascii  ->  art"

    segments = extract_reply_segments(text=reply)

    # No collapsing of interior runs of spaces anywhere in an ordinary reply.
    assert segments.display_text == reply


def test_extract_image_seam_is_local_only() -> None:
    """Removing an image block heals only its own gap, not double spaces elsewhere."""
    segments = extract_reply_segments(text="a  b <image>cat</image> c  d")

    assert segments.image_prompt == "cat"
    assert "<image>" not in segments.display_text
    # The image gap closes to a single space, but the unrelated double spaces are untouched.
    assert "a  b" in segments.display_text
    assert "c  d" in segments.display_text
    assert "b c" in segments.display_text


def test_preview_hides_complete_image_block() -> None:
    """A finished image block never shows in the live preview."""
    assert strip_tags_for_preview(text="這是 <image>a cat</image> 圖") == "這是  圖".strip()


def test_preview_hides_unclosed_image_tail() -> None:
    """A streaming image block (open tag, not yet closed) is hidden from its tag onward."""
    assert strip_tags_for_preview(text="hello <image>a red ca") == "hello"


def test_preview_hides_partial_image_open() -> None:
    """A half-typed image open tag at the very end is trimmed off the preview."""
    assert strip_tags_for_preview(text="hello <imag") == "hello"
    assert strip_tags_for_preview(text="hello <im") == "hello"


def test_preview_keeps_voice_content_drops_tags() -> None:
    """A voice span shows its content in the preview with the tags removed."""
    assert strip_tags_for_preview(text="keep <voice>spoken</voice> end") == "keep spoken end"


def test_preview_hides_partial_voice_tag() -> None:
    """A half-typed voice tag at the end is trimmed before it can flicker in."""
    assert strip_tags_for_preview(text="keep <voice>spoken</vo") == "keep spoken"
    assert strip_tags_for_preview(text="嗆你 </voic") == "嗆你"


def test_preview_leaves_plain_text() -> None:
    """Plain text with no tag fragment is returned unchanged."""
    assert strip_tags_for_preview(text="正常文字") == "正常文字"


def test_preview_keeps_tags_inside_inline_code() -> None:
    """A backticked tag in a streaming snapshot stays visible (it is code, not a control tag)."""
    assert strip_tags_for_preview(text="see `<image>` in code") == "see `<image>` in code"
