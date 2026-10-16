"""Tests for parsing the inline markers out of an answer."""

from __future__ import annotations

import pytest

from discordbot.cogs.gen_reply.markers import (
    MAX_MEMORY_NOTES,
    MAX_INLINE_IMAGES,
    extract_inline_markers,
    scrub_markers_for_preview,
)


def test_extract_inline_markers_voice_keeps_content() -> None:
    """A <generate-voice> segment stays in the visible text; only the tags are stripped."""
    markers = extract_inline_markers(text="嗆爆你 <generate-voice>聽好了</generate-voice> 滾")
    assert markers.cleaned_text == "嗆爆你 聽好了 滾"
    assert markers.voice_text == "聽好了"
    assert markers.voice_requested is True
    assert markers.image_prompts == []


def test_extract_inline_markers_multiple_voice_segments_concatenate() -> None:
    """Multiple <generate-voice> segments concatenate into one spoken input, all content kept."""
    markers = extract_inline_markers(
        text="<generate-voice>第一</generate-voice>中間<generate-voice>第二</generate-voice>"
    )
    assert markers.voice_text == "第一\n第二"
    assert markers.cleaned_text == "第一中間第二"


def test_extract_inline_markers_image_block_removed() -> None:
    """An <generate-image> block (tags AND content) is pulled from the visible reply."""
    markers = extract_inline_markers(
        text="看這張\n<generate-image>a red cat on a sofa</generate-image>"
    )
    assert markers.image_prompts == ["a red cat on a sofa"]
    assert "<generate-image>" not in markers.cleaned_text
    assert "a red cat" not in markers.cleaned_text
    assert markers.cleaned_text == "看這張"
    assert markers.voice_requested is False


def test_extract_inline_markers_multiple_image_blocks_in_order() -> None:
    """Every <generate-image> block becomes an image request, kept in document order."""
    markers = extract_inline_markers(
        text="先看\n<generate-image>a red cat</generate-image>\n再看\n<generate-image>a blue dog</generate-image>"
    )
    assert markers.image_prompts == ["a red cat", "a blue dog"]
    assert "<generate-image>" not in markers.cleaned_text
    assert "red cat" not in markers.cleaned_text
    assert "blue dog" not in markers.cleaned_text


def test_extract_inline_markers_closed_then_unclosed_image_both_pulled() -> None:
    """A complete block plus a trailing unclosed <generate-image> are both captured, in order."""
    markers = extract_inline_markers(
        text="看\n<generate-image>a red cat</generate-image>\n還有\n<generate-image>a blue dog"
    )
    assert markers.image_prompts == ["a red cat", "a blue dog"]
    assert "<generate-image>" not in markers.cleaned_text


def test_extract_inline_markers_unclosed_image_is_pulled() -> None:
    """An unclosed trailing <generate-image> (model forgot to close) never leaks its description."""
    markers = extract_inline_markers(text="來囉\n<generate-image>a sunset over the sea")
    assert markers.image_prompts == ["a sunset over the sea"]
    assert "<generate-image>" not in markers.cleaned_text
    assert "sunset" not in markers.cleaned_text
    assert markers.cleaned_text == "來囉"


_CLIP_MARKERS = pytest.mark.parametrize(
    ("tag", "field"), [("generate-music", "music_prompt"), ("generate-video", "video_prompt")]
)


@_CLIP_MARKERS
def test_extract_inline_markers_clip_block_removed(tag: str, field: str) -> None:
    """A `<generate-music>` / `<generate-video>` block (tags AND content) is pulled from the reply."""
    markers = extract_inline_markers(text=f"動起來\n<{tag}>a wave crashing on rocks</{tag}>")
    assert getattr(markers, field) == "a wave crashing on rocks"
    assert f"<{tag}>" not in markers.cleaned_text
    assert "wave" not in markers.cleaned_text
    assert markers.cleaned_text == "動起來"


@_CLIP_MARKERS
def test_extract_inline_markers_only_first_clip_block_kept(tag: str, field: str) -> None:
    """Only the first non-empty clip block is kept (a single clip per reply)."""
    markers = extract_inline_markers(
        text=f"<{tag}>first scene</{tag}>中間<{tag}>second scene</{tag}>"
    )
    assert getattr(markers, field) == "first scene"
    assert f"<{tag}>" not in markers.cleaned_text
    assert "second scene" not in markers.cleaned_text


@_CLIP_MARKERS
def test_extract_inline_markers_unclosed_clip_is_pulled(tag: str, field: str) -> None:
    """An unclosed trailing clip tag (model forgot to close) never leaks its description."""
    markers = extract_inline_markers(text=f"等我一下\n<{tag}>a slow zoom over a city")
    assert getattr(markers, field) == "a slow zoom over a city"
    assert f"<{tag}>" not in markers.cleaned_text
    assert "zoom" not in markers.cleaned_text
    assert markers.cleaned_text == "等我一下"


def test_extract_inline_markers_ignores_real_html_svg_ssml_tags() -> None:
    """A reply that only SHOWS `<video>` / `<image>` / `<voice>` example markup is left untouched.

    The markers are hyphenated (`generate-*`) precisely so a real HTML `<video>`, SVG `<image>`, or
    SSML `<voice>` tag the answer is explaining is never mistaken for a generation request, even
    when it is not wrapped in a code block.
    """
    text = (
        "HTML 的 <video></video> 嵌入影片,SVG 用 <image href='a.png'/>,"
        "SSML 用 <voice>Hi</voice> 指定嗓音。"
    )
    markers = extract_inline_markers(text=text)
    # No generation is triggered and the whole explanation survives verbatim.
    assert markers.video_prompt is None
    assert markers.image_prompts == []
    assert markers.voice_requested is False
    assert markers.cleaned_text == text


def test_extract_inline_markers_memory_notes_are_pulled_per_kind() -> None:
    """The three memory tags are collected separately and none of them reaches the reader.

    A note is instruction to the memory pipeline, so it is pulled whole like an image block
    rather than left visible like a voice span: a reply that recites what it just recorded reads
    as the bot talking about itself instead of answering.
    """
    markers = extract_inline_markers(
        text=(
            "沒問題<write-memory>使用者希望用繁體中文回覆</write-memory>"
            "<forget-memory>使用者已經不住台中了</forget-memory>"
            "<write-server-memory>這個社群把週五叫做炸雞日</write-server-memory>,還有什麼要問的"
        )
    )
    assert markers.memory_notes == ["使用者希望用繁體中文回覆"]
    assert markers.forget_notes == ["使用者已經不住台中了"]
    assert markers.server_memory_notes == ["這個社群把週五叫做炸雞日"]
    assert markers.cleaned_text == "沒問題,還有什麼要問的"


def test_extract_inline_markers_server_memory_tag_is_not_read_as_a_user_one() -> None:
    """`<write-server-memory>` shares a prefix with `<write-memory>` and must not be split by it."""
    markers = extract_inline_markers(
        text="<write-server-memory>這裡週五吃炸雞</write-server-memory>"
    )
    assert markers.server_memory_notes == ["這裡週五吃炸雞"]
    assert markers.memory_notes == []
    assert markers.cleaned_text == ""


def test_extract_inline_markers_unclosed_memory_note_is_pulled() -> None:
    """An unclosed trailing memory tag still never leaks the note into the visible reply."""
    markers = extract_inline_markers(text="好喔\n<forget-memory>使用者不再玩那款遊戲")
    assert markers.forget_notes == ["使用者不再玩那款遊戲"]
    assert markers.cleaned_text == "好喔"


def test_extract_inline_markers_caps_memory_notes_per_kind() -> None:
    """A model that emits a note per sentence is trimmed rather than trusted.

    The cap is a sanity bound, not a Discord limit: the evaluator downstream still decides
    whether any kept note survives, but a turn producing twenty notes has misread the
    instruction and should not be able to flood the raw file with them.
    """
    text = "".join(f"<write-memory>note {index}</write-memory>" for index in range(12))
    markers = extract_inline_markers(text=text)
    assert markers.memory_notes == [f"note {index}" for index in range(MAX_MEMORY_NOTES)]


def test_extract_inline_markers_caps_images_but_counts_every_request() -> None:
    """Extraction keeps the first `MAX_INLINE_IMAGES` descriptions and still counts the rest."""
    text = "".join(
        f"<generate-image>image {index}</generate-image>" for index in range(MAX_INLINE_IMAGES + 3)
    )
    markers = extract_inline_markers(text=text)
    assert markers.image_prompts == [f"image {index}" for index in range(MAX_INLINE_IMAGES)]
    assert markers.image_requests == MAX_INLINE_IMAGES + 3


def test_scrub_markers_for_preview_hides_a_streaming_memory_note() -> None:
    """A half-streamed memory tag must not flicker into the live preview.

    The preview is edited as deltas arrive, so a note that becomes invisible only at finalize
    time would still be readable in the channel for the seconds before that.
    """
    assert scrub_markers_for_preview(text="好的 <write-memory>使用者喜歡") == "好的"
    assert scrub_markers_for_preview(text="好的 <write-mem") == "好的"
    assert scrub_markers_for_preview(text="好的 <write-server-memory>這裡") == "好的"


def test_scrub_markers_for_preview_hides_streaming_fragments() -> None:
    """Markers arriving mid-stream are hidden from the live preview before the final extract."""
    # A partial trailing tag is trimmed; the content before it stays.
    assert scrub_markers_for_preview(text="嗆你 <generate-voi") == "嗆你"
    # A complete <generate-voice> pair is stripped but its content stays visible.
    assert (
        scrub_markers_for_preview(text="嗆你 <generate-voice>聽好</generate-voice>") == "嗆你 聽好"
    )
    # An unclosed <generate-image> open and everything after it is hidden whole (the block is pulled).
    assert scrub_markers_for_preview(text="看這 <generate-image>a red ca") == "看這"
    # A complete <generate-image> block is removed whole.
    assert (
        scrub_markers_for_preview(text="看這<generate-image>a cat</generate-image>之後")
        == "看這之後"
    )
    # A still-streaming <generate-video> open and a complete block are both hidden whole.
    assert scrub_markers_for_preview(text="動起來 <generate-video>a wa") == "動起來"
    assert (
        scrub_markers_for_preview(text="看這<generate-video>a wave</generate-video>之後")
        == "看這之後"
    )
    assert scrub_markers_for_preview(text="正常文字") == "正常文字"
