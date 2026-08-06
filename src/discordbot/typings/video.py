"""The video quality vocabulary `/download_video` and both downloaders share.

`VideoQuality` names the four presets a user can ask for, and it is their only spelling, which is
what keeps the set closed. `VideoDownloader.quality_formats` (`utils/downloader.py`) is keyed by
it and turns one into a yt-dlp format string, `DouyinDownloader.quality_ratios`
(`utils/douyin.py`) likewise into the `ratio` Douyin's play endpoint takes. `QUALITY_CHOICES`
(`cogs/video/cog.py`) runs the other way, mapping the label Discord shows onto the preset, since
that is the shape nextcord's `choices=` consumes; its closed-set guarantee therefore lives in its
values rather than its keys.
The reply pipeline's link builders pick a preset off the type too (`AI_INGEST_QUALITY = "low"` in
`gen_reply/link_sources/douyin.py` and `bilibili.py`), deliberately below what the human-facing
expansion posts.

It sits in `typings/` rather than beside either downloader because the cog and both downloaders
need it, and neither `utils/` module should import its peer's vocabulary for a four-string alias.
`tests/test_download.py::test_every_quality_preset_is_answered_everywhere` is what makes a preset
added here fail loudly until every mapping answers it. Its last assertion guards the opposite
direction: the slash option's own default has to still name a live preset, so renaming or removing
one while `SlashOption(default=...)` still carries the old spelling fails there. That default is
read off the registered command rather than restated in the test, since nextcord types
`SlashOption(default=...)` as `Any`, leaving it the one preset site `ty` cannot check.
"""

from typing import Literal

# A `Literal` rather than a `StrEnum` because nextcord reads a slash option's type off the
# annotation and understands `Literal` but rejects an arbitrary enum subclass; an explicit
# `choices=` still overrides the choices it derives, so the readable labels survive. Both
# downloader tables index it without a default, so a preset nobody answered fails instead of
# silently downgrading the download.
VideoQuality = Literal["best", "high", "medium", "low"]

__all__ = ["VideoQuality"]
