"""Shared vocabulary for the video quality presets."""

from typing import Literal

# The presets `/download_video` offers, and the only strings the two downloaders accept. A
# `Literal` rather than a `StrEnum` because nextcord reads a slash option's type off the
# annotation and understands `Literal` but rejects an arbitrary enum subclass; an explicit
# `choices=` still overrides the choices it derives, so the readable labels survive.
# The two downloader maps (`VideoDownloader.quality_formats`, `DouyinDownloader.quality_ratios`)
# are keyed by this type and indexed without a default, so an unanswered preset fails outright
# instead of silently downgrading; the command's own `QUALITY_CHOICES` carries the presets as its
# values. A preset added here has to be answered in all three.
VideoQuality = Literal["best", "high", "medium", "low"]

__all__ = ["VideoQuality"]
