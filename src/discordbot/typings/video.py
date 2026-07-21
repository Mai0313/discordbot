"""Shared vocabulary for the video quality presets."""

from typing import Literal

# The presets `/download_video` offers, and the only strings the two downloaders accept. A
# `Literal` rather than a `StrEnum` because nextcord reads a slash option's type off the
# annotation and understands `Literal` but rejects an arbitrary enum subclass; an explicit
# `choices=` still overrides the choices it derives, so the localized labels survive.
# Every site that maps a preset onto something (`VideoDownloader.quality_formats`,
# `DouyinDownloader.quality_ratios`, the command's own choices) is keyed by this type and
# indexed without a default, so a preset added here has to be answered everywhere.
VideoQuality = Literal["best", "high", "medium", "low"]

__all__ = ["VideoQuality"]
