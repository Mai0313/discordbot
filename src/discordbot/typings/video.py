"""Shared vocabulary for the video quality presets."""

from typing import Literal

# `Literal` (not `StrEnum`) because nextcord accepts `Literal` in slash-option annotations
# while rejecting arbitrary enum subclasses; explicit `choices=` overrides the labels.
# Every preset here must be mapped in `VideoDownloader.quality_formats`,
# `DouyinDownloader.quality_ratios`, and `/download_video`'s `QUALITY_CHOICES`.
VideoQuality = Literal["best", "high", "medium", "low"]
