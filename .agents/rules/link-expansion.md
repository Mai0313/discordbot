---
description: "Read before changing, planning or reviewing link expansion: cogs/parse_*/, cogs/video/, gen_reply/link_sources/, services/platforms/, the expansion, link-error, URL and Douyin-delivery utils, and their tests and helpers."
trigger: model_decision
paths:
  - "src/discordbot/cogs/parse_*/**"
  - "src/discordbot/cogs/video/**"
  - "src/discordbot/cogs/gen_reply/link_sources/**"
  - "src/discordbot/services/platforms/**"
  - "src/discordbot/utils/douyin_delivery.py"
  - "src/discordbot/utils/expansion_cog.py"
  - "src/discordbot/utils/expansion_placeholder.py"
  - "src/discordbot/utils/link_errors.py"
  - "src/discordbot/utils/urls.py"
  - "tests/cogs/parse_*/**"
  - "tests/cogs/video/**"
  - "tests/cogs/gen_reply/link_sources/**"
  - "tests/services/platforms/**"
  - "tests/guards/test_expansion_contract.py"
  - "tests/guards/test_link_source_emojis.py"
  - "tests/guards/test_platform_shape.py"
  - "tests/utils/test_expansion_cog.py"
  - "tests/utils/test_expansion_placeholder.py"
  - "tests/utils/test_link_errors.py"
  - "tests/utils/test_urls.py"
  - "tests/helpers/link_sources.py"
---

## Link Expansion

- **An auto-expansion cog subclasses `utils/expansion_cog.py::ExpansionCog` (or its `ConversationExpansionCog`), whose docstrings list the hooks it supplies**: `tests/guards/test_expansion_contract.py` discovers cogs by that base and pins what one may not take back from the shell, so a cog written beside it escapes every check.
- **An expansion cog and an AI link source are a pair, and matching the existing ones is the default; diverging needs a reason.** A cog's `SOURCE` must be a registered source (`tests/guards/test_expansion_contract.py`, `tests/guards/test_link_source_emojis.py`); the reverse is unchecked, so a new registry source is not done until its cog is built or why not is written down, and whether its parser carries the shared conversation shape is decided (`tests/guards/test_platform_shape.py`).
- **Bilibili (a registry source with no cog) and YouTube (not a source, see `watch_video` in `reply-pipeline.md`) are AI-only on purpose, because of length.** A post on an expanding platform is bounded by what anyone may upload there, while these links can be a four-hour video that expanding would download and post.
- **A platform reader raises `utils/link_errors.py` classes (or parents its own onto them)**: the shell picks the reaction and log severity off the class, so any other exception but a `TimeoutError` paints the failure cross.
- **A URL `parse_threads` publishes must never name whoever shared the post** (`services/platforms/threads.py::ThreadsDownloader._build_conversation`). Do not reintroduce the `total_size` byte guard (media hosting replaced it), a media-count guard, or the permalink follow-up replies (#645).
- **Build a conversation once and never mutate it**: `target` is a `cached_property`, which pydantic does not invalidate on a `chain` append or reassignment.
- **A new platform reader meets the existing readers' traps, each stated in its module**: the full browser header set, not the User-Agent, unlocks a page (`page_json.py`); comment nesting comes from parent ids, never page order, and a page-wide comment scan is scoped by post id (`facebook.py`); a lookalike non-post URL is refused (`is_<platform>_post_url`); and an expiring signed video URL is never a card's link, which outlives it (`parse_instagram/cog.py::_video_link`).
- **The expansion scrapers are not unified onto yt-dlp**: that was weighed and dropped (no coverage gained, a second failure mode); a new video-only platform reuses `services/platforms/ytdlp.py::VideoDownloader`. Nor is the Douyin share-page reader swapped for a third-party download service (tiktokio.com, weighed 2026-10-11: undocumented API, no author, 720p against the bot's 1080p, terms that forbid automation).
- **Every expansion card's images are uploaded by the shell, not linked** (`utils/expansion_cog.py::ExpansionCog._attach_card_images`, #1122): Discord probes a linked embed image once at send and stores a failed probe as a 0x0 image it never retries, which on 2026-10-26 emptied a third of Threads and Instagram cards, all-or-nothing per card (Facebook and Twitter none), against the 2 of 23 images #721's upload was reverted over as too heavy. An image the message has no room for keeps its link rather than being hosted; author icons stay linked.
