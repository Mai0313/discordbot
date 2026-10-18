"""Platform URL patterns, page readers, and media downloaders.

`base.py` defines the shared contract: `parse_metadata(url=...)` returns either a
`<Platform>Conversation` (a post with its surrounding discussion) or a `<Platform>Metadata`
(a single piece of media). A platform that writes files to `output_folder` adds a second method
of its own; `base.py` explains why the base class leaves that method undeclared.

`youtube.py` and `bilibili.py` carry only a URL pattern each: YouTube is handled by the answer
backend directly rather than by scraping a page, and Bilibili downloads go through `ytdlp.py`.
`ytdlp.py` is the only module named for a tool rather than a site, since several platforms reach
the network through `VideoDownloader`.

This package is Discord-free and never imports `nextcord`, directly or transitively.
"""
