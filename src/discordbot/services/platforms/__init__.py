"""Platform URL patterns, page readers, and media downloaders.

`base.py` defines the shared contract: `parse_metadata(url=...)` returns either a
`<Platform>Conversation` or a `<Platform>Metadata`. Platforms that write files to `output_folder`
add a platform-specific `parse` or `download` method.

`youtube.py` and `bilibili.py` export URL patterns only; Bilibili downloads go through
`ytdlp.py::VideoDownloader`. This package never imports `nextcord`, directly or transitively.
"""
