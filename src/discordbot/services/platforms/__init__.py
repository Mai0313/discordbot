"""Reading a third-party link: one module per platform, one shape across them.

Every module here answers the same question — given a URL, what is on the other side — for the
two callers that ask it: an auto-expansion cog turning a pasted link into a card, and a
`gen_reply` link source feeding the answer model. That second caller is why these live under
`services/` rather than `utils/`: a cog may not import a peer cog, so the engine both need sits
one layer down. `utils/` is the layer below this one, for things with no platform knowledge at
all (the URL anchor, the scratch dir, the link-error vocabulary).

`base.py` owns the contract. The entry point is `parse_metadata(*, url=...)`, which returns
either a `<Platform>Conversation` (a post with a discussion around it) or a `<Platform>Metadata`
(a single piece of media). `parse` is the optional second half, and only a platform that writes
a file to `output_folder` has one.

Two modules deliberately carry no downloader. `youtube.py` and `bilibili.py` are a URL pattern
each: YouTube is answered by swapping the answer backend rather than by reading a page, and
Bilibili goes through `ytdlp.py`. A module here is allowed to be just a pattern; what it may not
be is a second spelling of a pattern that already exists.

`ytdlp.py` is the other exception, and the only one to the naming rule: yt-dlp is a tool several
platforms reach the network through, not a platform, so its `VideoDownloader` is named for what
it does rather than for a site.

This layer is Discord-free. Nothing here imports nextcord, directly or through another module,
and `tests/test_package_layering.py` holds it to that — which is why planning a Douyin send lives
in `utils/douyin_delivery.py` while reading a Douyin post lives here.
"""
