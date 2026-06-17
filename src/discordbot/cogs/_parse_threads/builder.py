"""Builds answer-model input blocks from a Threads post the user linked.

When the current message carries a Threads URL, `gen_reply` self-parses the post
(metadata only, no download) and injects the result as input blocks so the answer
model can see and answer about the linked post directly. The media rides as URLs
(`image_url` / `file_url`): the LiteLLM proxy fetches the signed Threads CDN URLs
server-side, so nothing is downloaded or transcoded here. Only the first Threads
URL in the message is parsed.

This is the rebuild of the reverted #294: the old design waited on the
`parse_threads` cog to download + post an expansion and read it back through a
relay, which raced the route gate. Here the parse is independent and metadata-only
(one HTTP fetch), so it fits comfortably inside the existing `route_done` grace.
"""

import asyncio
import tempfile

import logfire
from openai.types.responses.response_input_param import EasyInputMessageParam
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.utils.threads import ThreadsOutput, ThreadsDownloader

# Cap on media parts injected across the whole reply chain, mirroring the
# parse_threads cog's 10-embed ceiling so a huge carousel cannot bloat the answer
# input. Media is collected target-post-first, so the linked post never loses a slot.
MAX_THREADS_MEDIA_PARTS = 10

# Leads the injected blocks. The wording is load-bearing: it tells the model the link
# is ALREADY fetched below, so it answers about the post instead of falling back to
# "I cannot open this link" (the exact failure the reverted design produced).
THREADS_CONTEXT_SEPARATOR = (
    "==== The Threads link in the user's message, already fetched for you below (the post's "
    "text and images). This IS the linked post's content; answer about it directly and do NOT "
    "say you cannot open or read the link. ===="
)

# Returned when the post cannot be read (private, deleted, or otherwise unavailable) so the
# model states that plainly instead of inventing the post's contents.
THREADS_UNAVAILABLE_NOTICE = (
    "==== We tried to fetch the Threads link in the user's message but the post is private, "
    "deleted, or unavailable, so its content could not be read. Tell the user this plainly; do "
    "not invent the post's contents. ===="
)


def _system_block(text: str) -> EasyInputMessageParam:
    """Wraps one separator/notice string as a low-authority system block."""
    return EasyInputMessageParam(
        role="system", content=[ResponseInputTextParam(text=text, type="input_text")]
    )


def _render_post_text(post: ThreadsOutput, label: str) -> str:
    """Renders one post's metadata (author, time, body, engagement, url) as compact text."""
    lines = [f"[{label}] @{post.author_name}".rstrip()]
    if post.taken_at is not None:
        lines.append(f"Posted: {post.taken_at.isoformat(timespec='seconds')}")
    if post.text:
        lines.append(post.text)
    lines.append(
        f"❤️ {post.like_count:,} | 💬 {post.reply_count:,} | 🔁 {post.repost_count:,} | "
        f"🔗 {post.quote_count:,} | ↗️ {post.reshare_count:,}"
    )
    if post.url:
        lines.append(post.url)
    return "\n".join(lines)


async def build_threads_context_messages(
    *, url: str, answer_model_is_gemini: bool
) -> list[EasyInputMessageParam]:
    """Parses a Threads URL into answer-model input blocks (metadata only, no download).

    Returns `[separator, user-content-with-media]` for a readable post, or a single
    "unavailable" notice block for a private/deleted/empty post. Never raises: any parse
    error degrades to the unavailable notice so the reply pipeline is never broken by it.
    Media is rendered as URL parts for a Gemini answer model; for any other model the URLs
    ride as text instead, since the server-side URL fetch is Gemini-oriented.

    Args:
        url: The Threads post URL found in the current message.
        answer_model_is_gemini: Whether the answer model can resolve the media URLs.

    Returns:
        Input blocks ready to splice into the answer input before the current message.
    """
    try:
        with logfire.span("gen_reply threads context"):
            downloader = ThreadsDownloader(output_folder=tempfile.gettempdir())
            results = await asyncio.to_thread(downloader.parse_metadata, url=url)
    except Exception:
        logfire.warn("Threads metadata parse failed; injecting unavailable notice", _exc_info=True)
        return [_system_block(text=THREADS_UNAVAILABLE_NOTICE)]

    if not results:
        return [_system_block(text=THREADS_UNAVAILABLE_NOTICE)]

    # The chain is [root, ..., direct_parent, target]; the target (last) is the linked post.
    target_index = len(results) - 1
    text_sections = [
        _render_post_text(
            post=post,
            label=(
                "TARGET (the post the user linked)"
                if index == target_index
                else "ANCESTOR (reply-chain context)"
            ),
        )
        for index, post in enumerate(results)
    ]
    # Walk the target first so its media fills the cap before any ancestor's.
    ordered = [results[target_index], *results[:target_index]]

    if answer_model_is_gemini:
        media_parts: list[ResponseInputImageParam | ResponseInputFileParam] = []
        video_counter = 0
        for post in ordered:
            for image_url in post.image_urls:
                media_parts.append(
                    ResponseInputImageParam(image_url=image_url, detail="auto", type="input_image")
                )
            for video_url in post.video_urls:
                media_parts.append(
                    ResponseInputFileParam(
                        file_url=video_url,
                        filename=f"threads_video_{video_counter}.mp4",
                        type="input_file",
                    )
                )
                video_counter += 1
        content: list[
            ResponseInputTextParam | ResponseInputImageParam | ResponseInputFileParam
        ] = [
            ResponseInputTextParam(text="\n\n".join(text_sections), type="input_text"),
            *media_parts[:MAX_THREADS_MEDIA_PARTS],
        ]
    else:
        # Non-Gemini answer model: ride the URLs as text rather than URL media parts.
        image_urls = [image_url for post in ordered for image_url in post.image_urls]
        video_urls = [video_url for post in ordered for video_url in post.video_urls]
        extra_lines = []
        if image_urls:
            extra_lines.append("Images: " + ", ".join(image_urls[:MAX_THREADS_MEDIA_PARTS]))
        if video_urls:
            extra_lines.append("Video: " + ", ".join(video_urls[:MAX_THREADS_MEDIA_PARTS]))
        content = [
            ResponseInputTextParam(
                text="\n\n".join([*text_sections, *extra_lines]), type="input_text"
            )
        ]

    return [
        _system_block(text=THREADS_CONTEXT_SEPARATOR),
        EasyInputMessageParam(role="user", content=content),
    ]
