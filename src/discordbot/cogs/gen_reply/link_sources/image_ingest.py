"""Fetching a linked post's images and uploading them for the answer model to look at.

Shared by every source whose media is a set of image URLs the page handed over. A source that
downloads a file of its own does not use this.

Two properties are the whole point. Every item is independent and best-effort, so one expired
signed CDN url never costs the rest; and the step is bounded here rather than left to the
caller's grace, so a slow fetch still produces the honest text-only block instead of being
cancelled with nothing to inject.
"""

import asyncio

from google import genai
import logfire
from openai.types.responses.response_input_file_param import ResponseInputFileParam

from discordbot.typings.timeouts import LINK_MEDIA_TIMEOUT_SECONDS
from discordbot.cogs.gen_reply.files_api import upload_as_input_file
from discordbot.cogs.gen_reply.attachment.loaders import load_image_bytes


async def _upload_each(
    *, platform: str, post_url: str, image_urls: list[str], gemini_client: genai.Client
) -> list[ResponseInputFileParam]:
    """Uploads the images concurrently, keeping whatever succeeded."""

    async def image_part(index: int, image_url: str) -> ResponseInputFileParam | None:
        """Fetches, downscales and uploads one image.

        `load_image_bytes` downscales to the provider's effective resolution, which matters
        because these are full-resolution originals.
        """
        loaded = await load_image_bytes(source=image_url)
        return await upload_as_input_file(
            client=gemini_client,
            source=loaded.data,
            mime_type=loaded.mime_type,
            filename=f"{platform.lower()}_image_{index}.jpg",
            timeout_seconds=LINK_MEDIA_TIMEOUT_SECONDS,
        )

    results = await asyncio.gather(
        *(image_part(index, image_url) for index, image_url in enumerate(image_urls)),
        return_exceptions=True,
    )
    parts: list[ResponseInputFileParam] = []
    for result in results:
        if isinstance(result, BaseException):
            logfire.warn(
                f"{platform} image ingestion failed for one item",
                url=post_url,
                error_type=type(result).__name__,
                _exc_info=result,
            )
            continue
        if result is not None:
            parts.append(result)
    return parts


async def upload_post_images(
    *, platform: str, post_url: str, image_urls: list[str], cap: int, gemini_client: genai.Client
) -> list[ResponseInputFileParam]:
    """Uploads up to `cap` of a post's images, degrading to none rather than raising.

    Args:
        platform: The platform's display name, for the log lines.
        post_url: The post the images came from, so a warning can be joined to it.
        image_urls: Every image the post carries, in page order.
        cap: How many of them one reply may pay a fetch and an upload for.
        gemini_client: Direct-to-Google client the upload goes through.

    Returns:
        The parts that made it, which the caller must COUNT rather than assume: a block saying
        the post carries three images beside a separator saying they are attached below is how a
        model ends up describing pictures it was never given.
    """
    try:
        async with asyncio.timeout(delay=LINK_MEDIA_TIMEOUT_SECONDS):
            return await _upload_each(
                platform=platform,
                post_url=post_url,
                image_urls=image_urls[:cap],
                gemini_client=gemini_client,
            )
    except TimeoutError:
        logfire.warn(
            f"{platform} image ingestion exceeded its bound; answering from the text",
            url=post_url,
            timeout_seconds=LINK_MEDIA_TIMEOUT_SECONDS,
            _exc_info=True,
        )
        return []
    except Exception as error:
        # Broad on purpose: this must degrade to the text-only block rather than raise into the
        # reply pipeline, so the type is recorded as a field instead of by narrowing.
        logfire.warn(
            f"{platform} image ingestion failed; answering from the text",
            url=post_url,
            error_type=type(error).__name__,
            _exc_info=error,
        )
        return []


def image_count_line(*, carried: int, attached: int) -> str:
    """Says how many of a post's images the model was actually handed.

    The two numbers differ whenever the cap binds or an upload fails, which is routine, and the
    separator above the block already promises that images are attached below. Naming only the
    count the post carries turns that into a claim about pictures the model never received.

    Args:
        carried: How many images the post has.
        attached: How many of them rode into the block.

    Returns:
        One line for the rendered post text.
    """
    if attached:
        return f"The post carries {carried} image(s), {attached} of them attached below."
    return f"The post carries {carried} image(s), none of them attached."
