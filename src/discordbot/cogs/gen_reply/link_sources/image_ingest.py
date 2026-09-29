"""Fetching a linked post's media and uploading it for the answer model to look at.

`upload_image` is the one-image step. `upload_post_images` runs it over a post's image URLs, and
two properties are the whole point of that. Every item is independent and best-effort, so one
expired signed CDN url never costs the rest; and the step is bounded rather than left to the
caller's grace, so a slow fetch still produces the honest text-only block instead of being
cancelled with nothing to inject. A source that keeps its own per-URL accounting, or downloads
files of its own beside the images, runs the one-image step itself.

`bounded_media_step` is that bound, and every source's media step runs under it, whether or not
its media is images.
"""

from typing import Any
import asyncio
from collections.abc import Coroutine

from google import genai
import logfire
from openai.types.responses.response_input_file_param import ResponseInputFileParam

from discordbot.typings.timeouts import LINK_MEDIA_TIMEOUT_SECONDS
from discordbot.cogs.gen_reply.files_api import upload_as_input_file
from discordbot.cogs.gen_reply.attachment.loaders import load_image_bytes


async def bounded_media_step[ResultT](  # noqa: PLR0913 -- the step, its log wording, its fallback value and its bound all vary per source
    *,
    step: Coroutine[Any, Any, ResultT],
    subject: str,
    fallback: str,
    degraded: ResultT,
    url: str,
    timeout_seconds: float,
    timeout_fields: dict[str, Any] | None = None,
) -> ResultT:
    """Runs a source's media step under its own bound, degrading rather than raising.

    Bounded here rather than left to the caller's grace so a slow fetch still produces the honest
    text-only block instead of being cancelled with nothing to inject.

    Args:
        step: The fetch-and-upload work, not yet awaited.
        subject: What the log lines call the step, e.g. "Douyin media".
        fallback: What the log lines say the answer falls back to, e.g. "the caption".
        degraded: What comes back instead when the step times out or fails.
        url: The post the media belongs to, so a warning can be joined to it.
        timeout_seconds: The bound.
        timeout_fields: Extra fields for the warning a timeout logs.

    Returns:
        The step's own result, or `degraded`.
    """
    try:
        async with asyncio.timeout(delay=timeout_seconds):
            return await step
    except TimeoutError:
        logfire.warn(
            f"{subject} ingestion exceeded its bound; answering from {fallback}",
            url=url,
            timeout_seconds=timeout_seconds,
            **(timeout_fields or {}),
            _exc_info=True,
        )
        return degraded
    except Exception as error:
        # Broad on purpose: this must degrade to the text-only block rather than raise into the
        # reply pipeline, so the type is recorded as a field instead of by narrowing.
        logfire.warn(
            f"{subject} ingestion failed; answering from {fallback}",
            url=url,
            error_type=type(error).__name__,
            _exc_info=error,
        )
        return degraded


async def upload_image(
    *, image_url: str, filename: str, gemini_client: genai.Client
) -> ResponseInputFileParam | None:
    """Fetches, downscales and uploads one image, raising whatever the fetch or upload raised.

    `load_image_bytes` downscales to the provider's effective resolution, which matters because
    these are full-resolution originals. What one lost image costs is the caller's call.

    Args:
        image_url: The image to fetch.
        filename: The name the upload carries.
        gemini_client: Direct-to-Google client the upload goes through.

    Returns:
        The uploaded part, or None when the upload produced none.
    """
    loaded = await load_image_bytes(source=image_url)
    return await upload_as_input_file(
        client=gemini_client,
        source=loaded.data,
        mime_type=loaded.mime_type,
        filename=filename,
        timeout_seconds=LINK_MEDIA_TIMEOUT_SECONDS,
    )


async def _upload_each(
    *, platform: str, post_url: str, image_urls: list[str], gemini_client: genai.Client
) -> list[ResponseInputFileParam]:
    """Uploads the images concurrently, keeping whatever succeeded."""
    results = await asyncio.gather(
        *(
            upload_image(
                image_url=image_url,
                filename=f"{platform.lower()}_image_{index}.jpg",
                gemini_client=gemini_client,
            )
            for index, image_url in enumerate(image_urls)
        ),
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
    return await bounded_media_step(
        step=_upload_each(
            platform=platform,
            post_url=post_url,
            image_urls=image_urls[:cap],
            gemini_client=gemini_client,
        ),
        subject=f"{platform} image",
        fallback="the text",
        degraded=[],
        url=post_url,
        timeout_seconds=LINK_MEDIA_TIMEOUT_SECONDS,
    )


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
