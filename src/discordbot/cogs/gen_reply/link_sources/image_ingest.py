"""Fetching a linked post's media and rendering it for the answer model to look at.

`render_linked_image` is the one-image step, delegating to the turn's `AttachmentRenderer` so
keyed Gemini and Claude turns upload via Files API while other or disabled-Files-API turns inline
as base64.
`ingest_post_images` runs it over a post's image URLs, and two properties are the whole point of
that. Every item is independent and best-effort, so one expired signed CDN url never costs the
rest; and the step is bounded rather than left to the caller's grace, so a slow fetch still
produces the honest text-only block instead of being cancelled with nothing to inject. A source
that keeps its own per-URL accounting, or downloads files of its own beside the images, runs the
one-image step itself.

`bounded_media_step` is that bound, and every source's media step runs under it, whether or not
its media is images.
"""

from typing import Any
import asyncio
from pathlib import Path
from mimetypes import guess_type
from collections.abc import Sequence, Coroutine

import logfire

from discordbot.typings.media import RenderedPart
from discordbot.typings.timeouts import LINK_MEDIA_DEGRADE_MARGIN_SECONDS
from discordbot.cogs.gen_reply.attachment.base import AttachmentSource, AttachmentRenderer


async def bounded_media_step[ResultT](  # noqa: PLR0913 -- the step, its log wording, its fallback value and its bound all vary per source
    step: Coroutine[Any, Any, ResultT],
    subject: str,
    fallback: str,
    degraded: ResultT,
    url: str,
    deadline: float,
    timeout_fields: dict[str, Any] | None = None,
) -> ResultT:
    """Runs a source's media step under `deadline`, returning `degraded` on timeout or error.

    Stops `LINK_MEDIA_DEGRADE_MARGIN_SECONDS` short of `deadline` so a slow media fetch still
    returns the text-only block instead of cancelling the whole context build.

    Args:
        step (Coroutine[Any, Any, ResultT]): Unawaited fetch-and-upload coroutine.
        subject (str): Label for log messages, such as `"Douyin media"`.
        fallback (str): Fallback description for log messages, such as `"the caption"`.
        degraded (ResultT): Value returned when no time remains or `step` times out or raises.
        url (str): Post URL associated with the media step.
        deadline (float): Event-loop timestamp at which the pipeline cancels the whole build.
        timeout_fields (dict[str, Any] | None): Extra structured fields attached to timeout logs.

    Returns:
        The awaited result of `step`, or `degraded`.
    """
    media_deadline = deadline - LINK_MEDIA_DEGRADE_MARGIN_SECONDS
    budget_seconds = media_deadline - asyncio.get_running_loop().time()
    if budget_seconds <= 0:
        # Never started rather than started and cancelled at once: an abandoned download can
        # wait on its worker's stop for most of the margin, which the read has already spent.
        step.close()
        logfire.warn(
            f"{subject} ingestion had no time left after the read; answering from {fallback}",
            url=url,
            timeout_seconds=budget_seconds,
            **(timeout_fields or {}),
        )
        return degraded
    try:
        async with asyncio.timeout_at(when=media_deadline):
            return await step
    except TimeoutError:
        logfire.warn(
            f"{subject} ingestion exceeded its bound; answering from {fallback}",
            url=url,
            timeout_seconds=budget_seconds,
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


async def render_linked_image(
    source: str | Path,
    filename: str,
    image_renderer: AttachmentRenderer,
    cache_key: str | None = None,
) -> RenderedPart | None:
    """Returns one fetched, downscaled, and rendered image part, or `None` when skipped.

    Delegates to `image_renderer.render_image` with `allow_dead_cache=False` so transient CDN
    failures on linked posts never poison the renderer's dead-source cache.

    Args:
        source (str | Path): Remote image URL or local file path to render.
        filename (str): Filename attached to the rendered part.
        image_renderer (AttachmentRenderer): Turn attachment renderer uploading or inlining images.
        cache_key (str | None): Stable key for upload/pending caches; defaults to `str(source)`.
    """
    rendered = await image_renderer.render_image(
        source=AttachmentSource(
            handle=source,
            kind="image",
            content_type=guess_type(filename)[0] or "image/jpeg",
            cache_key=cache_key or str(source),
            filename=filename,
        ),
        allow_dead_cache=False,
    )
    return rendered.part if rendered is not None else None


async def gather_media_parts(
    tasks: Sequence[Coroutine[Any, Any, RenderedPart | None]], warning: str, url: str
) -> list[RenderedPart]:
    """Runs `tasks` concurrently, returning rendered parts while logging per-item failures.

    Re-raises non-`Exception` `BaseException`s so cancellations and signals are never swallowed.

    Args:
        tasks (Sequence[Coroutine[Any, Any, RenderedPart | None]]): Per-item render/upload
            coroutines.
        warning (str): Log message recorded for each failing item.
        url (str): Post URL attached to failure logs.
    """
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException) and not isinstance(result, Exception):
            raise result
    parts: list[RenderedPart] = []
    for result in results:
        if isinstance(result, BaseException):
            logfire.warn(warning, url=url, error_type=type(result).__name__, _exc_info=result)
            continue
        if result is not None:
            parts.append(result)
    return parts


async def _render_each(
    platform: str, post_url: str, image_urls: list[str], image_renderer: AttachmentRenderer
) -> list[RenderedPart]:
    """Renders the images concurrently, keeping whatever succeeded."""
    tasks: list[Coroutine[Any, Any, RenderedPart | None]] = []
    for index, image_url in enumerate(image_urls):
        tasks.append(
            render_linked_image(
                source=image_url,
                filename=f"{platform.lower()}_image_{index}.jpg",
                image_renderer=image_renderer,
            )
        )
    return await gather_media_parts(
        tasks=tasks, warning=f"{platform} image ingestion failed for one item", url=post_url
    )


async def ingest_post_images(  # noqa: PLR0913 -- the post, its images and cap, the renderer and the build's deadline
    platform: str,
    post_url: str,
    image_urls: list[str],
    cap: int,
    image_renderer: AttachmentRenderer,
    deadline: float,
) -> list[RenderedPart]:
    """Returns up to `cap` rendered image parts for a post, degrading to `[]` on failure.

    Callers must count the returned parts rather than assuming `cap` succeeded so the rendered
    block never claims images that failed to render.

    Args:
        platform (str): Display name of the platform for log messages.
        post_url (str): Post URL the images belong to.
        image_urls (list[str]): Image URLs carried by the post, in page order.
        cap (int): Maximum number of images to fetch and render.
        image_renderer (AttachmentRenderer): Turn attachment renderer uploading or inlining images.
        deadline (float): Event-loop timestamp at which the pipeline cancels the build.
    """
    return await bounded_media_step(
        step=_render_each(
            platform=platform,
            post_url=post_url,
            image_urls=image_urls[:cap],
            image_renderer=image_renderer,
        ),
        subject=f"{platform} image",
        fallback="the text",
        degraded=[],
        url=post_url,
        deadline=deadline,
    )


def image_count_line(carried: int, attached: int, urls: Sequence[str] = ()) -> str:
    """Returns the rendered line stating how many of a post's `carried` images are `attached`.

    Separates `carried` from `attached` so a capped carousel or failed upload never leaves the
    block claiming unseen images are attached below.

    Args:
        carried (int): Total images the post carries.
        attached (int): Number of images actually attached to the input block.
        urls (Sequence[str]): Image URLs to list when `attached` is zero.
    """
    if attached:
        return f"The post carries {carried} image(s), {attached} of them attached below."
    if urls:
        return (
            f"The post carries {carried} image(s), none of them attached — URLs only: "
            + ", ".join(urls)
        )
    return f"The post carries {carried} image(s), none of them attached."
