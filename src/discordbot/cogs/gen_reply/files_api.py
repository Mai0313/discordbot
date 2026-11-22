"""Direct-to-Google Gemini Files API uploads for bot-fetched media.

Uploads go directly to Google and are polled to `FileState.ACTIVE` before returning a full URI
for `input_file` `file_id` parts. `upload_file` and `poll_while_processing` are unguarded
primitives; `upload_until_active` requires `ACTIVE` or raises; `upload_as_input_file` is
best-effort and gated on `file_api_enabled`.
"""

import io
import time
import asyncio
from pathlib import Path

from google import genai
import logfire
from google.genai.types import File, FileState
from openai.types.responses.response_input_file_param import ResponseInputFileParam

from discordbot.utils.asyncio_locks import LoopLocalSemaphore

# Provider ceiling (2 GiB); callers with a Content-Length abort before downloading.
FILES_API_MAX_BYTES = 2 * 1024**3

# Separate from attachment renderers' media_semaphore so slow video uploads do not starve
# ordinary per-message attachment renders.
INPUT_FILE_UPLOAD_CONCURRENCY = 2

input_file_upload_semaphore = LoopLocalSemaphore(
    capacity_provider=lambda: INPUT_FILE_UPLOAD_CONCURRENCY
)


async def upload_file(
    client: genai.Client, source: Path | bytes, mime_type: str, display_name: str
) -> File:
    """Returns the `File` resource reported by starting one Files API upload.

    The returned file may still be in `FileState.PROCESSING`, and `name` / `uri` are optional.
    A `Path` streams from disk; `bytes` are wrapped in `io.BytesIO`.

    Args:
        client (genai.Client): Direct-to-Google Gemini client.
        source (Path | bytes): File path or in-memory bytes to upload.
        mime_type (str): MIME type sent in the upload config.
        display_name (str): Display name sent in the upload config.

    Raises:
        Exception: Whatever the SDK or its transport raised, unchanged.
    """
    upload_source = io.BytesIO(source) if isinstance(source, bytes) else source
    return await client.aio.files.upload(
        file=upload_source, config={"mime_type": mime_type, "display_name": display_name}
    )


async def poll_while_processing(  # noqa: PLR0913 -- each bound is its caller's own decision
    client: genai.Client,
    uploaded: File,
    name: str,
    poll_interval_seconds: float,
    timeout_seconds: float | None,
    read_timeout_seconds: float | None,
) -> File:
    """Returns `uploaded` once it leaves `FileState.PROCESSING` or `timeout_seconds` elapses.

    `timeout_seconds` is checked only between reads, so a caller without an outer timeout must
    also bound each read via `read_timeout_seconds`.

    Args:
        client (genai.Client): Client the file was uploaded with.
        uploaded (File): File resource as returned by `upload_file`.
        name (str): Resource name (`files/<id>`) used to re-read the file.
        poll_interval_seconds (float): Seconds to sleep between polls.
        timeout_seconds (float | None): Total polling budget in seconds, or None for no bound.
        read_timeout_seconds (float | None): Per-read timeout in seconds, or None for no bound.

    Raises:
        TimeoutError: One read did not return within `read_timeout_seconds`.
        Exception: Whatever the SDK or its transport raised, unchanged.
    """
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    while uploaded.state == FileState.PROCESSING:
        if deadline is not None and time.monotonic() >= deadline:
            return uploaded
        await asyncio.sleep(poll_interval_seconds)
        async with asyncio.timeout(delay=read_timeout_seconds):
            uploaded = await client.aio.files.get(name=name)
    return uploaded


async def upload_until_active(
    client: genai.Client, source: Path | bytes, mime_type: str, display_name: str
) -> str:
    """Returns the `FileState.ACTIVE` URI after uploading `source` and polling it out of `PROCESSING`.

    Applies no timeout of its own; the caller wraps the call in a timeout covering both the
    upload and the poll.

    Args:
        client (genai.Client): Direct-to-Google Gemini client.
        source (Path | bytes): File path or in-memory bytes to upload.
        mime_type (str): MIME type of the media.
        display_name (str): Display name for the uploaded resource.

    Raises:
        RuntimeError: The upload reported no resource name, or ended outside `FileState.ACTIVE`
            or without a URI.
        Exception: Whatever the SDK or its transport raised, unchanged.
    """
    uploaded = await upload_file(
        client=client, source=source, mime_type=mime_type, display_name=display_name
    )
    file_name = uploaded.name
    if file_name is None:
        raise RuntimeError(f"Files API upload of {display_name} returned no resource name")
    uploaded = await poll_while_processing(
        client=client,
        uploaded=uploaded,
        name=file_name,
        poll_interval_seconds=1.0,
        timeout_seconds=None,
        read_timeout_seconds=None,
    )
    if uploaded.state != FileState.ACTIVE or uploaded.uri is None:
        raise RuntimeError(f"Files API upload of {display_name} failed: state={uploaded.state}")
    return uploaded.uri


async def upload_as_input_file(  # noqa: PLR0913 -- client, source, metadata, timeout, and kill-switch
    client: genai.Client,
    source: Path | bytes,
    mime_type: str,
    filename: str,
    timeout_seconds: float,
    file_api_enabled: bool = True,
) -> ResponseInputFileParam | None:
    """Returns an `input_file` part referencing the uploaded `FileState.ACTIVE` URI, or None on failure.

    Best-effort and gated on `file_api_enabled`; returns None when disabled, timed out, or failed
    so callers degrade without raising into the reply pipeline.

    Args:
        client (genai.Client): Direct-to-Google Gemini client.
        source (Path | bytes): Media bytes or path on disk.
        mime_type (str): Media MIME type passed to the upload.
        filename (str): Filename with its real extension (used by the Interactions path to
            classify the media kind).
        timeout_seconds (float): Timeout in seconds covering both the upload and activation poll.
        file_api_enabled (bool): Whether Gemini Files API uploads are enabled.
    """
    if not file_api_enabled:
        logfire.info("files api upload skipped by kill-switch", name=filename)
        return None
    started = time.monotonic()
    try:
        # Bound sits inside the slot and covers upload + poll: google-genai sets no default
        # transport timeout, so an unbounded upload on a wedged connection holds a slot forever.
        async with input_file_upload_semaphore.get(), asyncio.timeout(delay=timeout_seconds):
            file_uri = await upload_until_active(
                client=client, source=source, mime_type=mime_type, display_name=filename
            )
    except TimeoutError as exc:
        logfire.warn(
            "files api upload did not finish in time",
            name=filename,
            timeout_seconds=timeout_seconds,
            _exc_info=exc,
        )
        return None
    except Exception as exc:
        logfire.warn(
            "files api upload failed", name=filename, error_type=type(exc).__name__, _exc_info=exc
        )
        return None
    logfire.debug(
        "files api upload done",
        name=filename,
        elapsed_seconds=time.monotonic() - started,
        file_uri=file_uri,
    )
    return ResponseInputFileParam(type="input_file", file_id=file_uri, filename=filename)
