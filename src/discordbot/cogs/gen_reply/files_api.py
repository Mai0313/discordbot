"""Gemini Files API upload for media the bot fetched itself (linked posts, generated clips).

The one supported way to let the answer model read a media file is to upload it here and
reference the resulting full URI as an `input_file` `file_id`. Handing the model a remote
http(s) URL instead looks equivalent but is not: the LiteLLM proxy rewrites any http-bearing
`file_id` / `file_url` into base64 `inline_data`, so the media starts counting against the
request body and a failed fetch is swallowed silently; and the native Interactions path,
which has no proxy in the loop at all, only resolves Files API URIs and YouTube links. The
Files-API URI has a dedicated pass-through branch on both paths and is never re-fetched.

The upload itself goes direct to Google (never through the proxy) because only the direct
client can poll a file to `FileState.ACTIVE`: the proxy's file resource reports a deprecated
`uploaded` status, and referencing a not-yet-active file intermittently 400s the whole answer
request.

`upload_file` and `poll_while_processing` are the upload and the activation poll every direct
upload is made of. They decide nothing: each raises the SDK's own error (the poll also a
`TimeoutError` for a read past the bound its caller set), and what a missing resource name, a
file still `FileState.PROCESSING` at the bound, or a failure costs is the caller's call.
`upload_until_active` serves a caller with no later reference to re-poll from: it raises unless
the file reaches `FileState.ACTIVE`, and that caller bounds the whole transfer and gives up.
"""

import io
import time
import asyncio
from pathlib import Path

from google import genai
import logfire
from google.genai.types import File, FileState
from openai.types.responses.response_input_file_param import ResponseInputFileParam

from discordbot.typings.llm import LLMConfig
from discordbot.utils.asyncio_locks import LoopLocalSemaphore

# The Files API refuses anything larger, so a caller that can measure a download up front
# (a Content-Length) aborts at this ceiling instead of spending its whole time budget
# fetching bytes Google would reject. It is the provider's limit, not a policy of ours.
FILES_API_MAX_BYTES = 2 * 1024**3

# Caps concurrent `upload_as_input_file` transfers across all in-flight pipelines. Deliberately
# not the attachment renderers' shared `media_semaphore` (`MEDIA_CONCURRENCY`): a linked video can
# hold its slot for minutes, which would starve the ordinary per-message attachment renders that
# share that pool. Small on purpose — these uploads are large and few.
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


async def upload_as_input_file(
    client: genai.Client,
    source: Path | bytes,
    mime_type: str,
    filename: str,
    timeout_seconds: float,
) -> ResponseInputFileParam | None:
    """Returns an `input_file` part referencing the uploaded `FileState.ACTIVE` URI, or None on failure.

    Best-effort and gated on `LLMConfig.file_api_enabled`; returns None when disabled, timed
    out, or failed so callers degrade without raising into the reply pipeline.

    Args:
        client (genai.Client): Direct-to-Google Gemini client.
        source (Path | bytes): Media bytes or path on disk.
        mime_type (str): Media MIME type passed to the upload.
        filename (str): Filename with its real extension (used by the Interactions path to
            classify the media kind).
        timeout_seconds (float): Timeout in seconds covering both the upload and activation poll.
    """
    if not LLMConfig().file_api_enabled:
        logfire.info("files api upload skipped by kill-switch", name=filename)
        return None
    started = time.monotonic()
    try:
        # The bound covers the transfer as well as the poll, and sits inside the slot on
        # purpose. google-genai disables the transport timeout by default (`timeout=None`), so
        # an upload into a black-holed connection never returns; bounding only the poll would
        # let two such uploads wedge both slots for the life of the process, after which every
        # upload behind them burns its full budget waiting here and silently degrades.
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
