"""Gemini Files API upload for media the bot fetched itself (linked posts, generated clips).

The one supported way to let the answer model read a media file is to upload it here and
reference the resulting full uri as an `input_file` `file_id`. Handing the model a remote
http(s) URL instead looks equivalent but is not: the LiteLLM proxy rewrites any http-bearing
`file_id` / `file_url` into base64 `inline_data`, so the media starts counting against the
request body and a failed fetch is swallowed silently; and the native Interactions path,
which has no proxy in the loop at all, only resolves Files API uris and YouTube links. The
Files-API uri has a dedicated pass-through branch on both paths and is never re-fetched.

The upload itself goes DIRECT to Google (never through the proxy) because only the direct
client can poll a file to ACTIVE: the proxy's file resource reports a deprecated `uploaded`
status, and referencing a not-yet-ACTIVE file intermittently 400s the whole answer request.

`upload_file` and `poll_while_processing` are the upload and the activation poll every direct
upload is made of. They decide nothing: each raises the SDK's own error, and what a missing
resource name, a file still PROCESSING at the bound, or a failure costs is the caller's call.
`upload_to_files_api` serves the callers with no later reference to re-poll from (linked-post
media, a generated clip handed to its persona reply), so it bounds the whole transfer and gives up.
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

# Caps concurrent link-media uploads across all in-flight pipelines. Deliberately NOT the
# attachment renderers' shared `media_semaphore` (`MEDIA_CONCURRENCY`): a linked video can hold its
# slot for minutes, which would starve the ordinary per-message attachment renders that share
# that pool. Small on purpose — these uploads are large and few.
LINK_MEDIA_UPLOAD_CONCURRENCY = 2

link_media_upload_semaphore = LoopLocalSemaphore(
    capacity_provider=lambda: LINK_MEDIA_UPLOAD_CONCURRENCY
)


async def upload_file(
    *, client: genai.Client, source: Path | bytes, mime_type: str, display_name: str
) -> File:
    """Starts one Files API upload and returns the file as the upload reported it.

    The file may still be PROCESSING, and its `name` / `uri` may be None: the SDK types both as
    optional. A path is handed to the SDK untouched so it streams from disk; in-memory bytes get
    the file-like wrapper the SDK's `str | os.PathLike | io.IOBase` signature requires.

    Raises:
        Exception: Whatever the SDK or its transport raised, unchanged.
    """
    upload_source = io.BytesIO(source) if isinstance(source, bytes) else source
    return await client.aio.files.upload(
        file=upload_source, config={"mime_type": mime_type, "display_name": display_name}
    )


async def poll_while_processing(
    *,
    client: genai.Client,
    uploaded: File,
    name: str,
    poll_interval_seconds: float,
    timeout_seconds: float | None,
) -> File:
    """Re-reads an uploaded file until it leaves PROCESSING, returning the last state seen.

    The file comes back still PROCESSING only when `timeout_seconds`, counted from this call,
    elapsed first; None polls for as long as the file processes, for a caller that bounds the
    whole transfer itself.

    Args:
        client: The client the file was uploaded with (a file is readable only by that key).
        uploaded: The file as the upload returned it.
        name: The file's resource name (`files/<id>`), which the poll reads it back by.
        poll_interval_seconds: The wait between two reads.
        timeout_seconds: How long to keep polling, or None for no bound here.

    Raises:
        Exception: Whatever the SDK or its transport raised, unchanged.
    """
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    while uploaded.state == FileState.PROCESSING:
        if deadline is not None and time.monotonic() >= deadline:
            return uploaded
        await asyncio.sleep(poll_interval_seconds)
        uploaded = await client.aio.files.get(name=name)
    return uploaded


async def upload_to_files_api(
    *,
    client: genai.Client,
    source: Path | bytes,
    mime_type: str,
    display_name: str,
    timeout_seconds: float,
) -> str | None:
    """Uploads media to the Gemini Files API and returns its ACTIVE uri; None on any failure.

    Best-effort by design: every caller degrades rather than failing — the link builders to
    their text-only block, the generated-clip path by skipping its persona reply — so a
    failure here must not raise into the reply pipeline. The `file_api_enabled` kill-switch is
    checked here as a backstop, a switched-off upload taking the same path a failed one does.
    It saves no fetch: the media is fetched before this is called, so only a caller that checks
    the switch itself avoids that.

    `source` accepts a path as well as bytes (mirroring `MediaItem`) because the SDK's
    `files.upload` takes `str | os.PathLike | io.IOBase`: a clip already written to a temp
    file is streamed from disk rather than read whole into memory.

    Args:
        client: A Gemini client built with the Files API key (direct, never the proxy).
        source: The media bytes, or the path to the media file on disk.
        mime_type: The media's real MIME type; the upload needs it, the part does not carry one.
        display_name: Cosmetic name recorded on the uploaded file.
        timeout_seconds: Bound on the whole transfer, not just the activation poll.

    Returns:
        The full `https://.../files/<id>` uri, or None when the upload failed or never
        became ACTIVE in time.
    """
    if not LLMConfig().file_api_enabled:
        logfire.info("files api upload skipped by kill-switch", name=display_name)
        return None
    started = time.monotonic()
    try:
        # The bound covers the transfer as well as the poll, and sits INSIDE the slot on
        # purpose. google-genai disables the transport timeout by default (`timeout=None`), so
        # an upload into a black-holed connection never returns; bounding only the poll would
        # let two such uploads wedge both slots for the life of the process, after which every
        # link-media build burns its full budget waiting here and silently degrades to text.
        async with link_media_upload_semaphore.get(), asyncio.timeout(delay=timeout_seconds):
            uploaded = await upload_file(
                client=client, source=source, mime_type=mime_type, display_name=display_name
            )
            file_name = uploaded.name
            if file_name is None:
                logfire.warn("files api upload returned no resource name", name=display_name)
                return None
            uploaded = await poll_while_processing(
                client=client,
                uploaded=uploaded,
                name=file_name,
                poll_interval_seconds=1.0,
                timeout_seconds=None,
            )
    except TimeoutError as exc:
        logfire.warn(
            "files api upload did not finish in time",
            name=display_name,
            timeout_seconds=timeout_seconds,
            _exc_info=exc,
        )
        return None
    except Exception as exc:
        logfire.warn(
            "files api upload failed",
            name=display_name,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
        return None
    if uploaded.state != FileState.ACTIVE or uploaded.uri is None:
        logfire.warn(
            "files api upload reached a non-active state",
            name=display_name,
            state=str(uploaded.state),
        )
        return None
    logfire.debug(
        "files api upload done",
        name=display_name,
        elapsed_seconds=time.monotonic() - started,
        file_uri=uploaded.uri,
    )
    return uploaded.uri


async def upload_as_input_file(
    *,
    client: genai.Client,
    source: Path | bytes,
    mime_type: str,
    filename: str,
    timeout_seconds: float,
) -> ResponseInputFileParam | None:
    """Uploads media and wraps its uri as an `input_file` part; None when the upload failed.

    `filename` must carry the real extension: it is cosmetic on the proxied Responses path
    (the bridge drops it) but load-bearing on the native Interactions path, which classifies
    a part as video / audio / image / document purely by that extension.
    """
    file_uri = await upload_to_files_api(
        client=client,
        source=source,
        mime_type=mime_type,
        display_name=filename,
        timeout_seconds=timeout_seconds,
    )
    if file_uri is None:
        return None
    return ResponseInputFileParam(type="input_file", file_id=file_uri, filename=filename)
