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

Distinct from `attachment/gemini_file_api.py`, which owns the same upload for Discord
attachments plus a per-source render cache and a pending-upload re-poll keyed on the message
that carried them. A file this module uploads has no such later reference to adopt, so it
gets a plain bounded wait instead.
"""

import io
import time
import asyncio
from pathlib import Path

from google import genai
import logfire
from google.genai.types import FileState
from openai.types.responses.response_input_file_param import ResponseInputFileParam

from discordbot.utils.asyncio_locks import LoopLocalSemaphore

# The Files API refuses anything larger, so a caller that can measure a download up front
# (a Content-Length) aborts at this ceiling instead of spending its whole time budget
# fetching bytes Google would reject. It is the provider's limit, not a policy of ours.
FILES_API_MAX_BYTES = 2 * 1024**3

# Bound on the whole fetch + upload step for the media of a linked post, shared by every link
# context builder. It exists so the builder always returns within the pipeline's post-route
# grace and degrades to text itself, rather than being cancelled with nothing to show. Set
# well above a normal clip's cost: watching the linked video is the point, and the text block
# is already on hand, so waiting is cheaper than answering blind.
LINK_MEDIA_TIMEOUT_SECONDS = 170.0

# Caps concurrent link-media uploads across all in-flight pipelines. Deliberately NOT the
# attachment renderer's `_media_semaphore` (`MEDIA_CONCURRENCY`): a linked video can hold its
# slot for minutes, which would starve the ordinary per-message attachment renders that share
# that pool. Small on purpose — these uploads are large and few.
LINK_MEDIA_UPLOAD_CONCURRENCY = 2

link_media_upload_semaphore = LoopLocalSemaphore(
    capacity_provider=lambda: LINK_MEDIA_UPLOAD_CONCURRENCY
)


async def upload_to_files_api(
    *,
    client: genai.Client,
    source: Path | bytes,
    mime_type: str,
    display_name: str,
    timeout_seconds: float,
) -> str | None:
    """Uploads media to the Gemini Files API and returns its ACTIVE uri; None on any failure.

    Best-effort by design: every caller has a text-only degradation to fall back on, so a
    failure here must not raise into the reply pipeline.

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
    started = time.monotonic()
    # A path is handed to the SDK untouched so it streams from disk; only in-memory bytes need
    # the file-like wrapper the SDK's signature requires.
    upload_source = io.BytesIO(source) if isinstance(source, bytes) else source
    try:
        # The bound covers the transfer as well as the poll, and sits INSIDE the slot on
        # purpose. google-genai disables the transport timeout by default (`timeout=None`), so
        # an upload into a black-holed connection never returns; bounding only the poll would
        # let two such uploads wedge both slots for the life of the process, after which every
        # link-media build burns its full budget waiting here and silently degrades to text.
        async with link_media_upload_semaphore.get(), asyncio.timeout(delay=timeout_seconds):
            uploaded = await client.aio.files.upload(
                file=upload_source, config={"mime_type": mime_type, "display_name": display_name}
            )
            file_name = uploaded.name
            if file_name is None:
                logfire.warn("files api upload returned no resource name", name=display_name)
                return None
            while uploaded.state == FileState.PROCESSING:
                await asyncio.sleep(1.0)
                uploaded = await client.aio.files.get(name=file_name)
    except TimeoutError:
        logfire.warn(
            "files api upload did not finish in time",
            name=display_name,
            timeout_seconds=timeout_seconds,
        )
        return None
    except Exception:
        logfire.warn("files api upload failed", name=display_name, _exc_info=True)
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
