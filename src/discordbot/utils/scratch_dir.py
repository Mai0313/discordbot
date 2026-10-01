"""A private scratch directory whose teardown can never speak for the work that used it.

A caller opens one around a worker `asyncio.to_thread` cannot cancel, which keeps fetching past
the caller's give-up, and a directory of its own is what keeps that overshoot from writing over a
concurrent request's files or piling up in the system temp dir. What the directory is FOR past
that depends on the writer, so do not read one into another. A writer that opens into a folder it
never rebuilds (`services/platforms/file_downloads.py::stream_to_file`) takes the removal as its
stop signal: its next open fails, and the file it has open stops at its next chunk. yt-dlp
re-creates its output dir per DASH format and so cannot; its worker is stopped with a
`threading.Event` and a bounded join instead
(`services/platforms/ytdlp.py::download_with_stop_signal`), and reaches the removal as a live
writer only in the case that logs, where the worker ignored that join window.

What that leaves everywhere is a teardown that can lose a race with a writer still running:
`shutil.rmtree` walks a tree something is adding to, and its closing `rmdir` raises `ENOTEMPTY`
for a file that arrived after the scan (a file that VANISHED under it is already absorbed, by
`TemporaryDirectory`'s own handler, which is why the cleanup below stays that class's). That
exception goes to whoever owned the `with` block, which is always the wrong reader: by then the
caller has told the user what happened, so a raised cleanup replaces a timeout's own report with
a generic failure, relabels a delivered file as undelivered, or escapes a listener entirely. So
the removal is reported here instead, and never travels.

That holds for the `gen_reply/link_sources/` builders too, although they degrade to a notice on
any failure. A cleanup raising while the post-route grace unwinds a build replaces its
`CancelledError`, which `asyncio.wait_for` needs to raise `TimeoutError`, so a link that never
answered would reach the model as one that answered without its media rather than as the
source's own timeout notice. And each builder RETURNS its result from inside the `with`, so a
cleanup raising over a finished one would discard it.
"""

from typing import TYPE_CHECKING
import tempfile
import contextlib

import logfire

if TYPE_CHECKING:
    from collections.abc import Generator


@contextlib.contextmanager
def scratch_directory(*, prefix: str) -> "Generator[str]":
    """Yields a private temp directory, removing it on the way out.

    Args:
        prefix: Names the directory after the call site that owns it, so a leaked one is
            attributable in the system temp dir. Keep it unique across call sites.

    Yields:
        The directory path, for a downloader's `output_folder`.
    """
    holder = tempfile.TemporaryDirectory(prefix=prefix)
    try:
        yield holder.name
    finally:
        try:
            holder.cleanup()
        except OSError as error:
            # Swallowed rather than raised (see the module docstring), but still an error: what
            # is left behind is deleted nowhere else, so it names an environment to look at.
            logfire.error(
                "Could not remove a scratch directory",
                directory=holder.name,
                error_type=type(error).__name__,
                _exc_info=error,
            )
