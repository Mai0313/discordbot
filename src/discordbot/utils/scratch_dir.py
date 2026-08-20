"""A private scratch directory whose teardown can never speak for the work that used it.

Four call sites open one: `/download_video`'s two branches, `parse_douyin` and `parse_threads`.
Each can give up on a worker `asyncio.to_thread` cannot cancel, so it keeps fetching past the
give-up, and a directory of its own is what keeps the overshoot from writing over a concurrent
request's files or piling up in the system temp dir. What the directory is FOR past that differs
per site, so do not read one into the others. `parse_douyin` and `parse_threads` make its removal
the stop signal itself — their writers open into a folder they never rebuild, so the next write
fails — while `/download_video`'s yt-dlp branch stops its worker with a `threading.Event` and a
bounded join (`utils/downloader.py::download_with_stop_signal`, which re-creates the output dir
per DASH format and so could not use the directory for this), and reaches the removal as a live
writer only in the case it logs, where the worker ignored that join window.

`gen_reply/link_sources/` abandons workers too and deliberately stays on plain
`tempfile.TemporaryDirectory`: those builders are best-effort, and a raised teardown there
degrades the link context they already degrade on any failure rather than reaching a user who
was told something else.

What that leaves everywhere is a teardown that can lose a race with a writer still running:
`shutil.rmtree` walks a tree something is adding to, and its closing `rmdir` raises `ENOTEMPTY`
for a file that arrived after the scan (a file that VANISHED under it is already absorbed, by
`TemporaryDirectory`'s own handler, which is why the cleanup below stays that class's). That
exception goes to whoever owned the `with` block, which is always the wrong reader: by then the
caller has told the user what happened, so a raised cleanup replaces a timeout's own report with
a generic failure, relabels a delivered file as undelivered, or escapes a listener entirely. So
the removal is reported here instead, and never travels.
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
