import re
import sys
from typing import TextIO
from pathlib import Path
from datetime import datetime
from importlib.metadata import version

import dotenv
import logfire

dotenv.load_dotenv()
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
__version__ = version("discordbot")


class _TeeStream:
    """A stream that writes to both a console and a file, stripping ANSI codes for the file."""

    def __init__(self, console: TextIO, file: TextIO) -> None:
        """Initialises the stream wrapper.

        Args:
            console: Stream that receives the original data.
            file: Stream that receives data with ANSI escape sequences removed.
        """
        self._console = console
        self._file = file

    def write(self, data: str) -> int:
        """Writes data to both streams and flushes them.

        Args:
            data: Text to write.

        Returns:
            The length of the original text.
        """
        self._console.write(data)
        self._file.write(_ANSI_ESCAPE.sub(repl="", string=data))
        self._console.flush()
        self._file.flush()
        return len(data)

    def flush(self) -> None:
        """Flushes both streams."""
        self._console.flush()
        self._file.flush()

    def isatty(self) -> bool:
        """Returns whether the console stream is attached to a TTY.

        Returns:
            True if the console stream reports that it is a TTY.
        """
        return self._console.isatty()


def setup_logging() -> None:
    """Configures logging with logfire, teeing output to a file."""
    started_at = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = Path(f"./data/logs/{started_at}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open(mode="a", encoding="utf-8")
    logfire.configure(
        send_to_logfire=False,
        scrubbing=False,
        # We can remove `console` if log is no longer needed to be saved in a file.
        console=logfire.ConsoleOptions(
            colors="auto",
            span_style="show-parents",
            include_timestamps=True,
            verbose=True,
            min_log_level="debug",
            output=_TeeStream(console=sys.stdout, file=log_file),
        ),
    )
