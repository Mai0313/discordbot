"""Package root: loads `.env` on import, and defines the console + file logging setup."""

import re
import sys
from typing import TextIO, cast
from pathlib import Path
from datetime import datetime
import warnings
from importlib.metadata import version

import dotenv
import logfire

from discordbot.typings.config import load_bot_config

dotenv.load_dotenv()
_ANSI_ESCAPE = re.compile(pattern=r"\x1b\[[0-9;]*m")
__version__ = version(distribution_name="discordbot")


class _TeeStream:
    """A stream that writes to both a console and a file, stripping ANSI codes for the file."""

    def __init__(self, console: TextIO, file: TextIO) -> None:
        """Initialises the stream wrapper."""
        self._console = console
        self._file = file

    def write(self, data: str) -> int:
        """Writes `data` to both streams, flushes them, and returns `len(data)`."""
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
        """Returns whether the console stream is attached to a TTY."""
        return self._console.isatty()


def setup_logging() -> None:
    """Configures logfire logging to stdout and `./data/logs/<timestamp>.log`.

    Applies `[global.logging].level` (default `debug`) as the minimum console/file level and
    suppresses pydantic serializer warnings from Responses API payloads.
    """
    # LiteLLM forwards Responses API reasoning items that pydantic's union serializer does
    # not recognise, emitting a noisy UserWarning per reply while streaming works fine.
    warnings.filterwarnings(
        action="ignore", message=r"Pydantic serializer warnings:", category=UserWarning
    )
    started_at = datetime.now().strftime(format="%Y-%m-%d_%H-%M-%S")
    log_path = Path(f"./data/logs/{started_at}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open(mode="a", encoding="utf-8")
    logfire.configure(
        send_to_logfire=False,
        scrubbing=False,
        inspect_arguments=False,
        console=logfire.ConsoleOptions(
            colors="auto",
            span_style="show-parents",
            include_timestamps=True,
            verbose=True,
            min_log_level=load_bot_config().global_config.logging.level,
            output=cast("TextIO", _TeeStream(console=sys.stdout, file=log_file)),
        ),
    )
