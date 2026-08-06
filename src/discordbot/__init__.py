"""Package root: the import-time `.env` load, the distribution version, and logging setup.

Importing anything under `discordbot.` executes this module first, so the `dotenv.load_dotenv()`
below is the earliest one in the process. It is not what makes the settings classes work (each
config module under `typings/` repeats the idempotent call over the same file); it is what makes
the `LoggingConfig()` read here see `LOG_LEVEL` before any of them has been imported.

`setup_logging` is the only thing a caller uses: `cli.py::main` runs it before constructing the
bot, so every cog loaded afterwards logs through the configured logfire. logfire is used as a
purely local structured logger, with nothing exported off the process, and its console sink is
pointed at `_TeeStream`, the private wrapper that puts the same lines on stdout and in
`./data/logs/<start>.log`. `LoggingConfig` (`typings/config.py`) owns the one severity floor those
two sinks share; which severity a given failure earns is the ladder in
`.github/CONTRIBUTING.md#logging`.

`__version__` is read from the installed distribution metadata rather than written as a literal,
so it follows `pyproject.toml` with no second place to bump.
"""

import re
import sys
from typing import TextIO, cast
from pathlib import Path
from datetime import datetime
import warnings
from importlib.metadata import version

import dotenv
import logfire

from discordbot.typings.config import LoggingConfig

dotenv.load_dotenv()
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
__version__ = version("discordbot")


class _TeeStream:
    """A stream that writes to both a console and a file, stripping ANSI codes for the file.

    Implements only the `write` / `flush` / `isatty` subset that rich and logfire actually touch,
    so the call site has to cast it to `TextIO`; there is no real file object behind it.
    """

    def __init__(self, console: TextIO, file: TextIO) -> None:
        """Initialises the stream wrapper.

        Args:
            console (TextIO): Stream that receives the original data.
            file (TextIO): Stream that receives data with ANSI escape sequences removed.
        """
        self._console = console
        self._file = file

    def write(self, data: str) -> int:
        """Writes data to both streams and flushes them.

        Flushing on every write keeps the tail of `./data/logs` intact when the process dies
        without unwinding, which is the run a log file is read for.

        Args:
            data (str): Text to write.

        Returns:
            The length of the original text, which is what a `TextIO.write` caller counts; the
            file receives fewer characters once the escape sequences are stripped.
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

        Answers for the console rather than for the tee, because rich decides from this call
        whether it is writing to a terminal: a False here makes logfire drop rich and fall back
        to plain `print`, so stdout would lose its colors too. Colors reaching the tee is exactly
        why `write` has to strip them back out for the file half.

        Returns:
            True if the console stream reports that it is a TTY.
        """
        return self._console.isatty()


def setup_logging() -> None:
    """Configures logging with logfire, teeing output to a file.

    The floor is `LOG_LEVEL` (default `debug`), so `./data/logs` keeps the full trace
    a debugging session needs while a deployment that only wants outcomes can raise it
    without touching code. See the logging ladder in `.github/CONTRIBUTING.md`.

    The file is named from the clock at call time, so each run gets its own, and it is opened
    for append and deliberately never closed: the exporter writes to it until the process exits.
    Call this before anything that logs, `cli.py::main` does it first thing.
    """
    # LiteLLM forwards Responses API output as ResponseReasoningItem objects that
    # pydantic's discriminated-union serializer does not recognise, producing a
    # noisy multi-line UserWarning on every reply. The payload still streams fine.
    warnings.filterwarnings(
        action="ignore", message=r"Pydantic serializer warnings:", category=UserWarning
    )
    started_at = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = Path(f"./data/logs/{started_at}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open(mode="a", encoding="utf-8")
    logfire.configure(
        send_to_logfire=False,
        scrubbing=False,
        inspect_arguments=False,
        # The console sink is also the file sink: drop `console` and `./data/logs` goes with it.
        console=logfire.ConsoleOptions(
            colors="auto",
            span_style="show-parents",
            include_timestamps=True,
            verbose=True,
            min_log_level=LoggingConfig().log_level,
            output=cast("TextIO", _TeeStream(console=sys.stdout, file=log_file)),
        ),
    )
