"""Environment-backed settings for the parts of the bot that no feature module owns.

Three `BaseSettings` classes, each instantiated at the point of use rather than passed around:
`DiscordConfig` carries the gateway token `cli.py` boots with, `LoggingConfig` the severity floor
`discordbot/__init__.py` hands logfire before any cog loads, and `EconomyConfig` the one
central-bank escape hatch `cogs/economy/cog.py` reads when it builds a loan-approval view.

The settings that DO belong to a feature live with it instead, so this file is not the place to
look for every env name: `typings/llm.py` holds `LLMConfig` and the model kill-switches,
`typings/douyin.py` the Douyin auto-expansion switch (deliberately apart from `LLMConfig`, since
expansion is not an LLM feature), `typings/memory.py` the memory git-history switch
(`MEMORY_GIT_ENABLED`, its only field; the memory thresholds are plain constants in
`services/memory/constants.py` and are not environment-backed at all), and
`utils/media_delivery.py` / `utils/usage_log.py` keep theirs beside the helper that reads them.
`EconomyConfig` is the standing exception to that rule: it is an economy setting, read only by
`cogs/economy/cog.py`, and `typings/economy.py` does exist, but that module is pure vocabulary
and tuning constants with no environment-backed settings of its own, so the feature's single env
switch sits here beside the unowned ones.

Per the repository convention every field names its variable explicitly through
`validation_alias=AliasChoices("ENV_NAME")` rather than letting pydantic infer it from the
attribute, and `dotenv.load_dotenv()` runs at import, so whatever it finds is already in
`os.environ` by the time anything constructs one of these. Given no path it searches through
`find_dotenv`, which walks up from THIS module's own directory rather than from the working
directory (the cwd is used only in a REPL, under a debugger, or in a frozen build); that is why a
git worktree carrying no `.env` of its own still picks up the parent checkout's. `.env.example`
carries the same names with the wording an operator reads, bar
`ECONOMY_ALLOW_CENTRAL_BANK_SELF_APPROVAL`, which is deliberately absent from it because that
switch is a local-testing convenience meant to stay unset in production.
"""

from typing import Literal

import dotenv
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings

dotenv.load_dotenv()


class DiscordConfig(BaseSettings):
    """Configuration settings for the Discord bot, reading from environment variables.

    The token has no default, so a deployment missing `DISCORD_BOT_TOKEN` fails with a pydantic
    `ValidationError` in `cli.py::main`, at the first `DiscordConfig()`, before the bot object is
    built, before any cog loads, and so before the gateway is contacted.

    Attributes:
        discord_bot_token: The authentication token for the Discord bot, read only where `main`
            calls `bot.run`. `DiscordBot.__init__` builds a second config of its own and keeps it
            as `self.discord_config`, but nothing in the tree reads that attribute.
    """

    discord_bot_token: str = Field(
        ...,
        description="The token from discord for calling models.",
        examples=["MTEz-..."],
        validation_alias=AliasChoices("DISCORD_BOT_TOKEN"),
    )


class EconomyConfig(BaseSettings):
    """Economy feature settings loaded from environment variables.

    Attributes:
        allow_central_bank_self_approval: Lifts the one check in `accept_loan_proposal` that
            stops a central banker approving a central-bank request they filed themselves. It is
            a local-testing convenience and is meant to stay false in production; the separate
            `is_central_banker` requirement is unaffected either way.
    """

    allow_central_bank_self_approval: bool = Field(
        False,
        description="Allow central-bank borrowers to approve their own loan requests for local testing.",
        examples=[False],
        validation_alias=AliasChoices("ECONOMY_ALLOW_CENTRAL_BANK_SELF_APPROVAL"),
    )


class LoggingConfig(BaseSettings):
    """Console and log-file verbosity, loaded from environment variables.

    The console sink writes to a tee of stdout and `./data/logs/<start>.log`, so this one floor
    gates both; there is no separate file level.

    Attributes:
        log_level: The lowest severity logfire emits. The members are logfire's own level names
            rather than a hand-picked subset, and `tests/test_cogs_smoke.py` compares them
            against its `LEVEL_NUMBERS` table, so a name logfire renames or drops breaks that
            test rather than the console configuration at boot.
    """

    log_level: Literal["trace", "debug", "info", "notice", "warn", "warning", "error", "fatal"] = (
        Field(
            "debug",
            description="Lowest severity written to the console and to ./data/logs. Defaults to debug so the log file keeps the full trace; raise it to info on a deployment that only wants outcomes.",
            examples=["debug", "info"],
            validation_alias=AliasChoices("LOG_LEVEL"),
        )
    )


__all__ = ["DiscordConfig", "EconomyConfig", "LoggingConfig"]
