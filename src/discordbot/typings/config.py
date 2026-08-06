from typing import Literal

import dotenv
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings

dotenv.load_dotenv()


class DiscordConfig(BaseSettings):
    """Configuration settings for the Discord bot, reading from environment variables.

    Attributes:
        discord_bot_token: The authentication token for the Discord bot.
    """

    discord_bot_token: str = Field(
        ...,
        description="The token from discord for calling models.",
        examples=["MTEz-..."],
        validation_alias=AliasChoices("DISCORD_BOT_TOKEN"),
    )


class EconomyConfig(BaseSettings):
    """Economy feature settings loaded from environment variables."""

    allow_central_bank_self_approval: bool = Field(
        False,
        description="Allow central-bank borrowers to approve their own loan requests for local testing.",
        examples=[False],
        validation_alias=AliasChoices("ECONOMY_ALLOW_CENTRAL_BANK_SELF_APPROVAL"),
    )


class FeedbackConfig(BaseSettings):
    """User-report settings loaded from environment variables.

    Attributes:
        enabled: Kill-switch for the whole `/feedback` command.
        github_token: Token used to open and read this repository's issues.
        github_repository: The `owner/name` the reports are filed against.
        contact: Where to send people when reports cannot be filed at all.
        max_open_reports: How many unresolved reports one person may hold at once.
        submit_cooldown_seconds: Minimum gap between one person's submissions.
    """

    enabled: bool = Field(
        default=True,
        description="Whether the /feedback command accepts and shows user reports.",
        examples=[True],
        validation_alias=AliasChoices("FEEDBACK_ENABLED"),
    )
    github_token: str = Field(
        default="",
        description="Token with issue read/write on the reporting repository.",
        examples=["github_pat_..."],
        # Prefixed, unlike a provider credential: GitHub Actions always exports
        # GITHUB_REPOSITORY and `gh` users often export GITHUB_TOKEN, and `load_dotenv`
        # does not override an existing variable, so the bare names would silently lose
        # to whatever the surrounding shell happened to have set.
        validation_alias=AliasChoices("FEEDBACK_GITHUB_TOKEN"),
    )
    github_repository: str = Field(
        default="",
        description="The owner/name repository that user reports become issues on.",
        examples=["Mai0313/discordbot"],
        validation_alias=AliasChoices("FEEDBACK_GITHUB_REPOSITORY"),
    )
    contact: str = Field(
        default="",
        description="Contact shown when reports cannot be filed, e.g. a Discord handle.",
        examples=["mai9999"],
        validation_alias=AliasChoices("FEEDBACK_CONTACT"),
    )
    max_open_reports: int = Field(
        default=3,
        description="How many still-open reports one person may hold before being asked to wait.",
        examples=[3],
        validation_alias=AliasChoices("FEEDBACK_MAX_OPEN_REPORTS"),
    )
    submit_cooldown_seconds: int = Field(
        default=300,
        description="Minimum seconds between two submissions from the same person.",
        examples=[300],
        validation_alias=AliasChoices("FEEDBACK_SUBMIT_COOLDOWN_SECONDS"),
    )

    @property
    def github_ready(self) -> bool:
        """Whether an issue can be opened right now.

        Not the same question as whether the feature is on, which is what `enabled`
        answers. A missing token is an operational state, not a switch: reports are
        still taken and stored, and the retry sweep files them once one is configured.
        """
        return self.enabled and bool(self.github_token.strip()) and bool(self.repository_slug)

    @property
    def repository_slug(self) -> str:
        """The trimmed `owner/name` slug, empty when it is not a usable pair."""
        slug = self.github_repository.strip().strip("/")
        owner, _, name = slug.partition("/")
        return slug if owner and name else ""


class LoggingConfig(BaseSettings):
    """Console and log-file verbosity, loaded from environment variables."""

    log_level: Literal["trace", "debug", "info", "notice", "warn", "warning", "error", "fatal"] = (
        Field(
            "debug",
            description="Lowest severity written to the console and to ./data/logs. Defaults to debug so the log file keeps the full trace; raise it to info on a deployment that only wants outcomes.",
            examples=["debug", "info"],
            validation_alias=AliasChoices("LOG_LEVEL"),
        )
    )


__all__ = ["DiscordConfig", "EconomyConfig", "FeedbackConfig", "LoggingConfig"]
