"""Records what code under test hands the public-message cleanup, in place of deleting it."""

import pytest


class ScheduledDeletes:
    """What was handed to `schedule_public_message_delete`, recorded in place of a deletion."""

    def __init__(self) -> None:
        """Initializes one list per argument, in call order."""
        self.messages: list[object] = []
        self.user_names: list[str | None] = []
        self.interactions: list[object | None] = []

    def __call__(
        self,
        message: object,
        delay: float = 180,
        user_name: str | None = None,
        interaction: object | None = None,
    ) -> None:
        """Records one scheduled deletion."""
        del delay
        self.messages.append(message)
        self.user_names.append(user_name)
        self.interactions.append(interaction)


def record_scheduled_deletes(monkeypatch: pytest.MonkeyPatch) -> ScheduledDeletes:
    """Replaces the public-message cleanup with a recorder for the rest of the test.

    Every module that imports the scheduler holds its own reference, so each one is patched.
    """
    scheduled = ScheduledDeletes()
    for module in (
        "discordbot.cogs.economy.views",
        "discordbot.cogs.games.interactions",
        "discordbot.cogs.games.lobby",
        "discordbot.utils.interaction_responses",
    ):
        monkeypatch.setattr(f"{module}.schedule_public_message_delete", scheduled)
    return scheduled
