"""Shared helpers for game view interactions."""

from collections.abc import Iterable

import logfire
from nextcord import Interaction, ui


async def send_ephemeral_notice(
    interaction: Interaction, content: str, log_message: str
) -> None:
    """Sends an ephemeral interaction notice with response/followup fallback."""
    try:
        if interaction.response.is_done():
            await interaction.followup.send(content=content, ephemeral=True)
            return
        await interaction.response.send_message(content=content, ephemeral=True)
    except Exception:
        logfire.warn(log_message, _exc_info=True)


def disable_view_components(
    children: Iterable[ui.Item], component_types: tuple[type[object], ...]
) -> None:
    """Disables view children matching any supplied component type."""
    for child in children:
        if isinstance(child, component_types):
            child.disabled = True
