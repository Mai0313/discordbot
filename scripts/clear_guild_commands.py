"""One-shot cleanup: bulk-delete all guild-scoped slash commands from given guilds.

Run this once after switching all slash commands to pure global registration so
that leftover guild-scoped entries (registered by an earlier `force_global=True`
+ `guild_ids=…` setup) are removed from Discord. Without this step those stale
entries stay forever — nextcord's sync only touches guild registries that the
current code still references via `guild_ids`.

Usage::

    uv run python scripts/clear_guild_commands.py
"""

import asyncio

import nextcord
from nextcord import Client, Intents

from discordbot.typings.config import DiscordConfig

# Guilds that previously received guild-scoped registrations via the old
# `FAST_SYNC_GUILD_IDS` constant.
GUILDS_TO_CLEAR: list[int] = [981592566208282634, 1143289646042853487]


async def main() -> None:
    """Connects with the bot token and wipes guild commands from each target guild."""
    cfg = DiscordConfig()
    client = Client(intents=Intents.default())

    @client.event
    async def on_ready() -> None:
        app_id = client.application_id
        if app_id is None:
            print("application_id is None; aborting")
            await client.close()
            return

        for guild_id in GUILDS_TO_CLEAR:
            try:
                existing = await client.http.get_guild_commands(app_id, guild_id)
                print(f"guild {guild_id}: {len(existing)} command(s) before cleanup")
                # Bulk overwrite with an empty list deletes every guild command.
                await client.http.bulk_upsert_guild_commands(app_id, guild_id, [])
                print(f"guild {guild_id}: cleared")
            except nextcord.HTTPException as exc:
                print(f"guild {guild_id}: failed - {exc}")

        await client.close()

    await client.start(token=cfg.discord_bot_token)


if __name__ == "__main__":
    asyncio.run(main=main())
