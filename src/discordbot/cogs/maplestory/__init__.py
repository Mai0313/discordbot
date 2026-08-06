"""The MapleStory Artale lookup cog.

`cog.py` owns the Discord surface: the `/maplestory` group and its `monster` / `equip` /
`scroll` / `npc` / `quest` / `map` / `item` / `stats` subcommands. Beside it sit the read-only
search engine over the local JSON (`service.py`), the pydantic shapes those files validate into
(`models.py`), the embed builders (`embeds.py`) and the Chinese field templates they format
(`constants.py`), and the select menu that lets the user pick when a query matches more than one
entry (`views.py`).

The data lives in `./data/maplestory` and is maintained offline, so nothing here writes to it. A
missing or unparsable file leaves that category empty rather than failing the command, which is
why every lookup goes through `MapleStoryService` instead of touching the files directly.

This file stays import-free on purpose. `_load_cogs_sync` only ever imports
`discordbot.cogs.maplestory.cog`, and a re-exporting init would make any
`from discordbot.cogs.maplestory.service import ...` drag the whole Discord surface in as a side
effect of initialising the package.
"""
