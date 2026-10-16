"""A test server's memory scope, and builders for its facts, member-alias rows and deltas."""

from datetime import datetime
from functools import partial

from discordbot.typings.memory import MemoryFact, MemoryOwner, MemoryDurability
from discordbot.services.memory.store import GLOBAL_COMPARTMENT, server_scope, read_memory_document

from tests.helpers.memory import STAMPED_AT, make_fact

GUILD_ID = 777
SERVER_SCOPE = server_scope(server_id=GUILD_ID)
SERVER_OWNER = MemoryOwner(owner_id=GUILD_ID, owner_name="My Server")
# An ordinary community fact in the single compartment a server scope ever has.
_server_fact = partial(
    make_fact,
    owner=SERVER_OWNER,
    summary="社群文化",
    section="culture",
    text="社群慣於高強度的粗口互嗆",
)


def _alias_fact(
    fact_id: str,
    text: str,
    subject_id: int,
    durability: MemoryDurability = "permanent",
    last_confirmed: datetime = STAMPED_AT,
) -> MemoryFact:
    """Builds one member-alias row, the server flavor's carve-out from no-individuals."""
    return _server_fact(
        fact_id=fact_id,
        section="member_alias",
        durability=durability,
        text=text,
        last_confirmed=last_confirmed,
        subject_id=subject_id,
    )


def _server_document() -> str:
    """Renders the server scope the way both the reply path and the cog read it."""
    return read_memory_document(
        scope=SERVER_SCOPE, compartments=[GLOBAL_COMPARTMENT], flavor="server"
    )
