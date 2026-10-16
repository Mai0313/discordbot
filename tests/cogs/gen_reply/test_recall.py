"""Tests for which memory a reply reads, and how it is framed."""

from pathlib import Path

import pytest

from discordbot.cogs.gen_reply.recall import (
    RecallContext,
    compartments_for_reading,
    render_server_memory_block,
)
from discordbot.services.memory.store import (
    DM_COMPARTMENT,
    GLOBAL_COMPARTMENT,
    user_scope,
    write_fact,
    guild_compartment,
    read_memory_document,
)

from tests.helpers.user_memory import _fact


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        (RecallContext(guild_id=111, dm_partner_id=None), ["global", "g/111"]),
        # A guild the owner has no memory in still reads only global; the join simply
        # names a directory that does not exist.
        (RecallContext(guild_id=999, dm_partner_id=None), ["global", "g/999"]),
        # A group DM, and a third party's memory read inside a 1:1 DM, both get the
        # cross-server compartment only.
        (RecallContext(guild_id=None, dm_partner_id=None), ["global"]),
        (RecallContext(guild_id=None, dm_partner_id=222), ["global"]),
    ],
)
def test_compartments_for_reading_matrix(
    memory_isolated_dir: Path, context: RecallContext, expected: list[str]
) -> None:
    """Each reading context resolves to exactly the directories it may open."""
    assert compartments_for_reading(owner_id=111, context=context) == expected


def test_the_owner_reads_everything_in_their_own_dm(memory_isolated_dir: Path) -> None:
    """A user's own information cannot leak to themselves, so their DM opens it all."""
    scope = user_scope(user_id=111)
    for compartment in (GLOBAL_COMPARTMENT, guild_compartment(guild_id=222), DM_COMPARTMENT):
        write_fact(scope=scope, fact=_fact(compartment=compartment))
    compartments = compartments_for_reading(
        owner_id=111, context=RecallContext(guild_id=None, dm_partner_id=111)
    )
    assert set(compartments) == {"global", "g/222", "dm"}


def test_the_owners_dm_renders_each_global_fact_once(memory_isolated_dir: Path) -> None:
    """The owner-DM read set must not name `global` twice.

    `list_compartments` already leads with it, and the reader concatenates whatever it is
    handed with no dedupe, so a prepended second copy renders every cross-server fact
    twice and charges it twice against the injection cap.
    """
    scope = user_scope(user_id=111)
    write_fact(scope=scope, fact=_fact(fact_id="a" * 16, text="全域事實"))
    write_fact(
        scope=scope, fact=_fact(fact_id="b" * 16, compartment=DM_COMPARTMENT, text="私訊事實")
    )
    compartments = compartments_for_reading(
        owner_id=111, context=RecallContext(guild_id=None, dm_partner_id=111)
    )
    assert compartments == list(dict.fromkeys(compartments))
    document = read_memory_document(scope=scope, compartments=compartments, flavor="user")
    assert document.count("全域事實") == 1
    assert document.count("私訊事實") == 1


def test_an_owner_with_no_stored_facts_still_reads_the_shared_compartment(
    memory_isolated_dir: Path,
) -> None:
    """A scope with no directories yet must still resolve to a readable compartment list."""
    assert compartments_for_reading(
        owner_id=111, context=RecallContext(guild_id=None, dm_partner_id=111)
    ) == [GLOBAL_COMPARTMENT]


def test_render_server_memory_block_is_low_authority_assistant_note() -> None:
    """A remembered community norm must not outrank the prompt or the current message."""
    block = render_server_memory_block(memory="## 伺服器輪廓\n這個社群很愛嘴")
    assert block["role"] == "assistant"
    content = block["content"]
    assert isinstance(content, str)
    assert "這個社群很愛嘴" in content
