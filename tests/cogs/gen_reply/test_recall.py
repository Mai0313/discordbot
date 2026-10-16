"""Tests for which memory a reply reads, and how it is framed."""

from typing import cast
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import nextcord
from nextcord import Message
from nextcord.enums import InteractionContextType

from discordbot.cogs.gen_reply.recall import (
    NO_STORED_MEMORY,
    RecallContext,
    RecallCandidate,
    build_recall_context,
    recall_user_memories,
    memory_lookup_credits,
    build_recall_allowlist,
    compartments_for_reading,
    render_server_memory_block,
    widen_allowlist_with_aliases,
)
from discordbot.services.memory.store import (
    DM_COMPARTMENT,
    GLOBAL_COMPARTMENT,
    user_scope,
    write_fact,
    guild_compartment,
    read_memory_document,
)
from discordbot.cogs.gen_reply.context import ReplyContextBuilder
from discordbot.cogs.gen_reply.surface import TurnSurface

from tests.helpers.casting import as_message
from tests.helpers.reply_fakes import FakeAuthor, FakeMessage, _cog, _seed_fact, _context_builder
from tests.helpers.user_memory import _fact
from tests.helpers.ask_interaction import ASKER_ID, GUILD_ID, _toolkit, _interaction


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


def _resolved_picks() -> asyncio.Future[list[str]]:
    """No recall picks, already resolved, as the pipeline hands them over once the route returns.

    Every direct `build` call takes one of these: `build` awaits the picks, and an unresolved
    future would hang the suite rather than fail it.
    """
    picks: asyncio.Future[list[str]] = asyncio.get_running_loop().create_future()
    picks.set_result([])
    return picks


def test_build_recall_allowlist_collects_authors_and_mentions_excluding_bot() -> None:
    """Trusted users are kept in order, deduped, and the bot is excluded."""
    author = FakeAuthor(user_id=1)
    mentioned = FakeAuthor(user_id=2)
    mentioned.name = "alice"
    mentioned.display_name = "Alice"
    bot = FakeAuthor(user_id=999)

    msg_with_mentions = FakeMessage(author=author)
    msg_with_mentions.mentions = [mentioned, bot]
    duplicate_author = FakeMessage(author=author)
    bot_authored = FakeMessage(author=bot)

    allowed = build_recall_allowlist(
        users=cast(
            "list[nextcord.Member | nextcord.User]",
            [
                msg_with_mentions.author,
                *msg_with_mentions.mentions,
                duplicate_author.author,
                bot_authored.author,
            ],
        ),
        bot_user_id=999,
    )

    # Insertion order preserved, bot (999) excluded from both author and mention slots.
    assert list(allowed.keys()) == [1, 2]
    # A participant carries the same label on both sides until aliases widen the prompt one.
    assert allowed[1] == RecallCandidate(
        prompt_label="Tester (tester)", credit_label="Tester (tester)"
    )
    assert allowed[2] == RecallCandidate(
        prompt_label="Alice (alice)", credit_label="Alice (alice)"
    )


def test_build_recall_allowlist_escapes_mention_labels() -> None:
    """Mention syntax in a display name is neutralized so a label cannot ping."""
    author = FakeAuthor(user_id=1)
    author.display_name = "@everyone"
    allowed = build_recall_allowlist(
        users=cast("list[nextcord.Member | nextcord.User]", [author]), bot_user_id=999
    )

    # The active @everyone is broken (zero-width space) while the text survives, on the
    # credit label too since that is the one a public footer renders.
    assert "@everyone" not in allowed[1].prompt_label
    assert "everyone" in allowed[1].prompt_label
    assert allowed[1].credit_label is not None
    assert "@everyone" not in allowed[1].credit_label


def test_recall_user_memories_enforces_allowlist() -> None:
    """Ids outside the allowlist drop, mention wrappers and dupes collapse, gaps signal clearly."""
    _seed_fact(scope=user_scope(user_id=1), text="甲的記憶")
    allowed = {
        1: RecallCandidate(prompt_label="A (a)", credit_label="A (a)"),
        2: RecallCandidate(prompt_label="B (b)", credit_label="B (b)"),
    }

    memories = recall_user_memories(
        user_id_list=["1", "<@1>", "3", "abc", "2"],
        allowed=allowed,
        context=RecallContext(guild_id=None, dm_partner_id=None),
    )

    by_id = {memory.user_id: memory for memory in memories}
    assert set(by_id) == {"1", "2"}
    assert "甲的記憶" in by_id["1"].memory
    assert by_id["1"].prompt_label == "A (a)"
    assert by_id["2"].memory == "(no stored memory for this user)"


def test_absent_member_is_counted_never_credited_by_id_or_the_alias_row() -> None:
    """A member named only by the nickname table is counted, not named and not id-dropped.

    The row is community prose the model reads; it can never be the public footer credit.
    Nothing else here can name them either: the guild member cache is empty for an absent
    member, and the identity the store stamps belongs to whichever guild's consolidation last
    wrote that fact, so it would put another server's nickname in this channel. A bare id would
    read as a memory the bot had just written rather than as a person it had read, so the
    count is what the footer gets.
    """
    _seed_fact(scope=user_scope(user_id=42), text="第三人的記憶")

    memories = recall_user_memories(
        user_id_list=["42"],
        allowed={42: RecallCandidate(prompt_label="Boss(社群暱稱:李董)")},
        context=RecallContext(guild_id=None, dm_partner_id=None),
    )

    footer_credits = memory_lookup_credits(memories=memories)
    assert footer_credits.named == ()
    assert footer_credits.unnamed == 1
    assert footer_credits.total == 1
    # The model still reads the row the credit refused.
    assert memories[0].prompt_label == "Boss(社群暱稱:李董)"


# One fact per compartment, so a document says by its body alone which directories the
# read opened.
_COMPARTMENT_FACTS = {
    GLOBAL_COMPARTMENT: "全域事實",
    guild_compartment(guild_id=111): "本群事實",
    guild_compartment(guild_id=222): "他群事實",
    DM_COMPARTMENT: "私訊事實",
}


@pytest.mark.parametrize(
    ("context", "compartments", "present", "absent"),
    [
        (
            RecallContext(guild_id=111, dm_partner_id=None),
            {"global", "g/111"},
            ["全域事實", "本群事實"],
            ["他群事實", "私訊事實"],
        ),
        (
            RecallContext(guild_id=222, dm_partner_id=None),
            {"global", "g/222"},
            ["全域事實", "他群事實"],
            ["本群事實", "私訊事實"],
        ),
        (
            RecallContext(guild_id=None, dm_partner_id=1),
            {"global", "g/111", "g/222", "dm"},
            ["全域事實", "本群事實", "他群事實", "私訊事實"],
            [],
        ),
        (
            RecallContext(guild_id=None, dm_partner_id=555),
            {"global"},
            ["全域事實"],
            ["本群事實", "他群事實", "私訊事實"],
        ),
        (
            RecallContext(guild_id=None, dm_partner_id=None),
            {"global"},
            ["全域事實"],
            ["本群事實", "他群事實", "私訊事實"],
        ),
    ],
    ids=["same-guild", "other-guild", "owner-own-dm", "other-owner-in-dm", "group-dm"],
)
def test_memory_read_opens_only_the_permitted_compartments(
    context: RecallContext, compartments: set[str], present: list[str], absent: list[str]
) -> None:
    """Where a reply happens decides which of an owner's compartments it may open.

    The cross-server boundary is a path join rather than a filter: a guild reads the
    shared compartment plus its own, a group DM and a third party's lookup inside a 1:1
    DM read the shared one alone, and the owner's own DM opens everything, since their
    own information cannot leak to themselves. Asserted end to end through
    `recall_user_memories`, the one call every reply path reads user memory through, so
    a compartment that is not listed is one whose facts never reach the model.
    """
    for compartment, text in _COMPARTMENT_FACTS.items():
        _seed_fact(scope=user_scope(user_id=1), text=text, compartment=compartment)

    assert set(compartments_for_reading(owner_id=1, context=context)) == compartments

    memories = recall_user_memories(
        user_id_list=["1"],
        allowed={1: RecallCandidate(prompt_label="A (a)", credit_label="A (a)")},
        context=context,
    )
    document = memories[0].memory
    for fragment in present:
        assert fragment in document
    for fragment in absent:
        assert fragment not in document


def _recall_context_for(message: Message) -> RecallContext:
    """The read context one gateway message produces, through the surface that decides it."""
    surface = TurnSurface.for_message(message=message)
    return build_recall_context(
        author_id=message.author.id,
        guild_id=surface.guild_id,
        is_direct_message=surface.is_direct_message,
    )


def test_build_recall_context_by_channel_kind() -> None:
    """Guild sets guild_id; a 1:1 DM sets dm_partner_id; a guildless non-DM channel sets neither."""
    guild_message = FakeMessage(content="hi")
    guild_context = _recall_context_for(message=as_message(fake=guild_message))
    assert guild_context.guild_id == 1
    assert guild_context.dm_partner_id is None

    dm_message = FakeMessage(content="hi", author=FakeAuthor(user_id=7))
    dm_message.guild = None
    dm_message.channel = MagicMock(spec=nextcord.DMChannel)
    dm_context = _recall_context_for(message=as_message(fake=dm_message))
    assert dm_context.guild_id is None
    assert dm_context.dm_partner_id == 7

    # A group DM has no guild but is not a DMChannel, so it fail-closes to neither.
    group_message = FakeMessage(content="hi")
    group_message.guild = None
    group_context = _recall_context_for(message=as_message(fake=group_message))
    assert group_context.guild_id is None
    assert group_context.dm_partner_id is None


def test_recall_user_memories_fully_locked_reads_as_no_memory() -> None:
    """A memory stored only in another guild resolves to the no-memory signal, uncredited."""
    _seed_fact(
        scope=user_scope(user_id=1),
        text="他群祕密",
        compartment=guild_compartment(guild_id=424242),
        section="permanent",
        durability="permanent",
    )

    memories = recall_user_memories(
        user_id_list=["1"],
        allowed={1: RecallCandidate(prompt_label="A (a)", credit_label="A (a)")},
        context=RecallContext(guild_id=111, dm_partner_id=None),
    )

    assert [memory.memory for memory in memories] == [NO_STORED_MEMORY]
    assert memory_lookup_credits(memories=memories).total == 0


def test_widen_allowlist_with_aliases_merges_participant_labels() -> None:
    """A participant keeps their label and gains aliases."""
    memory = (
        "## 成員稱呼\n* Mai(社群暱稱:李董、破貓親爹)[id: 123]\n* Bob(社群暱稱:阿伯)[id: 456]\n"
    )
    allowed = {123: RecallCandidate(prompt_label="Mai (mai9999)", credit_label="Mai (mai9999)")}
    widen_allowlist_with_aliases(allowed=allowed, memory=memory)

    # The conversation label leads and the table row rides behind it on the same line.
    assert allowed[123].prompt_label.startswith("Mai (mai9999)")
    assert "李董" in allowed[123].prompt_label
    # The footer credit stays the short Discord label; the row never reaches it.
    assert allowed[123].credit_label == "Mai (mai9999)"


def test_widen_allowlist_with_aliases_skips_absent_members() -> None:
    """Participants are enriched but absent members stay out.

    Naming a public nickname must not open an absent member's personal memory, even though
    the nickname table itself is public content.
    """
    memory = (
        "## 成員稱呼\n* Mai(社群暱稱:李董、破貓親爹)[id: 123]\n* Bob(社群暱稱:阿伯)[id: 456]\n"
    )
    allowed = {123: RecallCandidate(prompt_label="Mai (mai9999)", credit_label="Mai (mai9999)")}
    widen_allowlist_with_aliases(allowed=allowed, memory=memory)

    # The present participant is still enriched with community aliases.
    assert allowed[123].prompt_label.startswith("Mai (mai9999)")
    assert "李董" in allowed[123].prompt_label
    # The absent member is not added, so their personal memory stays unreachable here.
    assert 456 not in allowed


async def test_deterministic_memory_lookup_skips_locked_author_memory() -> None:
    """Deterministic lookup injects nothing when memory lives in another guild.

    The direct path opens exactly the compartments the optional lookup does, so a
    pick cannot reach a directory the resolver would not have opened.
    """
    cog = _cog()
    _seed_fact(
        scope=user_scope(user_id=1),
        text="他群祕密",
        compartment=guild_compartment(guild_id=424242),
        section="permanent",
        durability="permanent",
    )
    builder = _context_builder(
        cog=cog,
        message=as_message(fake=FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))),
    )

    context = await builder.build(
        history_limit=2,
        parts_task=asyncio.create_task(coro=builder.render_parts()),
        recall=builder.plan_recall(),
        recall_picks=_resolved_picks(),
    )

    assert context.memory_block is None
    assert context.memory_credits.total == 0


@pytest.mark.usefixtures("memory_isolated_dir")
@pytest.mark.parametrize(
    ("context", "guild_id", "expected"),
    [
        (
            InteractionContextType.guild,
            GUILD_ID,
            RecallContext(guild_id=GUILD_ID, dm_partner_id=None),
        ),
        (
            InteractionContextType.bot_dm,
            None,
            RecallContext(guild_id=None, dm_partner_id=ASKER_ID),
        ),
        (
            InteractionContextType.private_channel,
            None,
            RecallContext(guild_id=None, dm_partner_id=None),
        ),
    ],
)
def test_an_ask_turn_reads_memory_scoped_to_where_it_happens(
    context: InteractionContextType, guild_id: int | None, expected: RecallContext
) -> None:
    """The recall plan takes both facts off the surface, since the message carries neither.

    `Message.guild` is None on this route whatever the interaction says, so a plan reading it
    would take a group DM for the asker's own DM, handing a channel full of strangers every
    compartment the asker has, and would drop a server turn's guild compartment.
    """
    interaction = _interaction(guild_id=guild_id, context=context)
    builder = ReplyContextBuilder(
        toolkit=_toolkit(interaction=interaction),
        surface=TurnSurface.for_interaction(interaction=interaction, question="在幹嘛"),
    )

    assert builder.plan_recall().recall_context == expected
