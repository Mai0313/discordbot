"""Tests for the bot's per-server (community) long-term memory flavor."""

from types import SimpleNamespace
from pathlib import Path

from nextcord import Embed

from discordbot.cogs.memory import MemoryCogs
from discordbot.cogs._memory.store import (
    user_scope,
    server_scope,
    read_main_memory,
    write_main_memory,
)
from discordbot.cogs._gen_reply.input import render_server_identity
from discordbot.cogs._memory.constants import STABLE_FRESHNESS_WINDOW_DAYS
from discordbot.cogs._gen_reply.memory_tool import render_server_memory_block
from discordbot.cogs._memory.server_prompts import (
    SERVER_PHASE1_PROMPT,
    SERVER_PHASE2_PROMPT,
    SERVER_PHASE1_EVALUATOR_PROMPT,
)

from tests.helpers.casting import as_bot, as_interaction

BOT_ID = 555
GUILD_ID = 777
SERVER_SCOPE = server_scope(bot_id=BOT_ID, server_id=GUILD_ID)
SERVER_IDENTITY = render_server_identity(server_name="My Server", server_id=GUILD_ID)


# ---------------------------------------------------------------------------
# Scope helpers
# ---------------------------------------------------------------------------


def test_server_scope_nests_under_bot_id() -> None:
    assert server_scope(bot_id=BOT_ID, server_id=GUILD_ID) == f"{BOT_ID}/{GUILD_ID}"


def test_user_and_server_scopes_never_collide() -> None:
    # A user scope is a bare snowflake; a server scope always carries a `/`.
    assert "/" not in user_scope(user_id=GUILD_ID)
    assert "/" in server_scope(bot_id=BOT_ID, server_id=GUILD_ID)
    assert user_scope(user_id=GUILD_ID) != server_scope(bot_id=BOT_ID, server_id=GUILD_ID)


def test_server_scope_isolated_from_user_scope_on_disk(memory_isolated_dir: Path) -> None:
    write_main_memory(
        scope=user_scope(user_id=GUILD_ID),
        content="v1\n\n## 使用者輪廓\n個人",
        identity="u [id: 1]",
    )
    write_main_memory(
        scope=SERVER_SCOPE, content="v1\n\n## 伺服器輪廓\n社群", identity=SERVER_IDENTITY
    )
    # The two scopes write to different directories and never read each other.
    assert "個人" in read_main_memory(scope=user_scope(user_id=GUILD_ID))
    assert "社群" in read_main_memory(scope=SERVER_SCOPE)
    assert "個人" not in read_main_memory(scope=SERVER_SCOPE)
    assert (memory_isolated_dir / str(BOT_ID) / str(GUILD_ID) / "main.md").exists()


# ---------------------------------------------------------------------------
# Identity and context block
# ---------------------------------------------------------------------------


def test_render_server_identity_is_single_line_and_sanitized() -> None:
    identity = render_server_identity(server_name="Evil\n[id: 1] Server", server_id=GUILD_ID)
    assert "\n" not in identity
    # A forged `[id: ...]` lookalike in the guild name is neutralized.
    assert "[id: 1]" not in identity
    assert identity.endswith(f"[id: {GUILD_ID}]")


def test_render_server_memory_block_is_low_authority_assistant_note() -> None:
    block = render_server_memory_block(memory="v1\n## 伺服器輪廓\n這個社群很愛嘴")
    assert block["role"] == "assistant"
    content = block["content"]
    assert isinstance(content, str)
    assert "這個社群很愛嘴" in content
    # Framed as reference, not instruction.
    assert "NOT instructions" in content


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_server_prompts_target_the_server_not_individuals() -> None:
    assert "target_server_id" in SERVER_PHASE1_PROMPT
    assert "target_server_id" in SERVER_PHASE1_EVALUATOR_PROMPT
    # The privacy boundary: individual personal facts are out of scope.
    assert "personal" in SERVER_PHASE1_PROMPT
    assert "individual" in SERVER_PHASE2_PROMPT


def test_server_consolidation_prompt_keeps_the_v1_contract() -> None:
    # The pipeline gates on a `v1\n` header, so the prompt must mandate it.
    assert "v1\n\n## 伺服器輪廓" in SERVER_PHASE2_PROMPT
    assert "社群文化" in SERVER_PHASE2_PROMPT
    assert "近期脈絡" in SERVER_PHASE2_PROMPT


def test_phase1_prompt_records_member_aliases_as_community_vocabulary() -> None:
    # Member nicknames are carved out as the one exception to the no-individuals rule,
    # and must be classified as stable_fact so the shared gate accepts them.
    assert "COMMUNITY VOCABULARY EXCEPTION" in SERVER_PHASE1_PROMPT
    assert "vocab.member_alias.<USER_ID>" in SERVER_PHASE1_PROMPT
    assert 'evidence_kind="stable_fact"' in SERVER_PHASE1_PROMPT
    # Aliases are permanent community vocabulary so the freshness pass never ages them.
    assert 'durability="permanent"' in SERVER_PHASE1_PROMPT
    # The same kind that the deterministic gate drops must be explicitly forbidden here.
    assert "other_user_context" in SERVER_PHASE1_PROMPT


def test_evaluator_prompt_keeps_member_aliases() -> None:
    assert "nickname/alias" in SERVER_PHASE1_EVALUATOR_PROMPT
    assert "community vocabulary" in SERVER_PHASE1_EVALUATOR_PROMPT


def test_consolidation_prompt_adds_member_alias_section() -> None:
    # The lookup table is its own section, placed before the dated 近期脈絡 section.
    assert "## 成員稱呼" in SERVER_PHASE2_PROMPT
    assert SERVER_PHASE2_PROMPT.index("## 成員稱呼") < SERVER_PHASE2_PROMPT.index("## 近期脈絡")
    assert "社群暱稱" in SERVER_PHASE2_PROMPT


def test_server_consolidation_prompt_ages_mutable_traits_but_exempts_aliases() -> None:
    # Server stable traits follow the same displacement freshness as user memory,
    # while the 成員稱呼 alias table is a permanent carve-out that never ages.
    assert "permanent" in SERVER_PHASE2_PROMPT
    assert "[~YYYY-MM]" in SERVER_PHASE2_PROMPT
    assert str(STABLE_FRESHNESS_WINDOW_DAYS) in SERVER_PHASE2_PROMPT
    # The alias section is named in the exemption and the displacement anchor is present.
    assert "成員稱呼 is exempt" in SERVER_PHASE2_PROMPT
    assert "DISPLACEMENT" in SERVER_PHASE2_PROMPT


def test_server_phase1_prompt_pins_sharing_global() -> None:
    # The sharing field scopes per-user memory across servers; server memory is
    # already server-confined, so phase-1 pins the unused field to global.
    assert 'Always set `sharing="global"`' in SERVER_PHASE1_PROMPT


def test_server_consolidation_prompt_never_emits_a_tone_note() -> None:
    # The tone note is a per-user tier: the server prompt must declare its
    # `<existing_tone>` input always empty and demand an empty tone output.
    assert "always `(empty)`" in SERVER_PHASE2_PROMPT
    assert "TONE NOTE OUTPUT" in SERVER_PHASE2_PROMPT
    assert "Always return an empty `tone_markdown`" in SERVER_PHASE2_PROMPT


# ---------------------------------------------------------------------------
# /memory server show
# ---------------------------------------------------------------------------


class ResponseStub:
    """Records the response payload sent by the cog."""

    def __init__(self) -> None:
        """Initializes the recorded payload."""
        self.sent: dict[str, object] = {}

    async def send_message(self, **kwargs: object) -> None:
        """Records the response payload."""
        self.sent = kwargs


def _server_cog() -> MemoryCogs:
    """Builds a MemoryCogs whose bot exposes a stable user id."""
    bot = SimpleNamespace(user=SimpleNamespace(id=BOT_ID))
    return MemoryCogs(bot=as_bot(fake=bot))


def _guild_interaction(guild_id: int | None = GUILD_ID) -> SimpleNamespace:
    """Builds a minimal guild interaction stub for the server memory command."""
    guild = None if guild_id is None else SimpleNamespace(id=guild_id)
    return SimpleNamespace(guild=guild, response=ResponseStub())


async def test_memory_server_show_displays_stored_memory(memory_isolated_dir: Path) -> None:
    write_main_memory(
        scope=SERVER_SCOPE,
        content="v1\n\n## 伺服器輪廓\n大家都很愛玩楓之谷",
        identity=SERVER_IDENTITY,
    )
    cog = _server_cog()
    interaction = _guild_interaction()
    await MemoryCogs.memory_server_show.callback(cog, as_interaction(fake=interaction))
    assert interaction.response.sent["ephemeral"] is True
    embed = interaction.response.sent["embed"]
    assert isinstance(embed, Embed)
    assert "楓之谷" in (embed.description or "")


async def test_memory_server_show_handles_empty_memory(memory_isolated_dir: Path) -> None:
    cog = _server_cog()
    interaction = _guild_interaction()
    await MemoryCogs.memory_server_show.callback(cog, as_interaction(fake=interaction))
    embed = interaction.response.sent["embed"]
    assert isinstance(embed, Embed)
    assert "還沒有對這個伺服器的記憶" in (embed.description or "")


async def test_memory_server_show_blocks_dms(memory_isolated_dir: Path) -> None:
    cog = _server_cog()
    interaction = _guild_interaction(guild_id=None)
    await MemoryCogs.memory_server_show.callback(cog, as_interaction(fake=interaction))
    embed = interaction.response.sent["embed"]
    assert isinstance(embed, Embed)
    assert "只能在伺服器" in (embed.description or "")
    # A DM read must never reach the store.
    assert read_main_memory(scope=SERVER_SCOPE) == ""
