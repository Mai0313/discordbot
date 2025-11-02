from collections.abc import Generator

import pytest
import nextcord

from discordbot.cogs._lottery import state
from discordbot.cogs._lottery.embeds import add_participants_field
from discordbot.cogs._lottery.models import LotteryData, LotteryParticipant


@pytest.fixture(autouse=True)
def reset_lottery_state() -> Generator[None, None, None]:
    """Reset in-memory lottery state before each test to avoid cross-test pollution."""
    state.lotteries_by_id.clear()
    state.lottery_participants.clear()
    state.lottery_winners.clear()
    state._lottery_id_counter.reset(1)
    yield
    # Ensure clean after as well
    state.lotteries_by_id.clear()
    state.lottery_participants.clear()
    state.lottery_winners.clear()


def _create_discord_lottery(guild_id: int = 123) -> LotteryData:
    lottery_id = state.create_lottery({
        "guild_id": guild_id,
        "title": "單純測試抽獎",
        "description": "desc",
        "creator_id": 999,
        "creator_name": "tester",
        "registration_method": "discord",
    })
    return state.lotteries_by_id[lottery_id]


def _create_youtube_lottery(guild_id: int = 456) -> LotteryData:
    lottery_id = state.create_lottery({
        "guild_id": guild_id,
        "title": "YT測試抽獎",
        "description": "desc",
        "creator_id": 1000,
        "creator_name": "yt",
        "registration_method": "youtube",
        "youtube_url": "https://youtube.com/live/test",
        "youtube_keyword": "加入抽獎",
    })
    return state.lotteries_by_id[lottery_id]


def test_create_and_retrieve_lottery_discord() -> None:
    data = _create_discord_lottery(111)
    assert data.is_active is True
    assert state.lotteries_by_id[data.lottery_id] is data


def test_update_control_message_id() -> None:
    data = _create_discord_lottery(222)
    assert data.control_message_id is None
    state.update_control_message_id(data.lottery_id, 555555)
    assert data.control_message_id == 555555


def test_add_participant_duplicates() -> None:
    data = _create_discord_lottery(333)

    # Discord user can join
    dc_user = LotteryParticipant(id="123", name="DC", source="discord")
    assert state.add_participant(data.lottery_id, dc_user) is True
    assert len(state.get_participants(data.lottery_id)) == 1

    # Duplicate same-source join returns True but does not add another entry
    assert state.add_participant(data.lottery_id, dc_user) is True
    assert len(state.get_participants(data.lottery_id)) == 1


def test_remove_participant() -> None:
    data = _create_youtube_lottery(444)

    p1 = LotteryParticipant(id="A", name="A", source="youtube")
    p2 = LotteryParticipant(id="B", name="B", source="youtube")
    assert state.add_participant(data.lottery_id, p1) is True
    assert state.add_participant(data.lottery_id, p2) is True
    assert len(state.get_participants(data.lottery_id)) == 2

    state.remove_participant(data.lottery_id, "A", "youtube")
    assert len(state.get_participants(data.lottery_id)) == 1
    assert state.get_participants(data.lottery_id)[0].id == "B"


def test_winners_list_and_participants_remain_independent() -> None:
    data = _create_discord_lottery(555)
    p1 = LotteryParticipant(id="1", name="U1", source="discord")
    p2 = LotteryParticipant(id="2", name="U2", source="discord")
    state.add_participant(data.lottery_id, p1)
    state.add_participant(data.lottery_id, p2)

    # Simulate winners and ensure participants remain
    state.add_winner(data.lottery_id, p1)
    state.add_winner(data.lottery_id, p2)
    assert len(state.lottery_winners[data.lottery_id]) == 2
    assert len(state.get_participants(data.lottery_id)) == 2


def test_add_participants_field_unified() -> None:
    participants = [
        LotteryParticipant(id="1", name="A", source="discord"),
        LotteryParticipant(id="2", name="B", source="youtube"),
        LotteryParticipant(id="3", name="C", source="discord"),
    ]

    embed = nextcord.Embed(title="測試")
    add_participants_field(embed, participants)
    data = embed.to_dict()
    fields = data.get("fields", [])

    # Unified single field
    assert len(fields) == 1
    assert fields[0]["name"].startswith("參與者（3 人）")

    # Values contain joined names
    assert "A" in fields[0]["value"]
    assert "B" in fields[0]["value"]
    assert "C" in fields[0]["value"]


def test_close_lottery_clears_mappings() -> None:
    data = _create_discord_lottery(666)
    state.update_control_message_id(data.lottery_id, 12345)

    state.close_lottery(data.lottery_id)
    assert data.lottery_id not in state.lotteries_by_id


def test_reaction_helpers_removed() -> None:
    """Reaction helpers are removed; ensure module no longer exposes them."""
    assert not hasattr(state, "_get_reaction_lottery_or_none")
