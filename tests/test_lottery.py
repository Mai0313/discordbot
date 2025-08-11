import pytest

from discordbot.cogs import lottery as lot


@pytest.fixture(autouse=True)
def reset_lottery_state():
    """Reset in-memory lottery state before each test to avoid cross-test pollution."""
    lot.lotteries_by_id.clear()
    lot.lottery_participants.clear()
    lot.lottery_winners.clear()
    lot.next_lottery_id = 1
    yield
    # Ensure clean after as well
    lot.lotteries_by_id.clear()
    lot.lottery_participants.clear()
    lot.lottery_winners.clear()


def _create_reaction_lottery(guild_id: int = 123) -> lot.LotteryData:
    lottery_id = lot.create_lottery({
        "guild_id": guild_id,
        "title": "單純測試抽獎",
        "description": "desc",
        "creator_id": 999,
        "creator_name": "tester",
        "registration_method": "reaction",
    })
    return lot.lotteries_by_id[lottery_id]


def _create_youtube_lottery(guild_id: int = 456) -> lot.LotteryData:
    lottery_id = lot.create_lottery({
        "guild_id": guild_id,
        "title": "YT測試抽獎",
        "description": "desc",
        "creator_id": 1000,
        "creator_name": "yt",
        "registration_method": "youtube",
        "youtube_url": "https://youtube.com/live/test",
        "youtube_keyword": "加入抽獎",
    })
    return lot.lotteries_by_id[lottery_id]


def test_create_and_retrieve_lottery_reaction():
    data = _create_reaction_lottery(111)
    assert data.is_active is True
    assert lot.lotteries_by_id[data.lottery_id] is data


def test_update_reaction_message_id():
    data = _create_reaction_lottery(222)
    assert data.reaction_message_id is None
    lot.update_reaction_message_id(data.lottery_id, 555555)
    assert data.reaction_message_id == 555555


def test_add_participant_platform_validation_and_duplicates():
    data = _create_reaction_lottery(333)

    # YouTube user cannot join reaction-based lottery
    yt_user = lot.LotteryParticipant(id="yt_user", name="yt_user", source="youtube")
    assert lot.add_participant(data.lottery_id, yt_user) is False
    assert len(lot.get_participants(data.lottery_id)) == 0

    # Discord user can join
    dc_user = lot.LotteryParticipant(id="123", name="DC", source="discord")
    assert lot.add_participant(data.lottery_id, dc_user) is True
    assert len(lot.get_participants(data.lottery_id)) == 1

    # Duplicate same-source join returns True but does not add another entry
    assert lot.add_participant(data.lottery_id, dc_user) is True
    assert len(lot.get_participants(data.lottery_id)) == 1


def test_remove_participant():
    data = _create_youtube_lottery(444)

    p1 = lot.LotteryParticipant(id="A", name="A", source="youtube")
    p2 = lot.LotteryParticipant(id="B", name="B", source="youtube")
    assert lot.add_participant(data.lottery_id, p1) is True
    assert lot.add_participant(data.lottery_id, p2) is True
    assert len(lot.get_participants(data.lottery_id)) == 2

    lot.remove_participant(data.lottery_id, "A", "youtube")
    assert len(lot.get_participants(data.lottery_id)) == 1
    assert lot.get_participants(data.lottery_id)[0].id == "B"


def test_winners_list_and_participants_remain_independent():
    data = _create_reaction_lottery(555)
    p1 = lot.LotteryParticipant(id="1", name="U1", source="discord")
    p2 = lot.LotteryParticipant(id="2", name="U2", source="discord")
    lot.add_participant(data.lottery_id, p1)
    lot.add_participant(data.lottery_id, p2)

    # Simulate winners and ensure participants remain
    lot.add_winner(data.lottery_id, p1)
    lot.add_winner(data.lottery_id, p2)
    assert len(lot.lottery_winners[data.lottery_id]) == 2
    assert len(lot.get_participants(data.lottery_id)) == 2


def test_add_participants_fields_to_embed():
    import nextcord

    participants = [
        lot.LotteryParticipant(id="1", name="A", source="discord"),
        lot.LotteryParticipant(id="2", name="B", source="youtube"),
        lot.LotteryParticipant(id="3", name="C", source="discord"),
    ]

    embed = nextcord.Embed(title="測試")
    lot.add_participants_fields_to_embed(embed, participants)
    data = embed.to_dict()
    fields = data.get("fields", [])

    # Discord field, YouTube field, Total field
    assert len(fields) == 3
    names = [f["name"] for f in fields]
    assert any("Discord 參與者 (2 人)" in n for n in names)
    assert any("YouTube 參與者 (1 人)" in n for n in names)
    assert any("總參與人數" in n for n in names)

    # Values contain joined names
    discord_field = next(f for f in fields if f["name"].startswith("Discord 參與者"))
    youtube_field = next(f for f in fields if f["name"].startswith("YouTube 參與者"))
    assert "A" in discord_field["value"]
    assert "C" in discord_field["value"]
    assert "B" in youtube_field["value"]


def test_close_lottery_clears_mappings():
    data = _create_reaction_lottery(666)
    lot.update_reaction_message_id(data.lottery_id, 12345)

    lot.close_lottery(data.lottery_id)
    assert data.lottery_id not in lot.lotteries_by_id


def test_reaction_helpers_removed():
    """Reaction helpers are removed; ensure module no longer exposes them."""
    assert not hasattr(lot, "_get_reaction_lottery_or_none")
