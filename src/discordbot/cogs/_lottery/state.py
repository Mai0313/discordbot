from datetime import datetime
from collections import defaultdict
from collections.abc import Iterable

from .models import LotteryData, LotteryParticipant

# 全局狀態
lotteries_by_id: dict[int, LotteryData] = {}
lottery_participants: defaultdict[int, list[LotteryParticipant]] = defaultdict(list)
lottery_winners: defaultdict[int, list[LotteryParticipant]] = defaultdict(list)
message_to_lottery_id: dict[int, int] = {}
_next_lottery_id = 1


def create_lottery(lottery_data: dict) -> int:
    """Create and register a new lottery entry."""
    global _next_lottery_id
    lottery_id = _next_lottery_id
    _next_lottery_id += 1

    lottery = LotteryData(
        lottery_id=lottery_id,
        guild_id=lottery_data["guild_id"],
        title=lottery_data["title"],
        description=lottery_data.get("description", ""),
        creator_id=lottery_data["creator_id"],
        creator_name=lottery_data["creator_name"],
        created_at=datetime.now(),
        is_active=True,
        registration_method=lottery_data["registration_method"],
        youtube_url=lottery_data.get("youtube_url"),
        youtube_keyword=lottery_data.get("youtube_keyword"),
        control_message_id=lottery_data.get("control_message_id"),
        draw_count=max(1, int(lottery_data.get("draw_count", 1) or 1)),
    )

    lotteries_by_id[lottery_id] = lottery
    return lottery_id


def update_control_message_id(lottery_id: int, message_id: int) -> None:
    """Track the control message associated with a lottery."""
    lottery = lotteries_by_id.get(lottery_id)
    if lottery is not None:
        lottery.control_message_id = message_id
        message_to_lottery_id[message_id] = lottery_id


def get_lottery(lottery_id: int) -> LotteryData | None:
    """Return lottery data by identifier."""
    return lotteries_by_id.get(lottery_id)


def get_lottery_by_message_id(message_id: int) -> LotteryData | None:
    """Resolve a lottery using its control message id."""
    lottery_id = message_to_lottery_id.get(message_id)
    return lotteries_by_id.get(lottery_id) if lottery_id is not None else None


def add_participant(lottery_id: int, participant: LotteryParticipant) -> bool:
    """Add a participant if valid, ensuring idempotency per source."""
    lottery = lotteries_by_id.get(lottery_id)
    if lottery is None:
        return False

    for winner in lottery_winners.get(lottery_id, []):
        if winner.id == participant.id and winner.source == participant.source:
            return False

    for existing in lottery_participants[lottery_id]:
        if existing.id == participant.id and existing.source == participant.source:
            return True

    lottery_participants[lottery_id].append(participant)
    return True


def get_participants(lottery_id: int) -> list[LotteryParticipant]:
    """Return participant list for a lottery."""
    return lottery_participants[lottery_id]


def set_participants(lottery_id: int, participants: Iterable[LotteryParticipant]) -> None:
    """Replace participants for a lottery with the provided iterable."""
    lottery_participants[lottery_id] = list(participants)


def get_winners(lottery_id: int) -> list[LotteryParticipant]:
    """Return recorded winners for a lottery."""
    return lottery_winners[lottery_id]


def add_winner(lottery_id: int, participant: LotteryParticipant) -> None:
    """Record a winning participant."""
    lottery_winners[lottery_id].append(participant)


def remove_participant(lottery_id: int, participant_id: str, source: str) -> None:
    """Remove a participant by identifier and source."""
    lottery_participants[lottery_id] = [
        p
        for p in lottery_participants[lottery_id]
        if not (p.id == participant_id and p.source == source)
    ]


def close_lottery(lottery_id: int) -> None:
    """Close a lottery and detach its control message mapping."""
    lottery = lotteries_by_id.pop(lottery_id, None)
    if lottery is not None:
        lottery.is_active = False
        if lottery.control_message_id is not None:
            message_to_lottery_id.pop(lottery.control_message_id, None)
