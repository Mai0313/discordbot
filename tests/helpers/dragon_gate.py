"""Builders the 射龍門 tests share: a scripted deal, seats at the ante, and a recording ledger."""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING, TypeVar, cast

from discordbot.typings.economy import (
    JackpotSettlementResult,
    JackpotSettlementRequest,
    JackpotSettlementBatchResult,
)
from discordbot.cogs.games.dragon_gate import ANTE
from discordbot.services.economy.database import (
    apply_jackpot_settlement,
    apply_jackpot_settlement_batch,
)

from tests.helpers.games import seat
from tests.helpers.economy import seed_balance
from tests.helpers.message_cleanup import record_scheduled_deletes

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import pytest
    from _typeshed import SupportsLenAndGetItem

    from discordbot.typings.games import GameParticipant


T = TypeVar("T")


_RIGGED_FILLER: tuple[str, ...] = ("2", "♠") * 32


class RiggedRandom(Random):  # noqa: S311 -- scripted for determinism, not cryptography
    """Random subclass that returns a fixed rank/suit sequence (padded with filler)."""

    def __init__(self, choices: Sequence[str]) -> None:
        """Initializes the deterministic choice stream with safe filler values."""
        super().__init__(x=0)
        # Pad with safe filler so the rules engine can keep dealing extra turns
        # after the asserted hand finishes; the view layer is responsible for
        # finalising on jackpot exhaustion, not the rules.
        padded = tuple(choices) + _RIGGED_FILLER
        self._scripted_choices: Iterator[str] = iter(padded)

    def choice(self, seq: SupportsLenAndGetItem[T]) -> T:
        """Returns the next scripted choice and verifies it belongs to the input."""
        value = next(self._scripted_choices)
        assert value in [seq[index] for index in range(len(seq))]
        return cast("T", value)


def dragon_gate_seat(user_id: int, display_name: str, balance: int = 1_000_000) -> GameParticipant:
    """Builds a 射龍門 seat, which always stakes the ante."""
    return seat(user_id=user_id, display_name=display_name, bet=ANTE, balance_at_start=balance)


async def funded_dragon_gate_seat(
    user_id: int, display_name: str, balance: int = 1_000_000
) -> GameParticipant:
    """Builds a 射龍門 seat whose wallet in the isolated ledger holds what it sat down with."""
    participant = dragon_gate_seat(user_id=user_id, display_name=display_name, balance=balance)
    await seed_balance(user_id=user_id, name=participant.account_name, amount=balance)
    return participant


def record_jackpot_settlements(monkeypatch: pytest.MonkeyPatch) -> list[JackpotSettlementRequest]:
    """Settles every ante and bet against the isolated ledger, recording each request in order.

    Also stands in for the table cleanup, so no deletion is ever scheduled.
    """
    requests: list[JackpotSettlementRequest] = []

    async def settle(  # noqa: PLR0913 -- mirrors apply_jackpot_settlement
        player_id: int,
        player_account_name: str,
        player_delta: int,
        game_id: str,
        player_avatar_url: str = "",
        expected_jackpot_generation: int | None = None,
    ) -> JackpotSettlementResult:
        """Records the bet's request, then settles it."""
        requests.append(
            JackpotSettlementRequest(
                player_id=player_id,
                player_account_name=player_account_name,
                player_avatar_url=player_avatar_url,
                player_delta=player_delta,
                expected_jackpot_generation=expected_jackpot_generation,
            )
        )
        return await apply_jackpot_settlement(
            player_id=player_id,
            player_account_name=player_account_name,
            player_delta=player_delta,
            game_id=game_id,
            player_avatar_url=player_avatar_url,
            expected_jackpot_generation=expected_jackpot_generation,
        )

    async def settle_batch(
        game_id: str, settlements: Sequence[JackpotSettlementRequest]
    ) -> JackpotSettlementBatchResult:
        """Records the antes' requests, then settles them as one batch."""
        requests.extend(settlements)
        return await apply_jackpot_settlement_batch(game_id=game_id, settlements=settlements)

    monkeypatch.setattr("discordbot.cogs.games.dragon_gate_views.apply_jackpot_settlement", settle)
    monkeypatch.setattr("discordbot.cogs.games.lobby.apply_jackpot_settlement_batch", settle_batch)
    record_scheduled_deletes(monkeypatch=monkeypatch)
    return requests
