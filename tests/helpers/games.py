"""Builders the games tests share: cards, seats, settlement, and the controls a view shows."""

from typing import Any

from nextcord.ui import View, Button

from discordbot.typings.games import Card, GameParticipant, BlackjackPlayerSettlement
from discordbot.cogs.games.blackjack import SHOE_DECK_COUNT, BlackjackRound, hand_value, is_soft_17
from discordbot.cogs.games.settlement import settle_blackjack_player


def card(rank: str, suit: str = "♠") -> Card:
    """Builds one card; the suit has a default because every rule here reads the rank alone."""
    return Card(rank=rank, suit=suit)


def seat(
    *, user_id: int = 1, display_name: str = "Alice", bet: int = 100, balance_at_start: int = 1_000
) -> GameParticipant:
    """Builds a seated player whose account name is the display name lower-cased."""
    return GameParticipant(
        user_id=user_id,
        account_name=display_name.lower(),
        display_name=display_name,
        bet=bet,
        balance_at_start=balance_at_start,
        is_allin=False,
    )


async def settle_only_seat(round_state: BlackjackRound) -> BlackjackPlayerSettlement:
    """Settles a one-seat round's player against the ledger and returns the settlement."""
    player = round_state.players[0]
    return await settle_blackjack_player(
        round_state=round_state,
        player=player,
        player_id=player.participant.user_id,
        player_account_name=player.participant.account_name,
    )


def longest_hand_the_dealer_must_draw_on() -> int:
    """Returns the most cards an H17 dealer can hold while the rules still make it draw.

    Searched rather than reasoned out, because it is not the number anyone reaches by hand:
    eleven aces and a five is hard 16 and twelve cards, where a hand of small cards runs out
    sooner and a hand of aces stands early on the soft total. A rules change re-derives it.
    The draw rule is `_play_dealer_locked`'s own `should_hit`; change it there and change it
    here.
    """
    ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    per_shoe = 4 * SHOE_DECK_COUNT

    def must_draw(counts: dict[str, int]) -> bool:
        cards = [card(rank=rank) for rank, held in counts.items() for _ in range(held)]
        total = hand_value(cards=cards)
        return total < 17 or (total == 17 and is_soft_17(cards=cards))

    longest = 0
    seen: set[tuple[tuple[str, int], ...]] = set()
    # A hand is its multiset: `must_draw` cannot read an order, so one ordering settles them all.
    stack: list[dict[str, int]] = [{}]
    while stack:
        counts = stack.pop()
        key = tuple(sorted(counts.items()))
        if key in seen:
            continue
        seen.add(key)
        held = sum(counts.values())
        # The dealer's own first two cards are dealt, not drawn, so they are unconstrained.
        if held >= 2 and not must_draw(counts=counts):
            continue
        longest = max(longest, held)
        for rank in ranks:
            if counts.get(rank, 0) < per_shoe:
                stack.append({**counts, rank: counts.get(rank, 0) + 1})
    return longest


def component_ids(view: View) -> set[str]:
    """Returns the custom ids of the controls a view currently shows.

    Game views remove a control that cannot be used rather than disabling it, so which ids
    are attached is the state a player sees.
    """
    return set(component_rows(view=view))


def component_rows(view: View) -> dict[str, int | None]:
    """Returns `{custom_id: row}` for the controls a view currently shows."""
    rows: dict[str, int | None] = {}
    for child in view.children:
        custom_id = getattr(child, "custom_id", None)
        if isinstance(custom_id, str):
            rows[custom_id] = child.row
    return rows


def attached_button(view: View, custom_id: str) -> Button[Any]:
    """Returns the attached button with this custom id, failing the test when it is absent."""
    for child in view.children:
        if isinstance(child, Button) and child.custom_id == custom_id:
            return child
    raise AssertionError(f"no attached button {custom_id!r}; attached: {component_ids(view=view)}")
