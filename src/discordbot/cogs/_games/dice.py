"""Pure dice-roll helpers for the /dice command."""

from random import Random
from typing import Literal
from dataclasses import dataclass

DICE_PER_SIDE = 3
DICE_FACES = 6
DiceOutcome = Literal["win", "lose", "push"]


@dataclass(frozen=True)
class DiceResult:
    """Result of one dice round.

    Attributes:
        player_rolls: Player's three rolls in order.
        dealer_rolls: Dealer's three rolls in order.
        player_total: Sum of the player rolls.
        dealer_total: Sum of the dealer rolls.
        outcome: ``win`` / ``lose`` / ``push`` from the player's perspective.
    """

    player_rolls: tuple[int, ...]
    dealer_rolls: tuple[int, ...]
    player_total: int
    dealer_total: int
    outcome: DiceOutcome


def roll_dice(rng: Random) -> tuple[int, ...]:
    """Rolls one side's dice.

    Args:
        rng: Random source used to generate each die value.

    Returns:
        The rolled values in order.
    """
    return tuple(rng.randint(a=1, b=DICE_FACES) for _ in range(DICE_PER_SIDE))


def play_dice(rng: Random) -> DiceResult:
    """Rolls one player-vs-dealer dice round.

    Args:
        rng: Random source used for both sides.

    Returns:
        The rolls, totals, and player-facing outcome.
    """
    player_rolls = roll_dice(rng=rng)
    dealer_rolls = roll_dice(rng=rng)
    player_total = sum(player_rolls)
    dealer_total = sum(dealer_rolls)
    if player_total > dealer_total:
        outcome: DiceOutcome = "win"
    elif player_total < dealer_total:
        outcome = "lose"
    else:
        outcome = "push"
    return DiceResult(
        player_rolls=player_rolls,
        dealer_rolls=dealer_rolls,
        player_total=player_total,
        dealer_total=dealer_total,
        outcome=outcome,
    )


_DICE_FACE_EMOJI: dict[int, str] = {1: "⚀", 2: "⚁", 3: "⚂", 4: "⚃", 5: "⚄", 6: "⚅"}


def render_rolls(rolls: tuple[int, ...]) -> str:
    """Formats dice rolls with unicode die faces and total.

    Args:
        rolls: Die values to render.

    Returns:
        A display string containing die faces and the sum.
    """
    faces = " ".join(_DICE_FACE_EMOJI[value] for value in rolls)
    return f"{faces}  (= {sum(rolls)})"
