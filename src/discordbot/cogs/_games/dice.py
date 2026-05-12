"""Pure dice-roll helpers for the /dice command."""

from random import Random

from discordbot.typings.games import DiceResult, DiceOutcome


def roll_dice(rng: Random) -> tuple[int, ...]:
    """Rolls one side's three six-sided dice.

    Args:
        rng: Random source used to generate each die value.

    Returns:
        The rolled values in order.
    """
    return tuple(rng.randint(a=1, b=6) for _ in range(3))


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


def render_rolls(rolls: tuple[int, ...]) -> str:
    """Formats dice rolls with unicode die faces and total.

    Args:
        rolls: Die values to render.

    Returns:
        A display string containing die faces and the sum.
    """
    emoji = {1: "⚀", 2: "⚁", 3: "⚂", 4: "⚃", 5: "⚄", 6: "⚅"}
    faces = " ".join(emoji[value] for value in rolls)
    return f"{faces}  (= {sum(rolls)})"
