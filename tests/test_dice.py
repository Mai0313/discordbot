"""Tests for the /dice helper module."""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from random import Random

from discordbot.cogs._games.dice import play_dice, render_rolls
from discordbot.cogs._games.presentation import settlement_footer


def test_play_dice_with_seeded_rng_is_deterministic() -> None:
    """A seeded RNG produces a reproducible roll set."""
    first = play_dice(rng=Random(x=42))
    second = play_dice(rng=Random(x=42))
    assert first == second


def test_play_dice_roll_count_and_face_range() -> None:
    """Each side rolls three six-sided dice with faces in [1, 6]."""
    result = play_dice(rng=Random(x=1))
    assert len(result.player_rolls) == 3
    assert len(result.dealer_rolls) == 3
    assert all(1 <= face <= 6 for face in result.player_rolls)
    assert all(1 <= face <= 6 for face in result.dealer_rolls)


def test_play_dice_outcome_matches_totals() -> None:
    """The outcome label always agrees with the comparison of totals."""
    for seed in range(10):
        result = play_dice(rng=Random(x=seed))
        if result.player_total > result.dealer_total:
            assert result.outcome == "win"
        elif result.player_total < result.dealer_total:
            assert result.outcome == "lose"
        else:
            assert result.outcome == "push"


def test_render_rolls_includes_faces_and_total() -> None:
    """Rendered output uses unicode die faces and shows the running total."""
    rendered = render_rolls(rolls=(1, 2, 3))
    assert "⚀" in rendered
    assert "⚁" in rendered
    assert "⚂" in rendered
    assert "= 6" in rendered


def test_settlement_footer_does_not_prefix_positive_house_balance() -> None:
    """House balance is an absolute ledger balance, not this-round profit."""
    footer = settlement_footer(
        bet=100, delta=-100, new_balance=500, house_balance=1_200, is_allin=False
    )
    assert "莊家餘額 1,200 虛擬歡樂豆" in footer
    assert "莊家餘額 +1,200 虛擬歡樂豆" not in footer
