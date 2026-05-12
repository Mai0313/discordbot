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


def test_settlement_footer_shows_delta_and_balance() -> None:
    """Footer keeps only the round delta and post-settlement balance."""
    footer = settlement_footer(delta=-100, new_balance=500, is_allin=False)
    assert "本局 -100 虛擬歡樂豆" in footer
    assert "餘額 500 虛擬歡樂豆" in footer
    assert "下注" not in footer
    assert "莊家" not in footer


def test_settlement_footer_signs_positive_delta() -> None:
    """A winning round prefixes the delta with a `+`."""
    footer = settlement_footer(delta=200, new_balance=700, is_allin=False)
    assert "本局 +200 虛擬歡樂豆" in footer


def test_settlement_footer_appends_allin_note() -> None:
    """All-in rounds add the auto all-in suffix."""
    footer = settlement_footer(delta=0, new_balance=0, is_allin=True)
    assert "已自動 all-in" in footer
