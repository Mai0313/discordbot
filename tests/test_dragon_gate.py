"""Tests for the /dragon_gate pure-rules module."""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from random import Random

from discordbot.cogs._games.blackjack import Card
from discordbot.cogs._games.dragon_gate import (
    card_value,
    play_dragon_gate,
    render_card_value,
    dragon_gate_detail,
    settle_dragon_gate,
)


def test_card_value_treats_ace_as_high() -> None:
    """Dragon Gate ranks Ace above King."""
    assert card_value(card=Card(rank="A", suit="♠")) == 14
    assert card_value(card=Card(rank="K", suit="♠")) == 13


def test_settle_dragon_gate_inside_gate_wins() -> None:
    """The shot card wins only when it lands strictly between the gates."""
    result = settle_dragon_gate(
        first_gate=Card(rank="4", suit="♠"),
        second_gate=Card(rank="10", suit="♥"),
        shot=Card(rank="7", suit="♣"),
    )

    assert result.outcome == "win"


def test_settle_dragon_gate_outside_gate_loses() -> None:
    """A shot outside the gate loses the bet."""
    result = settle_dragon_gate(
        first_gate=Card(rank="4", suit="♠"),
        second_gate=Card(rank="10", suit="♥"),
        shot=Card(rank="K", suit="♣"),
    )

    assert result.outcome == "lose"


def test_settle_dragon_gate_boundary_loses() -> None:
    """A shot matching a gate post does not count as inside."""
    result = settle_dragon_gate(
        first_gate=Card(rank="4", suit="♠"),
        second_gate=Card(rank="10", suit="♥"),
        shot=Card(rank="10", suit="♣"),
    )

    assert result.outcome == "lose"


def test_settle_dragon_gate_closed_gate_pushes() -> None:
    """Pair or adjacent gate cards have no playable gap, so the bet is returned."""
    pair = settle_dragon_gate(
        first_gate=Card(rank="8", suit="♠"),
        second_gate=Card(rank="8", suit="♥"),
        shot=Card(rank="9", suit="♣"),
    )
    adjacent = settle_dragon_gate(
        first_gate=Card(rank="8", suit="♠"),
        second_gate=Card(rank="9", suit="♥"),
        shot=Card(rank="K", suit="♣"),
    )

    assert pair.outcome == "push"
    assert adjacent.outcome == "push"


def test_settle_dragon_gate_orders_gate_cards() -> None:
    """The result stores gates in low-to-high order for rendering."""
    result = settle_dragon_gate(
        first_gate=Card(rank="Q", suit="♠"),
        second_gate=Card(rank="3", suit="♥"),
        shot=Card(rank="8", suit="♣"),
    )

    assert result.lower_gate.rank == "3"
    assert result.upper_gate.rank == "Q"


def test_play_dragon_gate_with_seeded_rng_is_deterministic() -> None:
    """A seeded RNG produces a reproducible Dragon Gate round."""
    first = play_dragon_gate(rng=Random(x=42))
    second = play_dragon_gate(rng=Random(x=42))

    assert first == second


def test_render_card_value_includes_card_and_rank_value() -> None:
    """Rendered cards show both display card and rank value."""
    rendered = render_card_value(card=Card(rank="A", suit="♠"))

    assert "A♠" in rendered
    assert "= 14" in rendered


def test_dragon_gate_detail_explains_outcome() -> None:
    """Dealer prompt detail includes gates, shot, and settlement reason."""
    result = settle_dragon_gate(
        first_gate=Card(rank="4", suit="♠"),
        second_gate=Card(rank="10", suit="♥"),
        shot=Card(rank="7", suit="♣"),
    )
    detail = dragon_gate_detail(result=result)

    assert "龍門" in detail
    assert "射門牌" in detail
    assert "中間" in detail
