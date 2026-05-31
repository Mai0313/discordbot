"""Monte Carlo measurement of the bot player's Blackjack edge and variance.

The bot plays the EV engine's hole-aware `recommended_action` on every decision
and the count-based insurance recommendation, exactly the deterministic policy
the production bot uses. The default mode draws a fresh 4-deck shoe each round,
so it measures the bot's neutral-count edge `e` (mean net result in base-bet
units) and variance `sigma^2`.

Those two numbers are the constants the fractional-Kelly bet sizer needs. Run:

    uv run python scripts/simulate_bot_blackjack.py --rounds 100000 --seed 12345

The reported `half_kelly_fraction = 0.5 * e / sigma^2` is the fraction of
bankroll the bot should wager each round under half-Kelly. A positive `e`
confirms this table's player-favorable five-card rules make the hole-aware bot
+EV before any bet sizing.

`--persistent` carries one shoe across rounds (reshuffling near the cut) and
reports the edge binned by pre-deal Hi-Lo true count plus the fitted slope,
which is the `BOT_EDGE_PER_TRUE_COUNT` constant for count-based bet spreading.
"""

from random import Random
import argparse

from rich.table import Table
from rich.console import Console

from discordbot.typings.games import Card, BotAction, GameParticipant
from discordbot.cogs._games.blackjack import (
    BlackjackRound,
    BlackjackHandState,
    BlackjackPlayerHand,
    can_split,
    build_shoe,
    can_double,
    settle_hand,
    can_surrender,
    dealer_up_card,
    committed_wagers,
    is_five_card_twenty_one,
)
from discordbot.cogs._games.bot_player import fallback_insurance, build_bot_insurance_context
from discordbot.cogs._games.blackjack_ev import compute_action_evs, compute_true_count

# A large base bet keeps insurance (half-bet), doubles, and splits affordable and
# divisible; results are normalized back to base-bet units by dividing by it.
BASE_BET = 100
START_BALANCE = 100_000_000
_BOT_USER_ID = 0
_MAX_ACTION_STEPS = 64
# Persistent-shoe mode reshuffles once fewer than this many cards remain, matching
# the production per-channel shoe penetration cut (shoe.py RESHUFFLE_THRESHOLD_CARDS).
RESHUFFLE_THRESHOLD_CARDS = 96
console = Console()


def _allowed_actions(
    *, round_state: BlackjackRound, player: BlackjackPlayerHand, hand: BlackjackHandState
) -> tuple[BotAction, ...]:
    """Returns the legal actions for the active hand, mirroring the live view."""
    balance_remaining = player.participant.balance_at_start - committed_wagers(player=player)
    allowed: list[BotAction] = []
    if not hand.finished and not hand.is_split_aces:
        allowed.append("hit")
        allowed.append("stand")
    if can_double(hand=hand, balance_remaining=balance_remaining):
        allowed.append("double")
    if can_split(hand=hand, balance_remaining=balance_remaining):
        allowed.append("split")
    if can_surrender(hand=hand, peeked_blackjack=round_state.peeked_blackjack):
        allowed.append("surrender")
    return tuple(allowed)


def _recommended_action(
    *, round_state: BlackjackRound, hand: BlackjackHandState, allowed: tuple[BotAction, ...]
) -> BotAction:
    """Returns the EV engine's hole-aware recommended action for the active hand."""
    analysis = compute_action_evs(
        hand_cards=hand.cards,
        dealer_cards=round_state.dealer,
        shoe=round_state.shoe,
        allowed_actions=allowed,
        doubled=hand.doubled,
        bet=hand.bet,
    )
    if analysis.recommended_action in allowed:
        return analysis.recommended_action
    return allowed[0]


def _apply_action(*, round_state: BlackjackRound, action: BotAction) -> None:
    """Applies one chosen action to the round through the pure-rules mutators."""
    if action == "hit":
        round_state.hit(user_id=_BOT_USER_ID)
    elif action == "stand":
        round_state.stand(user_id=_BOT_USER_ID)
    elif action == "double":
        round_state.double_down(user_id=_BOT_USER_ID)
    elif action == "split":
        round_state.split(user_id=_BOT_USER_ID)
    elif action == "surrender":
        round_state.surrender(user_id=_BOT_USER_ID)


def _resolve_insurance(*, round_state: BlackjackRound) -> None:
    """Takes or declines insurance from the count-based recommendation."""
    context = build_bot_insurance_context(
        dealer_up=dealer_up_card(dealer=round_state.dealer),
        shoe=list(round_state.shoe),
        insurance_cost=BASE_BET // 2,
    )
    if fallback_insurance(insurance_context=context):
        round_state.take_insurance(user_id=_BOT_USER_ID, amount=BASE_BET // 2)
    else:
        round_state.decline_insurance(user_id=_BOT_USER_ID)


def _play_actions(*, round_state: BlackjackRound) -> None:
    """Drives the bot's hands with the recommended action until the round resolves."""
    steps = 0
    while round_state.phase == "player_actions" and not round_state.finished:
        player = round_state.active_player()
        hand = round_state.active_hand()
        if player is None or hand is None or player.participant.user_id != _BOT_USER_ID:
            break
        allowed = _allowed_actions(round_state=round_state, player=player, hand=hand)
        if not allowed:
            round_state.stand(user_id=_BOT_USER_ID)
        else:
            _apply_action(
                round_state=round_state,
                action=_recommended_action(round_state=round_state, hand=hand, allowed=allowed),
            )
        steps += 1
        if steps > _MAX_ACTION_STEPS:
            raise RuntimeError("Bot action loop exceeded the safety step limit")


def _round_delta(*, round_state: BlackjackRound) -> float:
    """Returns the round's net result in base-bet units, matching production economics."""
    player = round_state.players[0]
    base = sum(settle_hand(hand=hand, dealer=round_state.dealer)[1] for hand in player.hands)
    if player.insurance_bet > 0:
        base += player.insurance_bet * 2 if round_state.peeked_blackjack else -player.insurance_bet
    five_card_bonus = sum(
        hand.bet
        for hand in player.hands
        if not hand.doubled and is_five_card_twenty_one(cards=hand.cards)
    )
    return (base + five_card_bonus) / BASE_BET


def simulate_round(*, rng: Random, shoe: list[Card] | None = None) -> float:
    """Plays one hole-aware optimal round and returns its base-bet-unit result.

    When `shoe` is given the round deals from it in place (persistent-shoe mode),
    so the caller's list is left holding the cards that remain after the round.
    """
    participant = GameParticipant(
        user_id=_BOT_USER_ID,
        account_name="bot",
        display_name="bot",
        bet=BASE_BET,
        balance_at_start=START_BALANCE,
        is_allin=False,
    )
    round_state = BlackjackRound.from_participants(
        rng=rng, participants=[participant], auto_play_dealer=True
    )
    if shoe is not None:
        round_state.shoe = shoe
    round_state.deal_initial()
    if round_state.phase == "insurance":
        _resolve_insurance(round_state=round_state)
    _play_actions(round_state=round_state)
    return _round_delta(round_state=round_state)


def _render_report(  # noqa: PLR0913 -- aggregates the independent measurement tallies into one report.
    *, rounds: int, seed: int, total: float, total_sq: float, wins: int, pushes: int, losses: int
) -> Table:
    """Builds the rich summary table for the measured edge and variance."""
    mean = total / rounds
    variance = max(total_sq / rounds - mean * mean, 0.0)
    std = variance**0.5
    margin = 1.96 * std / (rounds**0.5)
    full_kelly = mean / variance if variance > 0 else 0.0
    table = Table(title=f"Bot Blackjack edge over {rounds:,} rounds (seed {seed})")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("edge e (mean, base-bet units)", f"{mean:+.5f}")
    table.add_row("95% CI of e", f"[{mean - margin:+.5f}, {mean + margin:+.5f}]")
    table.add_row("variance sigma^2", f"{variance:.5f}")
    table.add_row("std sigma", f"{std:.5f}")
    table.add_row("win rate", f"{wins / rounds:.4f}")
    table.add_row("push rate", f"{pushes / rounds:.4f}")
    table.add_row("loss rate", f"{losses / rounds:.4f}")
    table.add_row("full-kelly fraction e/sigma^2", f"{full_kelly:.5f}")
    table.add_row("half-kelly fraction", f"{0.5 * full_kelly:.5f}")
    return table


def _run_persistent(*, rng: Random, rounds: int, threshold: int) -> list[tuple[float, float]]:
    """Plays rounds from a persistent shoe, returning (pre-deal true count, delta) pairs."""
    records: list[tuple[float, float]] = []
    shoe: list[Card] = []
    for _ in range(rounds):
        if len(shoe) < threshold:
            shoe = build_shoe(rng=rng)
        true_count = compute_true_count(shoe=shoe)
        delta = simulate_round(rng=rng, shoe=shoe)
        records.append((true_count, delta))
    return records


def _render_count_report(*, records: list[tuple[float, float]], seed: int) -> Table:
    """Bins persistent-shoe results by Hi-Lo true count and fits the edge slope."""
    rounds = len(records)
    true_counts = [record[0] for record in records]
    deltas = [record[1] for record in records]
    tc_mean = sum(true_counts) / rounds
    delta_mean = sum(deltas) / rounds
    covariance = sum((tc - tc_mean) * (d - delta_mean) for tc, d in records) / rounds
    tc_variance = sum((tc - tc_mean) ** 2 for tc in true_counts) / rounds
    slope = covariance / tc_variance if tc_variance > 0 else 0.0
    intercept = delta_mean - slope * tc_mean
    bins: dict[int, list[float]] = {}
    for true_count, delta in records:
        bins.setdefault(round(true_count), []).append(delta)
    table = Table(
        title=f"Persistent-shoe edge vs Hi-Lo true count, {rounds:,} rounds (seed {seed})"
    )
    table.add_column("true_count_bin")
    table.add_column("rounds", justify="right")
    table.add_column("mean_edge", justify="right")
    for true_count_bin in sorted(bins):
        bucket = bins[true_count_bin]
        table.add_row(
            f"{true_count_bin:+d}", f"{len(bucket):,}", f"{sum(bucket) / len(bucket):+.4f}"
        )
    table.add_section()
    table.add_row("slope (edge per +1 TC)", "", f"{slope:+.5f}")
    table.add_row("intercept (edge at TC 0)", "", f"{intercept:+.5f}")
    return table


def main() -> None:
    """Runs the Monte Carlo simulation and prints the measured constants."""
    parser = argparse.ArgumentParser(description="Measure the bot player's Blackjack edge.")
    parser.add_argument("--rounds", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=12_345)
    parser.add_argument(
        "--persistent",
        action="store_true",
        help="Carry the shoe across rounds and report edge vs Hi-Lo true count.",
    )
    parser.add_argument("--reshuffle-threshold", type=int, default=RESHUFFLE_THRESHOLD_CARDS)
    args = parser.parse_args()
    rng = Random(args.seed)  # noqa: S311 -- reproducible measurement, not security-sensitive.
    if args.persistent:
        records = _run_persistent(rng=rng, rounds=args.rounds, threshold=args.reshuffle_threshold)
        console.print(_render_count_report(records=records, seed=args.seed))
        return
    total = 0.0
    total_sq = 0.0
    wins = pushes = losses = 0
    for _ in range(args.rounds):
        delta = simulate_round(rng=rng)
        total += delta
        total_sq += delta * delta
        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        else:
            pushes += 1
    console.print(
        _render_report(
            rounds=args.rounds,
            seed=args.seed,
            total=total,
            total_sq=total_sq,
            wins=wins,
            pushes=pushes,
            losses=losses,
        )
    )


if __name__ == "__main__":
    main()
