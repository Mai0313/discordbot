"""Shared vocabulary for the wagered casino games: Blackjack and 射龍門.

Pure data with no cog, util or database dependency, so the rules engines, the EV engine, the
views, the settlement helpers and the round-history store all speak one language. `cogs/games/`
is the only code that keys off these types, and the fishing mini-game living in that same
directory is not part of them: it has its own vocabulary in `typings/fishing.py`.

The models group into five layers:

- Seats. `GameParticipantIdentity` is who a player is, `GameParticipant` is that identity once
  a stake has been fixed against an observed balance, and `ParticipantPreparationResult` /
  `RefreshParticipantsResult` are the two answers a lobby needs: one when a player asks for a
  seat, one when every queued seat is re-checked before the round starts. `SystemIdentity` is
  the casino's own side of the table.
- Settlement. `WagerSettlement` is what one atomic `apply_round_settlement` write left behind
  (player wallet and casino ledger in the same transaction), and `BlackjackPlayerSettlement`
  extends it with the per-hand and insurance rows that add up to that single delta. Bets are
  never debited when a Blackjack round starts, so these models describe the only money a round
  moves. Every amount here is a plain `int`; the decimal-text `StoredInteger` storage that
  dodges SQLite's 64-bit ceiling belongs to the database layer, not to the vocabulary.
- History. `BlackjackHistoryRecord` is one persisted row read back for
  `/games blackjack_history`, and `BlackjackHistoryPayload` is the per-player snapshot its
  JSON column holds. The snapshot types deliberately repeat fields the settlement types
  already carry instead of reusing them: a row written months ago still has to render, so its
  shape cannot follow the settlement path as that path changes.
- Bot player. `DealerOutcome`, `ActionEv` and `ActionEvAnalysis` carry the EV engine's numbers
  for the bot's own turn, split along an information boundary described on `ActionEvAnalysis`.
- Dealer phase. `BlackjackDealerStep` is the audit trail of the deterministic H17 dealer run,
  rendered under the settled table. It belongs to the casino rather than to the bot player:
  the dealer follows fixed rules with no AI in the loop, while the bot sits at the table as an
  ordinary seat.

`DragonGatePlayerResult` is the odd one out: 射龍門 settles each bet into the player row and
the shared jackpot the moment it is placed, so its close-out model reports running totals
rather than a settlement still waiting to be applied.
"""

from typing import Literal
from datetime import datetime

from pydantic import Field, BaseModel, ConfigDict

# Every outcome label a Blackjack result can carry, per sub-hand and per aggregated player. The
# history store writes them verbatim into a text column, so a member renamed or dropped here
# orphans rows already persisted under the old spelling. `five_card_win` and
# `five_card_twenty_one` are this table's 過五關 house rules, not standard Blackjack.
SettleOutcome = Literal[
    "win",
    "lose",
    "push",
    "blackjack",
    "five_card_win",
    "five_card_twenty_one",
    "player_bust",
    "dealer_bust",
    "surrender",
]
# The two wagered tables the games cog runs.
GameKind = Literal["blackjack", "dragon_gate"]
# The only two moves the dealer has under H17.
BlackjackDealerAction = Literal["hit", "stand"]
# Which side produced a recorded dealer step: the H17 rule engine, or the loop's step-limit
# guard. The final embed labels the two differently, so a guard stand is not read as a rule.
BlackjackDealerStepSource = Literal["auto", "guard"]
# Every move the bot player can be offered, and exactly the set the EV engine prices.
BotAction = Literal["hit", "stand", "double", "split", "surrender"]


class Card(BaseModel):
    """A single playing card.

    Attributes:
        rank: One of A, 2-10, J, Q, K.
        suit: One of the four unicode suit glyphs.
    """

    model_config = ConfigDict(frozen=True)

    rank: str = Field(..., description="Card rank: one of A, 2-10, J, Q, K.")
    suit: str = Field(..., description="One of the four unicode suit glyphs.")

    def __str__(self) -> str:
        """Human-readable label like `A♠`.

        Returns:
            The rank followed by the suit glyph.
        """
        return f"{self.rank}{self.suit}"


class GameParticipant(BaseModel):
    """One seated player, with the stake already resolved against the balance seen at join time.

    Built only by `wagers.build_wager_participant`, which is where `bet` picks up its clamps: a
    table bet is capped at the player's balance and at `MAX_SINGLE_BET`, while an ante has to
    be covered in full and is never reduced. Frozen and snapshotted, so a balance that moves
    while the round is running cannot change the stake the round settles.

    Attributes:
        user_id: Discord user ID for the account row and interaction checks.
        account_name: Stable Discord username stored in the economy account row.
        display_name: Guild-aware display name shown in game embeds.
        avatar_url: Last-seen Discord avatar URL for the economy account row.
        bet: Effective wager for this player.
        balance_at_start: Balance observed when the game session starts.
        is_allin: True when the effective wager consumes the full observed balance.
    """

    model_config = ConfigDict(frozen=True)

    user_id: int = Field(
        ..., description="Discord user ID for the account row and interaction checks."
    )
    account_name: str = Field(
        ..., description="Stable Discord username stored in the economy account row."
    )
    display_name: str = Field(..., description="Guild-aware display name shown in game embeds.")
    avatar_url: str = Field(
        default="", description="Last-seen Discord avatar URL for the economy account row."
    )
    bet: int = Field(..., description="Effective wager for this player.")
    balance_at_start: int = Field(
        ..., description="Balance observed when the game session starts."
    )
    is_allin: bool = Field(
        ..., description="True when the effective wager consumes the full observed balance."
    )


class GameParticipantIdentity(BaseModel):
    """The half of a seat that carries no money, resolved once before a stake is decided.

    Kept apart from `GameParticipant` because filling it needs an await (the guild-aware avatar
    lookup) while turning it into a seat is pure arithmetic, so the lobby can re-stake the same
    identity against a fresh balance without touching Discord again.

    Attributes:
        user_id: Discord user ID for the account row and interaction checks.
        account_name: Stable Discord username stored in the economy account row.
        display_name: Guild-aware display name shown in game embeds.
        avatar_url: Last-seen Discord avatar URL for the economy account row.
    """

    model_config = ConfigDict(frozen=True)

    user_id: int = Field(
        ..., description="Discord user ID for the account row and interaction checks."
    )
    account_name: str = Field(
        ..., description="Stable Discord username stored in the economy account row."
    )
    display_name: str = Field(..., description="Guild-aware display name shown in game embeds.")
    avatar_url: str = Field(
        default="", description="Last-seen Discord avatar URL for the economy account row."
    )


class SystemIdentity(BaseModel):
    """The casino's own identity, drawn on the dealer side of a game table.

    Only `system_name` reaches the dealer seat today: the bot sits at the table as an ordinary
    player, so painting its avatar onto the dealer would collide with its own seat.

    Attributes:
        system_id: Discord user ID used for the casino system narrator.
        system_name: Display name used for the casino system narrator.
        system_avatar_url: Avatar URL used for the casino system narrator.
    """

    model_config = ConfigDict(frozen=True)

    system_id: int = Field(..., description="Discord user ID used for the casino system narrator.")
    system_name: str = Field(..., description="Display name used for the casino system narrator.")
    system_avatar_url: str = Field(
        default="", description="Avatar URL used for the casino system narrator."
    )


class ParticipantPreparationResult(BaseModel):
    """Result of preparing a Discord user for a wagered game seat.

    `balance` rides along even when preparation failed, because the refusal the user sees names
    the balance that was too small, and re-reading it would be a second query against a value
    that may already have moved.

    Attributes:
        participant: Prepared game participant, or None when preparation failed.
        balance: Player balance observed during preparation.
    """

    model_config = ConfigDict(frozen=True)

    participant: GameParticipant | None = Field(
        ..., description="Prepared game participant, or None when preparation failed."
    )
    balance: int = Field(..., description="Player balance observed during preparation.")


class RefreshParticipantsResult(BaseModel):
    """Result of re-checking every queued seat against a fresh balance before a lobby starts.

    A player who spent their balance between joining and the start is dropped rather than
    seated at a stake they can no longer cover, and is named back to whoever pressed Start;
    `dropped_names` therefore carries display names, since that notice is its only reader.

    Attributes:
        participants: Players still eligible to start the round.
        dropped_names: Display names of players dropped during the re-check.
    """

    model_config = ConfigDict(frozen=True)

    participants: list[GameParticipant] = Field(
        default_factory=list, description="Players still eligible to start the round."
    )
    dropped_names: list[str] = Field(
        default_factory=list, description="Display names of players dropped during the re-check."
    )


class WagerSettlement(BaseModel):
    """What one atomic round settlement left behind, read back for the final embed.

    The player wallet and the casino ledger move in one transaction, so the two balances here
    are mutually consistent once written. Neither reconciles against `delta`, which is the
    requested change: a loss larger than the balance is clamped at zero, and only the part
    actually collected is mirrored into the ledger. A wager is never debited when the round
    starts, so this is the whole of the money a round moved, and an unfinished in-memory round
    vanishing on restart costs nobody anything.

    Attributes:
        delta: Net point change for the round.
        payout: Positive player credit from the round, excluding losses and pushes.
        new_balance: Player balance after applying the signed round delta.
        casino_balance: Casino ledger balance after applying the casino-side settlement.
        base_delta: Net point change before any VIP payout bonus. `None` for
            legacy/manual test settlements that do not carry bonus details.
        vip_bonus: Extra points added by the VIP payout bonus.
        is_vip: Whether the VIP perk was active for this settlement.
    """

    model_config = ConfigDict(frozen=True)

    delta: int = Field(..., description="Net point change for the round.")
    payout: int = Field(
        ..., description="Positive player credit from the round, excluding losses and pushes."
    )
    new_balance: int = Field(
        ..., description="Player balance after applying the signed round delta."
    )
    casino_balance: int = Field(
        ..., description="Casino ledger balance after applying the casino-side settlement."
    )
    base_delta: int | None = Field(
        default=None,
        description=(
            "Net point change before any VIP payout bonus; None for legacy/manual test "
            "settlements that do not carry bonus details."
        ),
    )
    vip_bonus: int = Field(default=0, description="Extra points added by the VIP payout bonus.")
    is_vip: bool = Field(
        default=False, description="Whether the VIP perk was active for this settlement."
    )


class BlackjackHandSettlement(BaseModel):
    """Per-hand result for one sub-hand of a Blackjack player.

    Split turns a single participant into two settlement rows; otherwise
    each player has exactly one `BlackjackHandSettlement` aggregated into
    their `BlackjackPlayerSettlement`.

    `delta` and `five_card_bonus` are kept apart because they come out of different pockets:
    the delta is what the casino pays and mirrors into its ledger, while the five-card 21 bonus
    is minted by the system, credits the player, and never moves `/casino`.

    Attributes:
        cards: Cards held by this sub-hand at settlement time.
        bet: Effective wager for this hand (doubled bets land here as 2x).
        outcome: Player-facing outcome label for this sub-hand.
        delta: Dealer-paid signed point change for this single hand before
            VIP and five-card bonuses.
        five_card_bonus: System-funded bonus for a five-card 21.
        five_card_twenty_one: True when this hand made five or more cards
            totaling 21.
        doubled: True if this hand was doubled.
        surrendered: True if this hand was surrendered.
        is_split_hand: True if this hand came out of a Split.
    """

    model_config = ConfigDict(frozen=True)

    cards: list[Card] = Field(..., description="Cards held by this sub-hand at settlement time.")
    bet: int = Field(
        ..., description="Effective wager for this hand (doubled bets land here as 2x)."
    )
    outcome: SettleOutcome = Field(
        ..., description="Player-facing outcome label for this sub-hand."
    )
    delta: int = Field(
        ...,
        description=(
            "Dealer-paid signed point change for this single hand before VIP and "
            "five-card bonuses."
        ),
    )
    five_card_bonus: int = Field(default=0, description="System-funded bonus for a five-card 21.")
    five_card_twenty_one: bool = Field(
        default=False, description="True when this hand made five or more cards totaling 21."
    )
    doubled: bool = Field(default=False, description="True if this hand was doubled.")
    surrendered: bool = Field(default=False, description="True if this hand was surrendered.")
    is_split_hand: bool = Field(
        default=False, description="True if this hand came out of a Split."
    )


class BlackjackInsuranceSettlement(BaseModel):
    """Insurance side-bet result for one player, settled off the hole-card peek.

    The peek runs at the close of the insurance phase, so the outcome is already fixed before
    any player acts, and a peeked Blackjack ends the round on the spot with no hand ever
    played. The row itself is built with the hands at the end and folded into the player's
    single settlement write.

    Attributes:
        bet: Insurance bet amount (half the original wager).
        won: True only when the dealer's hole-card peek was a Blackjack.
        delta: Signed point change for this side bet (`+bet*2` on win,
            `-bet` on loss).
    """

    model_config = ConfigDict(frozen=True)

    bet: int = Field(..., description="Insurance bet amount (half the original wager).")
    won: bool = Field(
        ..., description="True only when the dealer's hole-card peek was a Blackjack."
    )
    delta: int = Field(
        ..., description="Signed point change for this side bet (+bet*2 on win, -bet on loss)."
    )


class BlackjackPlayerSettlement(WagerSettlement):
    """Aggregated Blackjack settlement for one participant.

    Combines every sub-hand result plus any insurance side bet into a
    single point delta and the one `apply_round_settlement` write that
    backs it. A split player therefore still moves money exactly once,
    and the inherited `delta` is the only figure the wallet ever saw.

    `detail` is a leftover: it is still built at settlement time, but nothing renders or
    persists it now that the casino has no narrator and the bot player runs on no LLM.

    Attributes:
        outcome: Aggregate player-facing outcome. Single-hand results without
            insurance preserve the hand outcome; insurance and multi-hand
            results collapse to win / lose / push by net base delta.
        detail: Short Chinese round summary; currently built but unread.
        hands: Per-hand settlements in display order.
        insurance: Insurance side-bet result, or `None` when the player
            never took insurance.
        five_card_bonus: Aggregate system-funded five-card 21 bonus.
    """

    outcome: SettleOutcome = Field(
        ...,
        description=(
            "Aggregate player-facing outcome. Single-hand results without insurance preserve "
            "the hand outcome; insurance and multi-hand results collapse to win / lose / push "
            "by net base delta."
        ),
    )
    detail: str = Field(
        ..., description="Short Chinese round summary; currently built but unread."
    )
    hands: list[BlackjackHandSettlement] = Field(
        default_factory=list, description="Per-hand settlements in display order."
    )
    insurance: BlackjackInsuranceSettlement | None = Field(
        default=None,
        description="Insurance side-bet result, or None when the player never took insurance.",
    )
    five_card_bonus: int = Field(
        default=0, description="Aggregate system-funded five-card 21 bonus."
    )


class BlackjackPlayerResult(BaseModel):
    """One seat and what it settled for, the pairing both the final embeds and history need.

    A settlement carries no identity of its own, so this is what lets a seat embed and a
    persisted row name the player the numbers belong to.

    Attributes:
        participant: Player identity and wager metadata.
        settlement: Database-backed result for that player's hand.
    """

    model_config = ConfigDict(frozen=True)

    participant: GameParticipant = Field(..., description="Player identity and wager metadata.")
    settlement: BlackjackPlayerSettlement = Field(
        ..., description="Database-backed result for that player's hand."
    )


class BlackjackHistoryHand(BaseModel):
    """One sub-hand snapshot persisted in a Blackjack round-history record.

    A projection of `BlackjackHandSettlement` rather than a reuse of it, so the settlement
    shapes stay free to change without invalidating rows already written. `total` is the one
    field it adds: the value is computed once at write time, so reading a hand back never has
    to re-run the ace-soft/hard evaluation the rules engine owns.

    Attributes:
        cards: Cards held by this sub-hand at settlement time.
        total: Final hand value for this sub-hand (bust totals exceed 21).
        bet: Effective wager for this hand (doubled bets land here as 2x).
        outcome: Player-facing outcome label for this sub-hand.
        delta: Dealer-paid signed point change for this single hand.
        five_card_bonus: System-funded bonus for a five-card 21.
        five_card_twenty_one: True when this hand made five or more cards totaling 21.
        doubled: True if this hand was doubled.
        surrendered: True if this hand was surrendered.
        is_split_hand: True if this hand came out of a Split.
    """

    model_config = ConfigDict(frozen=True)

    cards: list[Card] = Field(..., description="Cards held by this sub-hand at settlement time.")
    total: int = Field(
        ..., description="Final hand value for this sub-hand (bust totals exceed 21)."
    )
    bet: int = Field(
        ..., description="Effective wager for this hand (doubled bets land here as 2x)."
    )
    outcome: SettleOutcome = Field(
        ..., description="Player-facing outcome label for this sub-hand."
    )
    delta: int = Field(..., description="Dealer-paid signed point change for this single hand.")
    five_card_bonus: int = Field(default=0, description="System-funded bonus for a five-card 21.")
    five_card_twenty_one: bool = Field(
        default=False, description="True when this hand made five or more cards totaling 21."
    )
    doubled: bool = Field(default=False, description="True if this hand was doubled.")
    surrendered: bool = Field(default=False, description="True if this hand was surrendered.")
    is_split_hand: bool = Field(
        default=False, description="True if this hand came out of a Split."
    )


class BlackjackHistoryInsurance(BaseModel):
    """Insurance side-bet snapshot persisted in a Blackjack round-history record.

    The persisted twin of `BlackjackInsuranceSettlement`, kept separate for the same reason
    `BlackjackHistoryHand` is.

    Attributes:
        bet: Insurance bet amount (half the original wager).
        won: True only when the dealer's hole-card peek was a Blackjack.
        delta: Signed point change for this side bet.
    """

    model_config = ConfigDict(frozen=True)

    bet: int = Field(..., description="Insurance bet amount (half the original wager).")
    won: bool = Field(
        ..., description="True only when the dealer's hole-card peek was a Blackjack."
    )
    delta: int = Field(..., description="Signed point change for this side bet.")


class BlackjackHistoryPayload(BaseModel):
    """Full per-player round snapshot serialized into a history row's JSON column.

    Everything the renderer needs beyond the flat columns lives here, including the dealer hand
    the row's player was measured against. Every field carries a default, so a row written
    before a field existed still validates when it is read back.

    Attributes:
        hands: Per-hand snapshots in display order (one entry, or two after a Split).
        dealer_cards: Dealer's final hand at settlement time.
        dealer_total: Dealer's final hand value.
        insurance: Insurance side-bet snapshot, or None when never taken.
        vip_bonus: Extra points added by the VIP payout bonus.
        five_card_bonus: Aggregate system-funded five-card 21 bonus.
        balance_at_start: Player balance observed when the round started.
        new_balance: Player balance after applying the round delta.
    """

    model_config = ConfigDict(frozen=True)

    hands: list[BlackjackHistoryHand] = Field(
        default_factory=list, description="Per-hand snapshots in display order."
    )
    dealer_cards: list[Card] = Field(
        default_factory=list, description="Dealer's final hand at settlement time."
    )
    dealer_total: int = Field(default=0, description="Dealer's final hand value.")
    insurance: BlackjackHistoryInsurance | None = Field(
        default=None, description="Insurance side-bet snapshot, or None when never taken."
    )
    vip_bonus: int = Field(default=0, description="Extra points added by the VIP payout bonus.")
    five_card_bonus: int = Field(
        default=0, description="Aggregate system-funded five-card 21 bonus."
    )
    balance_at_start: int = Field(
        default=0, description="Player balance observed when the round started."
    )
    new_balance: int = Field(
        default=0, description="Player balance after applying the round delta."
    )


class BlackjackHistoryRecord(BaseModel):
    """One persisted Blackjack round result for a player, read back for `/games blackjack_history`.

    One row per seated player, the bot included, and `round_id` is what ties the rows of a
    single round together. `user_id`, `created_at`, `outcome` and `delta` are the flat columns
    the read path filters, orders and summarises on; everything richer sits in `payload`, so
    adding a detail to a round costs no schema change.

    Attributes:
        round_id: Shared identifier for every player row of the same round.
        channel_id: Discord channel the round was played in.
        guild_id: Discord guild the round was played in, or 0 for DMs.
        message_id: Discord message id of the settled table.
        user_id: Discord user id of the player.
        user_name: Stored Discord username of the player.
        is_bot: True when this row belongs to the bot player.
        is_vip: True when the VIP perk was active for this settlement.
        bet: Base wager for the player this round.
        outcome: Aggregate player-facing outcome for the round.
        delta: Net signed point change for the round.
        payload: Full per-player round snapshot used by the history renderer.
        created_at: Asia/Taipei timestamp the round settled at.
    """

    model_config = ConfigDict(frozen=True)

    round_id: str = Field(
        ..., description="Shared identifier for every player row of the same round."
    )
    channel_id: int = Field(..., description="Discord channel the round was played in.")
    guild_id: int = Field(..., description="Discord guild the round was played in, or 0 for DMs.")
    message_id: int = Field(..., description="Discord message id of the settled table.")
    user_id: int = Field(..., description="Discord user id of the player.")
    user_name: str = Field(..., description="Stored Discord username of the player.")
    is_bot: bool = Field(..., description="True when this row belongs to the bot player.")
    is_vip: bool = Field(..., description="True when the VIP perk was active for this settlement.")
    bet: int = Field(..., description="Base wager for the player this round.")
    outcome: SettleOutcome = Field(
        ..., description="Aggregate player-facing outcome for the round."
    )
    delta: int = Field(..., description="Net signed point change for the round.")
    payload: BlackjackHistoryPayload = Field(
        ..., description="Full per-player round snapshot used by the history renderer."
    )
    created_at: datetime = Field(..., description="Asia/Taipei timestamp the round settled at.")


class DealerOutcome(BaseModel):
    """Dealer final-total distribution under H17 over a no-replacement shoe.

    The six probabilities are mutually exclusive and sum to ~1.0. The one that leaves the EV
    engine on an `ActionEvAnalysis` is always the marginal estimate: read off the up-card with a
    hypothetical hole integrated out over the remaining shoe, and conditioned on no Blackjack
    when the dealer already peeked under an Ace or ten. So the bot player can weigh stand
    against hit without the estimate ever depending on the actual hole.
    """

    model_config = ConfigDict(frozen=True)

    bust_probability: float = Field(
        ..., description="Probability the dealer busts (final total over 21)."
    )
    total_17_probability: float = Field(
        ..., description="Probability the dealer's final total is exactly 17."
    )
    total_18_probability: float = Field(
        ..., description="Probability the dealer's final total is exactly 18."
    )
    total_19_probability: float = Field(
        ..., description="Probability the dealer's final total is exactly 19."
    )
    total_20_probability: float = Field(
        ..., description="Probability the dealer's final total is exactly 20."
    )
    total_21_probability: float = Field(
        ..., description="Probability the dealer's final total is exactly 21."
    )


class ActionEv(BaseModel):
    """Expected value of one Blackjack action, in units of the base hand bet.

    Split is the only action whose value is an approximation (the two hands are priced as if
    they drew from independent shoes, which runs slightly optimistic), so `is_estimate` and
    `note` exist to keep that visible rather than letting an over-stated number pass as exact.
    """

    model_config = ConfigDict(frozen=True)

    action: BotAction = Field(..., description="The action this expected value is computed for.")
    expected_value: float = Field(
        ..., description="Expected net return in multiples of the base hand bet; higher is better."
    )
    is_estimate: bool = Field(
        default=False,
        description="True when the value is an approximation rather than exact (split).",
    )
    note: str | None = Field(
        default=None, description="Optional caveat describing why a value is an estimate."
    )


class ActionEvAnalysis(BaseModel):
    """EV analysis for one bot-player action decision, split along an information boundary.

    `dealer_outcome`, `action_evs` and `recommended_expected_value` come from the marginal pass,
    which integrates a hypothetical hole out over the unseen deck, so they depend only on the
    up-card and the remaining shoe. `recommended_action` is picked by the engine's private
    hole-aware pass and is the bot's private informational edge on this decision (its bet
    sizing and its insurance call are separate edges that never read this model). Keeping the
    exposed numbers marginal is what stops anything that reads this analysis from
    reconstructing the dealer's hole card, which is also why the recommended action's reported
    EV is its marginal one rather than its exact one.
    """

    model_config = ConfigDict(frozen=True)

    dealer_outcome: DealerOutcome = Field(
        ..., description="Marginalized dealer final-total distribution shown to the model."
    )
    action_evs: tuple[ActionEv, ...] = Field(
        ...,
        description="Per-allowed-action marginalized expected values, ordered highest to lowest EV.",
    )
    recommended_action: BotAction = Field(
        ...,
        description="EV-maximizing legal action (split is only recommended past a safety margin).",
    )
    recommended_expected_value: float = Field(
        ...,
        description="Marginalized expected value of the recommended action, in base-bet units.",
    )


class BlackjackDealerStep(BaseModel):
    """One dealer action recorded during the Blackjack dealer phase.

    Appended as the H17 loop runs and rendered under the settled table as a compact decision
    path, so a player can check the dealer against the rules it claims to follow. The rendered
    part is the `source` label, the totals and any drawn card; `reason`, `forced` and
    `fallback` are recorded only. `forced` is true on every step the loop writes, rule-driven
    ones included, so `source` is what separates the step-limit guard from the rule engine, and
    `fallback` has no writer at all today.
    """

    model_config = ConfigDict(frozen=True)

    total_before: int = Field(..., description="Dealer hand total before this action.")
    action: BlackjackDealerAction = Field(..., description="Dealer hit or stand action taken.")
    reason: str = Field(..., description="Rationale recorded for this dealer action.")
    source: BlackjackDealerStepSource = Field(
        default="auto", description="Whether the action came from the auto engine or a guard."
    )
    drawn_card: Card | None = Field(
        default=None, description="Card drawn on a hit, or None for a stand."
    )
    total_after: int | None = Field(
        default=None, description="Dealer hand total after this action, when applicable."
    )
    fallback: bool = Field(
        default=False, description="True when this step came from a fallback path."
    )
    forced: bool = Field(default=False, description="True when this step was forced by a guard.")


class DragonGatePlayerResult(BaseModel):
    """Final outcome for one player after a 射龍門 table closes.

    Each bet settles into the player row and the shared jackpot pool the
    moment it's placed, so the table close-out has no per-player wager
    settlement to apply; this model just captures the running totals and
    whether "逆贏不拿" was triggered for the leaver.

    Attributes:
        participant: Player identity and ante metadata.
        delta: Running win/loss for the table (ante excluded; ante was
            already pushed into the jackpot when the round started).
        final_balance: Player balance after the last settlement event
            touching this account.
        withdrawn: True when the player left voluntarily before timeout
            or pool exhaustion.
        refunded_to_pool: Amount refunded into the jackpot under
            "逆贏不拿" when the player left while ahead.
    """

    model_config = ConfigDict(frozen=True)

    participant: GameParticipant = Field(..., description="Player identity and ante metadata.")
    delta: int = Field(
        ...,
        description=(
            "Running win/loss for the table (ante excluded; ante was already pushed into "
            "the jackpot when the round started)."
        ),
    )
    final_balance: int = Field(
        ..., description="Player balance after the last settlement event touching this account."
    )
    withdrawn: bool = Field(
        ..., description="True when the player left voluntarily before timeout or pool exhaustion."
    )
    refunded_to_pool: int = Field(
        default=0,
        description='Amount refunded into the jackpot under "逆贏不拿" when the player left while ahead.',
    )


__all__ = [
    "ActionEv",
    "ActionEvAnalysis",
    "BlackjackDealerAction",
    "BlackjackDealerStep",
    "BlackjackHandSettlement",
    "BlackjackInsuranceSettlement",
    "BlackjackPlayerResult",
    "BlackjackPlayerSettlement",
    "BotAction",
    "Card",
    "DealerOutcome",
    "DragonGatePlayerResult",
    "GameKind",
    "GameParticipant",
    "GameParticipantIdentity",
    "ParticipantPreparationResult",
    "RefreshParticipantsResult",
    "SettleOutcome",
    "SystemIdentity",
    "WagerSettlement",
]
