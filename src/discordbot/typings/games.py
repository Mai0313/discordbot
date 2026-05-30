from typing import Literal

from pydantic import Field, BaseModel, ConfigDict

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
GameKind = Literal["blackjack", "dragon_gate"]
BlackjackDealerAction = Literal["hit", "stand"]
BlackjackDealerStepSource = Literal["auto", "guard"]
BotAction = Literal["hit", "stand", "double", "split", "surrender"]


class Card(BaseModel):
    """A single playing card.

    Attributes:
        rank: One of A, 2-10, J, Q, K.
        suit: One of the four unicode suit glyphs.
    """

    model_config = ConfigDict(frozen=True)

    rank: str = Field(description="Card rank: one of A, 2-10, J, Q, K.")
    suit: str = Field(description="One of the four unicode suit glyphs.")

    def __str__(self) -> str:
        """Human-readable label like `A♠`."""
        return f"{self.rank}{self.suit}"


class GameParticipant(BaseModel):
    """A Discord user registered for a casino game session.

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

    user_id: int = Field(description="Discord user ID for the account row and interaction checks.")
    account_name: str = Field(
        description="Stable Discord username stored in the economy account row."
    )
    display_name: str = Field(description="Guild-aware display name shown in game embeds.")
    avatar_url: str = Field(
        default="", description="Last-seen Discord avatar URL for the economy account row."
    )
    bet: int = Field(description="Effective wager for this player.")
    balance_at_start: int = Field(description="Balance observed when the game session starts.")
    is_allin: bool = Field(
        description="True when the effective wager consumes the full observed balance."
    )


class GameParticipantIdentity(BaseModel):
    """Stable Discord identity for constructing a game participant."""

    model_config = ConfigDict(frozen=True)

    user_id: int = Field(description="Discord user ID for the account row and interaction checks.")
    account_name: str = Field(
        description="Stable Discord username stored in the economy account row."
    )
    display_name: str = Field(description="Guild-aware display name shown in game embeds.")
    avatar_url: str = Field(
        default="", description="Last-seen Discord avatar URL for the economy account row."
    )


class SystemIdentity(BaseModel):
    """Discord identity used for the casino system narrator in game views."""

    model_config = ConfigDict(frozen=True)

    system_id: int = Field(description="Discord user ID used for the casino system narrator.")
    system_name: str = Field(description="Display name used for the casino system narrator.")
    system_avatar_url: str = Field(
        default="", description="Avatar URL used for the casino system narrator."
    )


class ParticipantPreparationResult(BaseModel):
    """Result of preparing a Discord user for a wagered game seat."""

    model_config = ConfigDict(frozen=True)

    participant: GameParticipant | None = Field(
        description="Prepared game participant, or None when preparation failed."
    )
    balance: int = Field(description="Player balance observed during preparation.")


class RefreshParticipantsResult(BaseModel):
    """Result of re-checking seated players before a lobby starts."""

    model_config = ConfigDict(frozen=True)

    participants: list[GameParticipant] = Field(
        default_factory=list, description="Players still eligible to start the round."
    )
    dropped_names: list[str] = Field(
        default_factory=list, description="Display names of players dropped during the re-check."
    )


class WagerSettlement(BaseModel):
    """Database-backed settlement result for a finished wager.

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

    delta: int = Field(description="Net point change for the round.")
    payout: int = Field(
        description="Positive player credit from the round, excluding losses and pushes."
    )
    new_balance: int = Field(description="Player balance after applying the signed round delta.")
    casino_balance: int = Field(
        description="Casino ledger balance after applying the casino-side settlement."
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

    cards: list[Card] = Field(description="Cards held by this sub-hand at settlement time.")
    bet: int = Field(description="Effective wager for this hand (doubled bets land here as 2x).")
    outcome: SettleOutcome = Field(description="Player-facing outcome label for this sub-hand.")
    delta: int = Field(
        description=(
            "Dealer-paid signed point change for this single hand before VIP and "
            "five-card bonuses."
        )
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
    """Insurance side-bet result for one player.

    Attributes:
        bet: Insurance bet amount (half the original wager).
        won: True only when the dealer's hole-card peek was a Blackjack.
        delta: Signed point change for this side bet (`+bet*2` on win,
            `-bet` on loss).
    """

    model_config = ConfigDict(frozen=True)

    bet: int = Field(description="Insurance bet amount (half the original wager).")
    won: bool = Field(description="True only when the dealer's hole-card peek was a Blackjack.")
    delta: int = Field(
        description="Signed point change for this side bet (+bet*2 on win, -bet on loss)."
    )


class BlackjackPlayerSettlement(WagerSettlement):
    """Aggregated Blackjack settlement for one participant.

    Combines every sub-hand result plus any insurance side bet into a
    single point delta and the one `apply_round_settlement` write that
    backs it.

    Attributes:
        outcome: Aggregate player-facing outcome. Single-hand results without
            insurance preserve the hand outcome; insurance and multi-hand
            results collapse to win / lose / push by net base delta.
        hands: Per-hand settlements in display order.
        insurance: Insurance side-bet result, or `None` when the player
            never took insurance.
        detail: Short game-state summary for the dealer AI prompt.
        five_card_bonus: Aggregate system-funded five-card 21 bonus.
    """

    outcome: SettleOutcome = Field(
        description=(
            "Aggregate player-facing outcome. Single-hand results without insurance preserve "
            "the hand outcome; insurance and multi-hand results collapse to win / lose / push "
            "by net base delta."
        )
    )
    detail: str = Field(description="Short game-state summary for the dealer AI prompt.")
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
    """Settlement result for one player at a Blackjack table.

    Attributes:
        participant: Player identity and wager metadata.
        settlement: Database-backed result for that player's hand.
    """

    model_config = ConfigDict(frozen=True)

    participant: GameParticipant = Field(description="Player identity and wager metadata.")
    settlement: BlackjackPlayerSettlement = Field(
        description="Database-backed result for that player's hand."
    )


class BlackjackDealerDecision(BaseModel):
    """Structured hit / stand decision returned by the AI Blackjack dealer."""

    model_config = ConfigDict(frozen=True)

    action: BlackjackDealerAction = Field(description="Dealer hit or stand decision.")
    reason: str = Field(description="Rationale for the dealer's decision.")


class BotPlayerBetDecision(BaseModel):
    """Structured bet decision returned by the bot player AI."""

    model_config = ConfigDict(frozen=True)

    bet_amount: int = Field(ge=1, description="Chosen wager; at least 1.")
    reason: str = Field(description="Short Traditional Chinese rationale shown on the bot's seat.")


class BotPlayerActionDecision(BaseModel):
    """Structured hit / stand / double / split / surrender decision."""

    model_config = ConfigDict(frozen=True)

    action: BotAction = Field(description="Chosen Blackjack action for the current hand.")
    reason: str = Field(description="Short Traditional Chinese rationale shown on the bot's seat.")


class BotPlayerInsuranceDecision(BaseModel):
    """Structured insurance-take / decline decision."""

    model_config = ConfigDict(frozen=True)

    take_insurance: bool = Field(description="Whether the bot takes the insurance side bet.")
    reason: str = Field(description="Short Traditional Chinese rationale shown on the bot's seat.")


class BotFinancialContext(BaseModel):
    """Bot's lifetime + daily financial snapshot for decision context.

    `balance` is the spendable wallet today; `total_earned` / `total_spent` are
    lifetime gross flows. The three `daily_*` fields are zero outside today's
    Taipei calendar day so the model treats yesterday's loss as already
    "forgotten" — same convention as the loss leaderboard.
    """

    model_config = ConfigDict(frozen=True)

    balance: int = Field(description="Spendable wallet balance today.")
    total_earned: int = Field(description="Lifetime gross points earned.")
    total_spent: int = Field(description="Lifetime gross points spent.")
    daily_loss: int = Field(description="Casino loss accrued during today's Taipei calendar day.")
    daily_win: int = Field(description="Casino win accrued during today's Taipei calendar day.")
    daily_net: int = Field(description="Net casino result during today's Taipei calendar day.")


class OtherPlayerView(BaseModel):
    """One non-bot player's table state visible to the bot."""

    model_config = ConfigDict(frozen=True)

    display_name: str = Field(description="Guild-aware display name of the other player.")
    bet: int = Field(description="That player's effective wager.")
    hands: list[str] = Field(description="Human-readable summaries of that player's hands.")
    is_finished: bool = Field(description="True when that player has finished their turn.")


class DealerOutcome(BaseModel):
    """Exact dealer final-total distribution under H17 over a no-replacement shoe.

    The six probabilities are mutually exclusive and sum to ~1.0. They are
    computed from the dealer's known two cards (hole + up) and the true
    remaining shoe, so the bot player can reason about stand-versus-hit from
    the actual dealer outcome instead of guessing.
    """

    model_config = ConfigDict(frozen=True)

    bust_probability: float = Field(
        description="Probability the dealer busts (final total over 21)."
    )
    total_17_probability: float = Field(
        description="Probability the dealer's final total is exactly 17."
    )
    total_18_probability: float = Field(
        description="Probability the dealer's final total is exactly 18."
    )
    total_19_probability: float = Field(
        description="Probability the dealer's final total is exactly 19."
    )
    total_20_probability: float = Field(
        description="Probability the dealer's final total is exactly 20."
    )
    total_21_probability: float = Field(
        description="Probability the dealer's final total is exactly 21."
    )


class ActionEv(BaseModel):
    """Expected value of one Blackjack action, in units of the base hand bet."""

    model_config = ConfigDict(frozen=True)

    action: BotAction = Field(description="The action this expected value is computed for.")
    expected_value: float = Field(
        description="Expected net return in multiples of the base hand bet; higher is better."
    )
    is_estimate: bool = Field(
        default=False,
        description="True when the value is an approximation rather than exact (split).",
    )
    note: str | None = Field(
        default=None, description="Optional caveat describing why a value is an estimate."
    )


class ActionEvAnalysis(BaseModel):
    """Hole-card-aware exact EV analysis for one bot-player action decision."""

    model_config = ConfigDict(frozen=True)

    dealer_outcome: DealerOutcome = Field(description="Exact dealer final-total distribution.")
    action_evs: tuple[ActionEv, ...] = Field(
        description="Per-allowed-action expected values, ordered from highest to lowest EV."
    )
    recommended_action: BotAction = Field(
        description="EV-maximizing legal action (split is only recommended past a safety margin)."
    )
    recommended_expected_value: float = Field(
        description="Expected value of the recommended action, in base-bet units."
    )


class BlackjackDealerStep(BaseModel):
    """One dealer action recorded during the Blackjack dealer phase."""

    model_config = ConfigDict(frozen=True)

    total_before: int = Field(description="Dealer hand total before this action.")
    action: BlackjackDealerAction = Field(description="Dealer hit or stand action taken.")
    reason: str = Field(description="Rationale recorded for this dealer action.")
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

    participant: GameParticipant = Field(description="Player identity and ante metadata.")
    delta: int = Field(
        description=(
            "Running win/loss for the table (ante excluded; ante was already pushed into "
            "the jackpot when the round started)."
        )
    )
    final_balance: int = Field(
        description="Player balance after the last settlement event touching this account."
    )
    withdrawn: bool = Field(
        description="True when the player left voluntarily before timeout or pool exhaustion."
    )
    refunded_to_pool: int = Field(
        default=0,
        description='Amount refunded into the jackpot under "逆贏不拿" when the player left while ahead.',
    )


__all__ = [
    "ActionEv",
    "ActionEvAnalysis",
    "BlackjackDealerAction",
    "BlackjackDealerDecision",
    "BlackjackDealerStep",
    "BlackjackHandSettlement",
    "BlackjackInsuranceSettlement",
    "BlackjackPlayerResult",
    "BlackjackPlayerSettlement",
    "BotAction",
    "BotFinancialContext",
    "BotPlayerActionDecision",
    "BotPlayerBetDecision",
    "BotPlayerInsuranceDecision",
    "Card",
    "DealerOutcome",
    "DragonGatePlayerResult",
    "GameKind",
    "GameParticipant",
    "GameParticipantIdentity",
    "OtherPlayerView",
    "ParticipantPreparationResult",
    "RefreshParticipantsResult",
    "SettleOutcome",
    "SystemIdentity",
    "WagerSettlement",
]
