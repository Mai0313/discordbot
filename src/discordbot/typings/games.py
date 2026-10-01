"""Shared result types, enums, and payloads for the casino games."""

from typing import Literal
from datetime import datetime

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
BlackjackDealerAction = Literal["hit", "stand"]
BotAction = Literal["hit", "stand", "double", "split", "surrender"]


class Card(BaseModel):
    """A single playing card."""

    model_config = ConfigDict(frozen=True)

    rank: str = Field(..., description="Card rank: one of A, 2-10, J, Q, K.")
    suit: str = Field(..., description="One of the four unicode suit glyphs.")

    def __str__(self) -> str:
        """Human-readable label like `A♠`."""
        return f"{self.rank}{self.suit}"


class GameParticipantIdentity(BaseModel):
    """Stable Discord identity for constructing a game participant."""

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


class GameParticipant(GameParticipantIdentity):
    """A Discord user registered for a casino game session."""

    bet: int = Field(..., description="Effective wager for this player.")
    balance_at_start: int = Field(
        ..., description="Balance observed when the game session starts."
    )
    is_allin: bool = Field(
        ..., description="True when the effective wager consumes the full observed balance."
    )


class ParticipantPreparationResult(BaseModel):
    """Result of preparing a Discord user for a wagered game seat."""

    model_config = ConfigDict(frozen=True)

    participant: GameParticipant | None = Field(
        ..., description="Prepared game participant, or None when preparation failed."
    )
    balance: int = Field(..., description="Player balance observed during preparation.")


class RefreshParticipantsResult(BaseModel):
    """Result of re-checking seated players before a lobby starts."""

    model_config = ConfigDict(frozen=True)

    participants: list[GameParticipant] = Field(
        default_factory=list, description="Players still eligible to start the round."
    )
    dropped_names: list[str] = Field(
        default_factory=list, description="Display names of players dropped during the re-check."
    )


class BlackjackHandSettlement(BaseModel):
    """Per-hand result for one sub-hand of a Blackjack player.

    Split turns a single participant into two settlement rows; otherwise
    each player has exactly one `BlackjackHandSettlement` aggregated into
    their `BlackjackPlayerSettlement`.
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
    """Insurance side-bet result for one player."""

    model_config = ConfigDict(frozen=True)

    bet: int = Field(..., description="Insurance bet amount (half the original wager).")
    won: bool = Field(
        ..., description="True only when the dealer's hole-card peek was a Blackjack."
    )
    delta: int = Field(
        ..., description="Signed point change for this side bet (+bet*2 on win, -bet on loss)."
    )


class BlackjackPlayerSettlement(BaseModel):
    """Aggregated Blackjack settlement for one participant.

    Combines every sub-hand result plus any insurance side bet into a
    single point delta and the one database write that backs it.
    """

    model_config = ConfigDict(frozen=True)

    delta: int = Field(
        ...,
        description="Net point change actually applied for the round; a loss may be smaller than the rules' amount when the wallet cannot cover it.",
    )
    new_balance: int = Field(
        ..., description="Player balance after applying the signed round delta."
    )
    casino_balance: int = Field(
        ..., description="Casino ledger balance after applying the casino-side settlement."
    )
    base_delta: int = Field(..., description="Net point change before any VIP payout bonus.")
    vip_bonus: int = Field(default=0, description="Extra points added by the VIP payout bonus.")
    is_vip: bool = Field(
        default=False, description="Whether the VIP perk was active for this settlement."
    )
    outcome: SettleOutcome = Field(
        ...,
        description=(
            "Aggregate player-facing outcome. Single-hand results without insurance preserve "
            "the hand outcome; insurance and multi-hand results collapse to win / lose / push "
            "by net base delta."
        ),
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
    """Settlement result for one player at a Blackjack table."""

    model_config = ConfigDict(frozen=True)

    participant: GameParticipant = Field(..., description="Player identity and wager metadata.")
    settlement: BlackjackPlayerSettlement = Field(
        ..., description="Database-backed result for that player's hand."
    )


class BlackjackHistoryHand(BaseModel):
    """One sub-hand snapshot persisted in a Blackjack round-history record."""

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
    """Insurance side-bet snapshot persisted in a Blackjack round-history record."""

    model_config = ConfigDict(frozen=True)

    bet: int = Field(..., description="Insurance bet amount (half the original wager).")
    won: bool = Field(
        ..., description="True only when the dealer's hole-card peek was a Blackjack."
    )
    delta: int = Field(..., description="Signed point change for this side bet.")


class BlackjackHistoryPayload(BaseModel):
    """Full per-player round snapshot serialized into a history row's JSON column."""

    model_config = ConfigDict(frozen=True)

    hands: list[BlackjackHistoryHand] = Field(
        default_factory=list,
        description="Per-hand snapshots in display order (one entry, or two after a Split).",
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
    """One persisted Blackjack round result for a player, read back for display."""

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


class ActionEv(BaseModel):
    """Expected value of one Blackjack action, in units of the base hand bet."""

    model_config = ConfigDict(frozen=True)

    action: BotAction = Field(..., description="The action this expected value is computed for.")
    expected_value: float = Field(
        ..., description="Expected net return in multiples of the base hand bet; higher is better."
    )


class BlackjackDealerStep(BaseModel):
    """One dealer action recorded during the Blackjack dealer phase."""

    model_config = ConfigDict(frozen=True)

    total_before: int = Field(..., description="Dealer hand total before this action.")
    action: BlackjackDealerAction = Field(..., description="Dealer hit or stand action taken.")
    drawn_card: Card | None = Field(
        default=None, description="Card drawn on a hit, or None for a stand."
    )
    total_after: int | None = Field(
        default=None, description="Dealer hand total after this action, when applicable."
    )
