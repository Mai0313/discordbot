"""In-memory per-channel persistent Blackjack shoe for cross-round card counting.

Shoes carry over between rounds in the same Discord channel and reset on bot restart.
"""

from random import Random
from typing import Final

from pydantic import Field, BaseModel, PrivateAttr

from discordbot.typings.games import Card
from discordbot.cogs.games.blackjack import build_shoe
from discordbot.cogs.games.blackjack_ev import compute_true_count

# Reshuffle before a round when fewer cards remain. Bounded below by the most cards one round
# can draw (avoiding `draw_card`'s infinite fallback) and above by counting penetration
# (96 of 208 cards leaves ~54% dealt).
RESHUFFLE_THRESHOLD_CARDS: Final[int] = 96


class BlackjackShoeStore(BaseModel):
    """Per-channel persistent shoe store keyed by Discord channel id.

    Mutating methods are synchronous and atomic on the single-threaded event loop; concurrent
    games in one channel each get a fresh shoe rather than sharing a list.
    """

    shoes: dict[int, list[Card]] = Field(
        default_factory=dict,
        description="Persistent remaining shoe cards keyed by Discord channel id.",
    )
    _take_generation: dict[int, int] = PrivateAttr(default_factory=dict)
    _saved_generation: dict[int, int] = PrivateAttr(default_factory=dict)

    def take_shoe(self, channel_id: int, rng: Random) -> tuple[list[Card], int]:
        """Returns `(shoe, generation)` for a new round, removing the shoe from the store.

        Rebuilds a fresh shoe when the channel has none or fewer than
        `RESHUFFLE_THRESHOLD_CARDS` remain. Pass `generation` to `save_shoe`
        on settlement to prevent stale writes from overwriting a newer round's shoe.

        Args:
            channel_id (int): Discord channel ID owning the shoe.
            rng (Random): Random source used when building a fresh shoe.
        """
        generation = self._take_generation.get(channel_id, 0) + 1
        self._take_generation[channel_id] = generation
        existing = self.shoes.pop(channel_id, None)
        if existing is None or len(existing) < RESHUFFLE_THRESHOLD_CARDS:
            return build_shoe(rng=rng), generation
        return existing, generation

    def save_shoe(self, channel_id: int, cards: list[Card], generation: int | None = None) -> None:
        """Stores a copy of `cards` for the next round in `channel_id`.

        Drops the write when `generation` is older than the last saved one for
        `channel_id`; `None` writes unconditionally.

        Args:
            channel_id (int): Discord channel ID owning the shoe.
            cards (list[Card]): Remaining cards after the round settled.
            generation (int | None): Token `take_shoe` issued for the round, or None
                to force an unconditional write.
        """
        if generation is not None:
            if generation < self._saved_generation.get(channel_id, 0):
                return
            self._saved_generation[channel_id] = generation
        self.shoes[channel_id] = list(cards)

    def put_back_shoe(self, channel_id: int, cards: list[Card]) -> None:
        """Restores `cards` for a round that failed before showing a card.

        Leaves any shoe saved since untouched and records no generation, allowing
        an in-flight round to overwrite it.

        Args:
            channel_id (int): Discord channel ID owning the shoe.
            cards (list[Card]): Untouched shoe taken before the failed start.
        """
        self.shoes.setdefault(channel_id, list(cards))

    def true_count(self, channel_id: int) -> float:
        """Returns the Hi-Lo true count the next round in `channel_id` will start from.

        Returns 0.0 when `channel_id` has no stored shoe or is already due for a
        reshuffle.

        Args:
            channel_id (int): Discord channel ID to inspect.
        """
        existing = self.shoes.get(channel_id)
        if existing is None or len(existing) < RESHUFFLE_THRESHOLD_CARDS:
            return 0.0
        return compute_true_count(shoe=existing)
