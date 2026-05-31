"""In-memory per-channel persistent Blackjack shoe for cross-round card counting.

The shoe carries over between rounds in the same Discord channel so the bot's
Hi-Lo count has signal and the EV engine reasons over the real depleted shoe.
State is in-memory only: a bot restart drops every channel's shoe, which is an
acceptable natural reshuffle.
"""

from random import Random
from typing import Final

from pydantic import Field, BaseModel, ConfigDict, PrivateAttr

from discordbot.typings.games import Card
from discordbot.cogs._games.blackjack import build_shoe
from discordbot.cogs._games.blackjack_ev import compute_true_count

# Reshuffle a round before it starts once fewer than this many cards remain. It must
# exceed the worst-case cards a single round can deal so the shoe never empties
# mid-round into the infinite `draw_card` fallback (which would corrupt the count):
# 6 seats x 2 split hands x 5 cards (過五關 auto-stand) + a deep H17 dealer is < 96.
# That leaves ~54% penetration of the 208-card 4-deck shoe, deep enough to count.
RESHUFFLE_THRESHOLD_CARDS: Final[int] = 96


class BlackjackShoeStore(BaseModel):
    """Holds one persistent shoe per channel so the bot's card counting has signal.

    Shoes are keyed by Discord channel id. The mutating methods are synchronous and
    never await, so they are atomic under the single-threaded event loop; two
    concurrent games in one channel degrade gracefully to a fresh shoe rather than
    interleaving draws on a shared list, and the per-round generation token stops an
    earlier-started round from clobbering a newer table's shoe when they settle out of
    order.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    shoes: dict[int, list[Card]] = Field(default_factory=dict)
    # Per-channel monotonic generation counters. `take_shoe` stamps each round it hands
    # out with an increasing generation; `save_shoe` drops a write whose generation is
    # older than the last persisted one, so when two tables in the same channel settle out
    # of order the earlier-started round cannot overwrite the newer table's shoe.
    _take_generation: dict[int, int] = PrivateAttr(default_factory=dict)
    _saved_generation: dict[int, int] = PrivateAttr(default_factory=dict)

    def take_shoe(self, *, channel_id: int, rng: Random) -> tuple[list[Card], bool, int]:
        """Returns `(shoe, reshuffled, generation)` for a new round, removing it from the store.

        Rebuilds a fresh shoe when the channel has none or penetration crossed the
        reshuffle threshold. The round deals from this shoe and the caller persists
        depletion by saving the round's remaining shoe with `save_shoe` once it
        settles (the round may deal from a copy, so the returned list itself is not
        relied on to mutate). The `reshuffled` flag is True only for a genuine
        penetration cut, not for the first shoe in a channel, so the table only
        announces a real reshuffle. The `generation` stamps this round; pass it back to
        `save_shoe` so an older in-flight round cannot overwrite a newer table's shoe.
        """
        generation = self._take_generation.get(channel_id, 0) + 1
        self._take_generation[channel_id] = generation
        existing = self.shoes.pop(channel_id, None)
        if existing is None:
            return build_shoe(rng=rng), False, generation
        if len(existing) < RESHUFFLE_THRESHOLD_CARDS:
            return build_shoe(rng=rng), True, generation
        return existing, False, generation

    def save_shoe(
        self, *, channel_id: int, cards: list[Card], generation: int | None = None
    ) -> None:
        """Stores the cards remaining after a round for the next one in that channel.

        Copies the list so the stored shoe is decoupled from the live round object.
        `generation` is the token `take_shoe` issued for this round; a save whose
        generation is older than the last persisted one is dropped so an earlier-started
        overlapping round cannot clobber a newer table's shoe. `None` forces an
        unconditional write for direct seeding.
        """
        if generation is not None:
            if generation < self._saved_generation.get(channel_id, 0):
                return
            self._saved_generation[channel_id] = generation
        self.shoes[channel_id] = list(cards)

    def true_count(self, *, channel_id: int) -> float:
        """Returns the Hi-Lo true count the next round in this channel will start from.

        A channel with no stored shoe, or one already due for a reshuffle, is neutral
        (0.0) because the upcoming round deals from a fresh shoe.
        """
        existing = self.shoes.get(channel_id)
        if existing is None or len(existing) < RESHUFFLE_THRESHOLD_CARDS:
            return 0.0
        return compute_true_count(shoe=existing)
