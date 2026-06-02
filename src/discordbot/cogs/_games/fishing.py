"""Pure fishing rules: rarity rolls, fish selection, sizing, sell value, and EV.

Kept side-effect free so unit tests drive deterministic casts via a fixed `rng`.
The cog wires this up with `random.SystemRandom()` for production. The tuning
catalog lives in `discordbot.typings.fishing`; the EV helpers here let the
simulation script and the tests assert that every loadout is a net 虛擬歡樂豆
sink (negative per-cast expected value).
"""

from random import Random

from discordbot.typings.fishing import (
    RARITY_ORDER,
    BASE_RARITY_BPS,
    BPS_DENOMINATOR,
    SPECIES_BY_RARITY,
    Rarity,
    RodTier,
    BaitType,
    CastOutcome,
    FishSpecies,
)


def _miss_bps(rod: RodTier, bait: BaitType) -> int:
    """Returns the effective 空竿 chance in basis points for a loadout (clamped >= 0)."""
    return max(rod.miss_bps + bait.miss_bps_delta, 0)


def _rarity_weights(rod: RodTier, bait: BaitType) -> dict[Rarity, int]:
    """Returns the per-rarity integer weights for a loadout before normalisation."""
    return {
        rarity: BASE_RARITY_BPS[rarity] * rod.rarity_weight[rarity] * bait.rarity_weight[rarity]
        for rarity in RARITY_ORDER
    }


def roll_rarity(rng: Random, rod: RodTier, bait: BaitType) -> Rarity | None:
    """Rolls a rarity tier for one cast, or None for a 空竿 (no catch).

    The miss chance comes first; on a catch the rarity is drawn from the
    base distribution scaled by the rod and bait weight multipliers.
    """
    if rng.randint(a=0, b=BPS_DENOMINATOR - 1) < _miss_bps(rod=rod, bait=bait):
        return None
    weights = _rarity_weights(rod=rod, bait=bait)
    total = sum(weights.values())
    pick = rng.randint(a=0, b=total - 1)
    cumulative = 0
    for rarity in RARITY_ORDER:
        cumulative += weights[rarity]
        if pick < cumulative:
            return rarity
    return RARITY_ORDER[-1]


def select_species(rng: Random, rarity: Rarity) -> FishSpecies:
    """Picks one species uniformly from the given rarity tier."""
    return rng.choice(seq=SPECIES_BY_RARITY[rarity])


def roll_size(rng: Random, species: FishSpecies) -> int:
    """Rolls a size in millimetres uniformly within the species range."""
    return rng.randint(a=species.min_mm, b=species.max_mm)


def sell_value(species: FishSpecies, size_mm: int) -> int:
    """Returns the 虛擬歡樂豆 sell value for a caught fish of the given size."""
    return species.base_value + (size_mm - species.min_mm) * species.value_per_mm


def cast_fish(rng: Random, rod: RodTier, bait: BaitType) -> CastOutcome:
    """Performs one full cast roll and returns its pure outcome."""
    rarity = roll_rarity(rng=rng, rod=rod, bait=bait)
    if rarity is None:
        return CastOutcome(miss=True)
    species = select_species(rng=rng, rarity=rarity)
    size_mm = roll_size(rng=rng, species=species)
    return CastOutcome(
        miss=False,
        species=species,
        size_mm=size_mm,
        sell_value=sell_value(species=species, size_mm=size_mm),
    )


def expected_value_for_rarity(rarity: Rarity) -> float:
    """Returns the mean sell value of a catch in this rarity tier (uniform size and species)."""
    species_list = SPECIES_BY_RARITY[rarity]
    total = 0.0
    for species in species_list:
        expected_premium = (species.max_mm - species.min_mm) / 2 * species.value_per_mm
        total += species.base_value + expected_premium
    return total / len(species_list)


def loadout_cost(rod: RodTier, bait: BaitType) -> float:
    """Returns the amortised per-cast cost: bait price plus rod price spread over its durability."""
    return bait.cost + rod.cost / rod.durability


def per_cast_ev(rod: RodTier, bait: BaitType) -> float:
    """Returns the analytic net 虛擬歡樂豆 expected value per cast for a loadout.

    Negative for every catalog loadout by design; this is the anti-inflation
    invariant the fishing game enforces. The simulation script confirms the
    empirical mean matches this value.
    """
    catch_prob = (BPS_DENOMINATOR - _miss_bps(rod=rod, bait=bait)) / BPS_DENOMINATOR
    weights = _rarity_weights(rod=rod, bait=bait)
    total = sum(weights.values())
    expected_value_per_catch = sum(
        weights[rarity] / total * expected_value_for_rarity(rarity=rarity)
        for rarity in RARITY_ORDER
    )
    return catch_prob * expected_value_per_catch - loadout_cost(rod=rod, bait=bait)


__all__ = [
    "cast_fish",
    "expected_value_for_rarity",
    "loadout_cost",
    "per_cast_ev",
    "roll_rarity",
    "roll_size",
    "select_species",
    "sell_value",
]
