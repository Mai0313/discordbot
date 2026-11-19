"""Tunable thresholds for long-term memory.

Bounds pipeline cadence, concurrency, storage sizes, and fact aging. Per-request model input
budgets live in `typings/context_budgets.py`.
"""

# Minimum raw entries to trigger consolidation (above 1 to avoid fanning out every message).
RAW_CONSOLIDATION_THRESHOLD = 2

# Byte threshold that triggers consolidation immediately, bypassing entry count and cooldown.
RAW_CONSOLIDATION_MAX_BYTES = 16_384

# Hard cap on raw.md; oldest entries spill into detail.md when exceeded.
RAW_FILE_MAX_BYTES = 65_536

# Character clamps applied to model-authored observation fields before appending.
OBSERVATION_SUMMARY_MAX_CHARS = 800
OBSERVATION_QUOTE_MAX_CHARS = 240

# Default and upper clamp for recent_context observation ttl_days (distinct from sweep TTL below).
OBSERVATION_DEFAULT_TTL_DAYS = 30
OBSERVATION_MAX_TTL_DAYS = 90

# Minimum gap between entry-count-triggered consolidations per scope (stamped at attempt time).
MEMORY_CONSOLIDATION_COOLDOWN_SECONDS = 300.0

# Minimum gap between manual rebuilds per scope, tracked separately from automatic consolidation.
MEMORY_REGENERATION_COOLDOWN_SECONDS = 600.0

# Minimum gap between `/memory server catchup` runs per server.
MEMORY_CATCHUP_COOLDOWN_SECONDS = MEMORY_REGENERATION_COOLDOWN_SECONDS

# Process-wide cap on concurrent background memory tasks to limit proxy contention with replies.
MEMORY_GLOBAL_CONCURRENCY = 24

# Fact-body char trigger and target for per-compartment compaction; the gap prevents re-triggering
# on the next batch.
COMPACTION_TRIGGER_CHARS = 8_000
COMPACTION_TARGET_CHARS = 5_000

# Days behind a compartment's newest mutable fact before older `stable` facts are swept.
# Permanent facts and member_alias rows are exempt.
STABLE_FRESHNESS_WINDOW_DAYS = 45

# TTL in days from `last_confirmed` for non-permanent, non-stable facts and all `recent` facts.
RECENT_CONTEXT_TTL_DAYS = 30

# Max entries in the rendered-document cache before full eviction.
RENDER_CACHE_MAX_ENTRIES = 512

# Floor and ratio for max net fact loss (`deletes - creates > max(floor, int(existing * ratio))`).
# Forget passes are exempt because they only delete and a refusal would deadlock the scope.
MAX_NET_FACT_DELETIONS_FLOOR = 3

# Kept wide so compactions folding many facts into few are not rejected.
MAX_NET_FACT_DELETIONS_RATIO = 0.9

# Hard byte clamp on the per-user tone.md file injected on every reply.
TONE_FILE_MAX_BYTES = 4_096

# Hard cap and trim target for cold-tier detail.md. Both must stay above
# `MEMORY_DETAIL_CONTEXT_MAX_CHARS * 4` bytes so trimming never cuts into live consolidation input.
DETAIL_FILE_MAX_BYTES = 4_194_304
DETAIL_FILE_TRIM_TARGET_BYTES = 3_145_728

# No per-call timeout here: the OpenAI client bounds individual requests, while the
# multi-compartment fan-out is bounded by `MEMORY_CONSOLIDATE_TIMEOUT_SECONDS`.
