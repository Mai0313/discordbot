"""Tunable thresholds shared by the per-user memory store, writer, and pipeline."""

# Raw entries accumulated before a consolidation runs. Kept low so stored facts stay
# fresh; still above 1 (together with the consolidation cooldown) so a heavy chatter
# does not fan a consolidation out over every compartment on every single message.
RAW_CONSOLIDATION_THRESHOLD = 2

# Second consolidation trigger: a verbose raw batch consolidates early even
# below the entry-count threshold, bypassing the cooldown as the escape hatch.
RAW_CONSOLIDATION_MAX_BYTES = 16_384

# Hard cap for the raw file so repeated consolidation failures cannot grow it
# unbounded; the oldest entries are evicted into the detail file first.
RAW_FILE_MAX_BYTES = 65_536

# Clamps on one observation's model-authored fields, applied by `_sanitize_observation`
# before the entry is appended. Write-side like the cap above and unlike anything in
# `typings/context_budgets.py`: nothing here bounds what a model request may carry, only
# what a stored observation may hold. The quote is evidence for a summary that is already
# a one-line gist, so it is the tighter of the two.
OBSERVATION_SUMMARY_MAX_CHARS = 800
OBSERVATION_QUOTE_MAX_CHARS = 240

# How long a `recent_context` observation may claim to matter for. The model's `ttl_days`
# is free text, so it is clamped both ways: absent, zero or negative takes the default and
# anything longer than the ceiling is capped. Write-side too — the sweep that acts on the
# stored result reads `RECENT_CONTEXT_TTL_DAYS` below, which is a different question.
OBSERVATION_DEFAULT_TTL_DAYS = 30
OBSERVATION_MAX_TTL_DAYS = 90

# Minimum gap between entry-count-triggered consolidations per user. Not a cost
# guard: it batches the fan-out so the injected facts do not churn on every other
# message, and, recorded at attempt time, it also rate-limits a failing
# consolidation's retries. No data is lost while it waits (raw keeps accumulating, detail.md
# keeps verbatim evidence, and the raw byte trigger above bypasses it for a
# burst), so it stays short enough that new facts reach replies promptly.
MEMORY_CONSOLIDATION_COOLDOWN_SECONDS = 300.0

# Minimum gap between user-requested rebuilds. Recorded at
# attempt time like the consolidation cooldown, and tracked separately so a
# manual regeneration never delays the automatic consolidation or vice versa.
MEMORY_REGENERATION_COOLDOWN_SECONDS = 600.0

# Process-wide cap on concurrent background memory updates. The constraint is
# not cost but proxy contention: unbounded background consolidation
# against the shared LiteLLM proxy would compete with the latency-critical
# reply path for throughput and rate limits. Kept generous because the proxy
# can absorb it; lower it only if background memory work starts adding reply
# latency.
MEMORY_GLOBAL_CONCURRENCY = 24

# Past the trigger (measured on the compartment's own rendered facts), consolidation
# is told to spend the pass compacting it toward the target size. Compaction folds
# overlapping facts together and condenses low-signal ones rather than summarizing the
# set. A well-supported durable fact is merged or tightened, never dropped outright; what it
# drops first is the unsupported, weak, stale and one-off, and fine-grained evidence survives
# in the detail file regardless.
COMPACTION_TRIGGER_CHARS = 30_000
COMPACTION_TARGET_CHARS = 15_000

# Staleness window for mutable (`durability="stable"`) facts, measured RELATIVE to
# the newest mutable activity IN THE SAME COMPARTMENT, not to `today`. The sweep
# drops a mutable fact whose `last_confirmed` is more than this many days behind
# the freshest one, so a busy guild pushes its own stale traits out while a quiet
# compartment with no newer mutable signal ages nothing and forgets nothing.
# Per-compartment anchoring matters: anchoring on the whole scope would let one
# active guild age out the memory of a guild the user simply visits less often.
# Permanent facts and member-alias rows are exempt.
STABLE_FRESHNESS_WINDOW_DAYS = 45

# Lifetime of a `recent` fact, measured against `today`. Was a prompt rule dated by
# the model; now a code sweep, because `last_confirmed` is code-stamped and a
# deterministic date beats a rule the rewrite had to re-apply correctly every pass.
RECENT_CONTEXT_TTL_DAYS = 30

# Bound on the rendered-document cache. One live entry per (scope, reading context)
# is the working set, so this is only here to stop a long-lived process holding keys
# for scopes it will never serve again; the whole cache is dropped when it is hit.
RENDER_CACHE_MAX_ENTRIES = 512

# Net fact loss a single consolidation batch may cause before it is refused, as
# `deletes - creates > max(this, existing // 2)`. Net rather than raw deletes
# because merging four near-duplicates into one is consolidation's primary job and
# the median scope holds only a handful of facts, so a raw-delete cap would reject
# the common case. The regeneration path is exempt: rebuilding from evidence
# legitimately replaces the whole set.
MAX_NET_FACT_DELETIONS_FLOOR = 3

# Store-level backstop for the per-user tone note (tone.md). The note is
# injected on every reply for the message author, so it must stay small;
# shortness is enforced by the consolidation prompt and this clamp only stops a
# misbehaving rewrite from growing the always-read tier unbounded.
TONE_FILE_MAX_BYTES = 4_096

# Hard cap for the cold-tier detail file. Content past the consolidation read
# window (`typings/context_budgets.py::MEMORY_DETAIL_CONTEXT_MAX_CHARS` * 4 bytes;
# it sits there because the read window bounds a request while this bounds a file)
# is unreachable by every consumer, so trimming the oldest once the file outgrows it
# costs nothing functionally and keeps disk bounded. The gap between cap and
# trim target amortizes the O(file) rewrite to roughly once per megabyte of
# new evidence; the cap must stay above the read window so a trim can never
# cut into reachable content, which `tests/test_context_budgets.py` pins.
DETAIL_FILE_MAX_BYTES = 4_194_304
DETAIL_FILE_TRIM_TARGET_BYTES = 3_145_728

# The individual phase-1 and per-compartment calls carry no bound of their own. Nobody is
# waiting on a background memory update, and the OpenAI client bounds every request it makes,
# so the only thing a wrapper here bought was a TIGHTER liveness backstop — and a tighter one
# is worth nothing when the failure it guards against (a genuinely hung provider) ends in a
# retryable no-op either way. What it costs while stuck is one scope lock and one of the
# MEMORY_GLOBAL_CONCURRENCY permits, for the client's own ceiling rather than for 600s.
# What still needs a bound of its own is the whole consolidation fan-out, which is a LOOP over
# compartments rather than one call and so is bounded by nothing upstream:
# `MEMORY_CONSOLIDATE_TIMEOUT_SECONDS` in `typings/timeouts.py`, with the rest of the bot's
# deadlines. It is also what caps a single stuck compartment now that the inner bound is gone.
