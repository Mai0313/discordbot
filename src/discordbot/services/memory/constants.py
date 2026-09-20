"""Tunable thresholds shared by the per-user memory store, writer, and pipeline.

What lives here bounds the memory pipeline's own work — how often a consolidation runs, how
much of it the process does at once, how much a stored observation may hold, how large a file
or a cache may grow. A bound on what a model REQUEST may carry is a different question and
lives with the other request budgets.
"""

# Raw entries accumulated before a consolidation runs. Kept low so stored facts stay fresh; still
# above 1, together with the cooldown, so a heavy chatter does not fan a consolidation out over
# every compartment on every single message.
RAW_CONSOLIDATION_THRESHOLD = 2

# Second trigger: a verbose raw batch consolidates early even below the entry count, and bypasses
# the cooldown as the escape hatch.
RAW_CONSOLIDATION_MAX_BYTES = 16_384

# Hard cap for the raw file so repeated consolidation failures cannot grow it unbounded; the
# oldest entries are evicted into the detail file first.
RAW_FILE_MAX_BYTES = 65_536

# Clamps on one observation's model-authored fields, applied before the entry is appended. The
# quote is evidence for a summary that is already a one-line gist, so it is the tighter of the two.
OBSERVATION_SUMMARY_MAX_CHARS = 800
OBSERVATION_QUOTE_MAX_CHARS = 240

# How long a `recent_context` observation may claim to matter for. The model's `ttl_days` is free
# text, so it is clamped both ways: absent, zero or negative takes the default, anything longer is
# capped. Distinct from the sweep's own window further down, which acts on the stored result.
OBSERVATION_DEFAULT_TTL_DAYS = 30
OBSERVATION_MAX_TTL_DAYS = 90

# Minimum gap between entry-count-triggered consolidations per scope. Not a cost guard: it batches
# the fan-out so the injected facts do not churn on every other message, and, recorded at attempt
# time, it also rate-limits a failing consolidation's retries. Nothing is lost while it waits —
# raw keeps accumulating, the detail file keeps the evidence, and the byte trigger above bypasses
# it for a burst — so it stays short enough that new facts reach replies promptly.
MEMORY_CONSOLIDATION_COOLDOWN_SECONDS = 300.0

# Minimum gap between user-requested rebuilds, tracked separately so a manual regeneration never
# delays the automatic consolidation or the other way round.
MEMORY_REGENERATION_COOLDOWN_SECONDS = 600.0

# Process-wide cap on concurrent background memory updates. The constraint is proxy contention
# rather than cost: unbounded background consolidation would compete with the latency-critical
# reply path for throughput and rate limits. Lower it only if background memory work starts adding
# reply latency.
MEMORY_GLOBAL_CONCURRENCY = 24

# Past the trigger, measured on the compartment's own rendered facts, consolidation is told to
# spend the pass compacting toward the target. Compaction folds overlapping facts together and
# condenses low-signal ones rather than summarizing the set: a well-supported durable fact is
# merged or tightened, never dropped outright, and what goes first is the unsupported, weak, stale
# and one-off. Fine-grained evidence survives in the detail file regardless.
COMPACTION_TRIGGER_CHARS = 30_000
COMPACTION_TARGET_CHARS = 15_000

# Staleness window for mutable (`durability="stable"`) facts, measured RELATIVE to the newest
# mutable activity IN THE SAME COMPARTMENT rather than to `today`. The sweep drops a mutable fact
# whose `last_confirmed` is more than this far behind the freshest one, so a busy guild pushes its
# own stale traits out while a quiet compartment ages nothing. Anchoring on the whole scope instead
# would let one active guild age out the memory of a guild the user simply visits less often.
# Permanent facts and member-alias rows are exempt.
STABLE_FRESHNESS_WINDOW_DAYS = 45

# Lifetime of a `recent` fact, measured against `today`. A code sweep rather than a prompt rule,
# because `last_confirmed` is code-stamped and a deterministic date beats a rule a rewrite has to
# re-apply correctly every pass.
RECENT_CONTEXT_TTL_DAYS = 30

# Bound on the rendered-document cache. One live entry per (scope, reading context) is the working
# set, so this only stops a long-lived process holding keys for scopes it will never serve again;
# the whole cache is dropped when it is hit.
RENDER_CACHE_MAX_ENTRIES = 512

# Net fact loss one consolidation batch may cause before it is refused, as
# `deletes - creates > max(this, existing // 2)`. Net rather than raw, because merging four
# near-duplicates into one is consolidation's primary job and the median scope holds only a
# handful of facts, so a raw-delete cap would reject the common case. The regeneration path is
# exempt: rebuilding from evidence legitimately replaces the whole set.
MAX_NET_FACT_DELETIONS_FLOOR = 3

# Store-level backstop for the per-user tone note. The note is injected on every reply for the
# message author, so it must stay small; shortness is asked for in the prompt and this clamp only
# stops a misbehaving rewrite growing the always-read tier unbounded.
TONE_FILE_MAX_BYTES = 4_096

# Hard cap for the cold-tier detail file, and the size a trim brings it back to. Content past the
# window consolidation reads is unreachable by every consumer, so trimming the oldest costs
# nothing functionally and keeps disk bounded; the gap between the two amortizes the O(file)
# rewrite to roughly once per megabyte of new evidence. The cap MUST stay above that window —
# `MEMORY_DETAIL_CONTEXT_MAX_CHARS` at 4 bytes a character — or a trim cuts into content
# something still reads, which `tests/test_context_budgets.py` holds it to.
DETAIL_FILE_MAX_BYTES = 4_194_304
DETAIL_FILE_TRIM_TARGET_BYTES = 3_145_728

# There is deliberately no per-call bound here. Nobody waits on a background memory update and
# the client bounds every request it makes, so a tighter wrapper would only restate that ceiling
# — and a tighter one buys nothing when the failure it guards against ends in a retryable no-op
# either way. What does need its own bound is the consolidation fan-out, which is a LOOP over
# compartments and so is bounded by nothing upstream; that one is
# `MEMORY_CONSOLIDATE_TIMEOUT_SECONDS`, with the bot's other deadlines, and it is also what caps
# a single stuck compartment.
