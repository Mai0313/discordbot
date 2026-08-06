"""Tunable thresholds shared by the memory store, extraction and pipeline.

Every number deciding WHEN the two-phase memory pipeline acts and HOW MUCH it may read, keep or
inject lives here instead of beside the code that reads it, so the whole tuning surface can be
reviewed at once. They are plain module constants and deliberately NOT environment-backed — the
memory service's only env switch is `MEMORY_GIT_ENABLED` in `typings/memory.py` — because most
were set against a measurement of the live store rather than a preference, so retuning one is a
code change that goes through review and the tests. Each carries its own reasoning in the comment
above it; what follows is only how they fit together.

Who reads what: `store.py` owns the file-tier caps (raw, detail, tone, the render cache, and the
injection ceiling it renders down to), `pipeline.py` the consolidation triggers, cooldowns,
process-wide concurrency and the fan-out timeout, `deltas.py` the aging windows and the
net-deletion floor, `extraction.py` the transcript caps and the per-call timeouts, and
`prompts.py` renders `COMPACTION_TARGET_CHARS` into the compaction block the model is told to aim
at. Consumers import the names by value, so a test overrides one on the module that reads it
(`...memory.store.RAW_FILE_MAX_BYTES`, `...memory.pipeline.RAW_CONSOLIDATION_THRESHOLD`), never
here; `MEMORY_GLOBAL_CONCURRENCY` is the exception, read through a provider lambda when the
semaphore is built.

Four relationships have to survive a retune, and no test pins them against each other:

* `RAW_CONSOLIDATION_MAX_BYTES` stays well under `RAW_FILE_MAX_BYTES`, or a verbose burst is
  evicted into the detail file before it ever fires the consolidation it was meant to.
* `DETAIL_FILE_TRIM_TARGET_BYTES` stays above the consolidation read window
  (`MEMORY_DETAIL_CONTEXT_MAX_CHARS * 4` bytes, UTF-8's worst case), or a trim cuts into evidence
  the next consolidation can still reach.
* `MEMORY_INJECTION_WARN_CHARS` stays below `MEMORY_INJECTION_MAX_CHARS`, since a warning that
  only fires once the cap is already dropping facts on read arrives too late to act on.
* `MEMORY_COMPARTMENT_TIMEOUT_SECONDS` stays at or below `MEMORY_CONSOLIDATE_TIMEOUT_SECONDS`,
  which bounds the whole fan-out those per-compartment calls run inside.
"""

# Raw entries accumulated before a consolidation runs. Kept low so stored facts stay
# fresh; still above 1 (together with the consolidation cooldown) so a heavy chatter
# does not fan a consolidation out over every compartment on every single message.
RAW_CONSOLIDATION_THRESHOLD = 2

# Second consolidation trigger: verbose raw extractions consolidate early even
# below the entry-count threshold, bypassing the cooldown as the escape hatch.
RAW_CONSOLIDATION_MAX_BYTES = 16_384

# Hard cap for the raw file so repeated consolidation failures cannot grow it
# unbounded; the oldest entries are evicted into the detail file first.
RAW_FILE_MAX_BYTES = 65_536

# Minimum gap between entry-count-triggered consolidations per scope (a server scope
# consolidates through the same path as a user's). Not a cost guard: it batches the
# fan-out so the injected facts do not churn on every other message, and, recorded at
# attempt time, it also rate-limits a failing consolidation's retries. No data is lost
# while it waits (raw keeps accumulating, detail.md keeps verbatim evidence, and the raw
# byte trigger above bypasses it for a burst), so it stays short enough that new facts
# reach replies promptly.
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

# Measured per compartment against its rendered stored facts: past the trigger,
# consolidation is told to run a deep-summarization (compaction) pass over the
# compartment it is rewriting, aiming at roughly the target size.
# Compaction merges low-signal and stale facts; it never drops durable memory
# outright, and fine-grained evidence survives in the detail file regardless.
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

# Lifetime of a fact filed in the `recent` SECTION, measured against `today`. Keyed on
# the section rather than on the `recent` step of the durability ladder, which the sweep
# never reads for this rule. Was a prompt rule dated by the model; now a code sweep,
# because `last_confirmed` is code-stamped and a deterministic date beats a rule the
# rewrite had to re-apply correctly every pass.
RECENT_CONTEXT_TTL_DAYS = 30

# Ceiling on one rendered memory document (the merged compartments injected for one
# reply). Measured against the live store, today's documents run to a median of
# ~800 bytes and a maximum of 25 KB, so this is a backstop that fires on nobody: it
# exists so a runaway scope degrades to its newest facts plus an explicit notice
# instead of silently bloating every request. The warn threshold is the operator's
# signal that the prompt-side budget stopped working. Rendering stops at the cap;
# nothing is deleted, so the cap can never fight the next consolidation over
# content it would immediately write back.
MEMORY_INJECTION_MAX_CHARS = 30_000
MEMORY_INJECTION_WARN_CHARS = 24_000

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

# Tail window of the detail file fed to consolidation as low-trust provenance.
# Effectively the whole evidence log for any realistic user: this bot injects
# memory exactly once per reply with no on-demand retrieval (unlike codex), so the
# stored facts must be distilled from the full evidence base in the background. The
# bound only keeps a pathological log inside the consolidation input window
# (~500k zh-TW chars stays well under the 1M-token window with the stored facts
# and raw batch on top).
MEMORY_DETAIL_CONTEXT_MAX_CHARS = 500_000

# Hard cap for the cold-tier detail file. Content past the consolidation read
# window (MEMORY_DETAIL_CONTEXT_MAX_CHARS * 4 bytes) is unreachable by every
# consumer, so trimming the oldest entries once the file outgrows the cap
# costs nothing functionally and keeps disk bounded. The gap between cap and
# trim target amortizes the O(file) rewrite to roughly once per megabyte of
# new evidence; the cap must stay above the read window so a trim can never
# cut into reachable content.
DETAIL_FILE_MAX_BYTES = 4_194_304
DETAIL_FILE_TRIM_TARGET_BYTES = 3_145_728

# Phase-1 transcript truncation (keeps head and tail, drops the middle). Large
# on purpose: the reply history window should reach extraction whole, and the
# memory models accept 1M-token inputs.
MEMORY_TRANSCRIPT_MAX_CHARS = 100_000

# Cap for the bot's own reply inside the transcript. The reply is secondary
# evidence and is appended last, so without this cap a long (e.g. SUMMARY)
# reply fills the entire kept tail and the middle-truncation drops the current
# user message right before it.
MEMORY_REPLY_MAX_CHARS = 8_000

# Background LLM call timeouts, kept purely as a liveness backstop rather than
# a latency or cost guard (a slow background update is harmless). A genuinely
# stuck call would otherwise hold the scope's lock and a global-concurrency
# permit forever, so that user/server would never get another memory update.
# The memory models run at high reasoning effort on a Pro tier, so the bound stays
# well above a legitimately slow rewrite (minutes) and only fires on a truly hung
# call. Consolidation now fans out over a scope's compartments, so the two bounds
# nest: each compartment call gets the shorter one and the whole fan-out is wrapped
# in the longer one, keeping the worst-case lock hold at what it is today rather
# than multiplying it by the number of compartments.
MEMORY_EXTRACT_TIMEOUT_SECONDS = 600.0
MEMORY_CONSOLIDATE_TIMEOUT_SECONDS = 600.0
MEMORY_COMPARTMENT_TIMEOUT_SECONDS = 300.0
