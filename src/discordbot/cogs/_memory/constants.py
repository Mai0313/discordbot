"""Tunable thresholds shared by the per-user memory store, extraction, and pipeline."""

# Raw entries accumulated before consolidation rewrites the main memory file.
# Kept low so main.md is re-summarized from the full document often and stays
# fresh; still above 1 (together with the consolidation cooldown) so a heavy
# chatter does not trigger a whole-file rewrite on every single message.
RAW_CONSOLIDATION_THRESHOLD = 2

# Second consolidation trigger: verbose raw extractions consolidate early even
# below the entry-count threshold, bypassing the cooldown as the escape hatch.
RAW_CONSOLIDATION_MAX_BYTES = 16_384

# Hard cap for the raw file so repeated consolidation failures cannot grow it
# unbounded; the oldest entries are evicted into the detail file first.
RAW_FILE_MAX_BYTES = 65_536

# Minimum gap between entry-count-triggered consolidations per user. Not a cost
# guard: it batches the lossy whole-file rewrite so main.md (the only tier
# injected into replies) does not drift from rewriting on every other message,
# and, recorded at attempt time, it also rate-limits a failing consolidation's
# retries. No data is lost while it waits (raw keeps accumulating, detail.md
# keeps verbatim evidence, and the raw byte trigger above bypasses it for a
# burst), so it stays short enough that new facts reach replies promptly.
MEMORY_CONSOLIDATION_COOLDOWN_SECONDS = 300.0

# Minimum gap between user-requested main-file regenerations. Recorded at
# attempt time like the consolidation cooldown, and tracked separately so a
# manual regeneration never delays the automatic consolidation or vice versa.
MEMORY_REGENERATION_COOLDOWN_SECONDS = 600.0

# Process-wide cap on concurrent background memory updates. The constraint is
# not cost but proxy contention: unbounded fan-out of whole-file rewrites
# against the shared LiteLLM proxy would compete with the latency-critical
# reply path for throughput and rate limits. Kept generous because the proxy
# can absorb it; lower it only if background memory work starts adding reply
# latency.
MEMORY_GLOBAL_CONCURRENCY = 24

# The main file has no hard size clamp. Past the trigger, consolidation runs a
# deep-summarization (compaction) pass aiming at roughly the target size. The
# bound exists because consolidation rewrites the whole file in one response, so
# the ceiling is the answer model's own output-token limit (~64k tokens on
# Gemini Pro), not the 1M-token input window. The memory calls no longer set a
# lower explicit cap, so the trigger sits well inside that ceiling. Compaction
# summarizes low-signal and stale content; it never drops durable memory
# outright, and fine-grained evidence survives in the detail file regardless.
MAIN_COMPACTION_TRIGGER_CHARS = 30_000
MAIN_COMPACTION_TARGET_CHARS = 15_000

# Staleness window for mutable (dated `[~YYYY-MM]`) bullets in main.md's stable
# sections, measured RELATIVE to the newest mutable activity in the file, not to
# `today`. Consolidation drops a mutable bullet whose last-confirmed month is
# more than this many days behind the freshest mutable bullet, so a busy channel
# pushes stale traits out while a quiet stretch with no newer mutable signal ages
# nothing and forgets nothing. Permanent identity facts and enforced standing
# directives live in the undated `## 永久事實` section and are exempt. Read only
# by the consolidation prompt (PHASE2_PROMPT / SERVER_PHASE2_PROMPT).
STABLE_FRESHNESS_WINDOW_DAYS = 45

# Tail window of the detail file fed to consolidation as low-trust provenance.
# Effectively the whole evidence log for any realistic user: this bot injects
# memory exactly once per reply with no on-demand retrieval (unlike codex), so
# main.md must be distilled from the full evidence base in the background. The
# bound only keeps a pathological log inside the consolidation input window
# (~500k zh-TW chars stays well under the 1M-token window with the main file
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
# The memory models run at high reasoning effort on a Pro tier and
# consolidation rewrites the whole main file, so the bound stays well above a
# legitimately slow rewrite (minutes) and only fires on a truly hung call.
MEMORY_EXTRACT_TIMEOUT_SECONDS = 600.0
MEMORY_CONSOLIDATE_TIMEOUT_SECONDS = 600.0
