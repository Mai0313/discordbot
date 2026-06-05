"""Tunable thresholds shared by the per-user memory store, extraction, and pipeline."""

# Raw entries accumulated before consolidation rewrites the main memory file.
# Batching plus the consolidation cooldown keep heavy chatters from triggering
# a whole-file rewrite on every other message.
RAW_CONSOLIDATION_THRESHOLD = 4

# Second consolidation trigger: verbose raw extractions consolidate early even
# below the entry-count threshold, bypassing the cooldown as the escape hatch.
RAW_CONSOLIDATION_MAX_BYTES = 16_384

# Hard cap for the raw file so repeated consolidation failures cannot grow it
# unbounded; the oldest entries are evicted into the detail file first.
RAW_FILE_MAX_BYTES = 65_536

# Minimum gap between entry-count-triggered consolidations per user. Recorded
# at attempt time so repeated LLM failures are rate-limited too; the raw byte
# trigger above ignores the cooldown so a burst still consolidates.
MEMORY_CONSOLIDATION_COOLDOWN_SECONDS = 600.0

# Process-wide cap on concurrent background memory updates so a busy server
# cannot fan out unbounded whole-file rewrites against the shared LiteLLM
# proxy and starve the reply path (mirrors codex's stage-1 concurrency limit).
MEMORY_GLOBAL_CONCURRENCY = 8

# The main file has no hard size clamp. Past the trigger, consolidation runs a
# deep-summarization (compaction) pass aiming at roughly the target size. The
# bound exists because consolidation rewrites the whole file in one response,
# so the ceiling is the model's output-token limit (~65k tokens on Gemini
# Flash), not the 1M-token input window. Compaction summarizes low-signal and
# stale content; it never drops durable memory outright, and fine-grained
# evidence survives in the detail file regardless.
MAIN_COMPACTION_TRIGGER_CHARS = 30_000
MAIN_COMPACTION_TARGET_CHARS = 15_000

# Explicit output budget for both memory LLM calls. Far above any legitimate
# memory rewrite (~15k zh-TW chars, roughly 10k tokens) plus reasoning room, but
# below the provider ceiling so a runaway response fails as a detectable
# `incomplete` status instead of silently exhausting the provider limit.
MEMORY_MAX_OUTPUT_TOKENS = 32_768

# Tail window of the detail file fed to consolidation as low-trust provenance.
# Effectively the whole evidence log for any realistic user: this bot injects
# memory exactly once per reply with no on-demand retrieval (unlike codex), so
# main.md must be distilled from the full evidence base in the background. The
# bound only keeps a pathological log inside the consolidation input window
# (~500k zh-TW chars stays well under the 1M-token window with the main file
# and raw batch on top).
MEMORY_DETAIL_CONTEXT_MAX_CHARS = 500_000

# Tail window of the detail file shown by `/memory show detail`; older content
# stays on disk only.
MEMORY_DETAIL_VIEW_MAX_CHARS = 100_000

# Phase-1 transcript truncation (keeps head and tail, drops the middle). Large
# on purpose: the reply history window should reach extraction whole, and the
# memory models accept 1M-token inputs.
MEMORY_TRANSCRIPT_MAX_CHARS = 100_000

# Cap for the bot's own reply inside the transcript. The reply is secondary
# evidence and is appended last, so without this cap a long (e.g. SUMMARY)
# reply fills the entire kept tail and the middle-truncation drops the current
# user message right before it.
MEMORY_REPLY_MAX_CHARS = 8_000

# Background LLM call timeouts. The memory models run with high reasoning
# effort and consolidation rewrites the whole main file, so these are looser
# than interactive paths but still bounded so a stuck call cannot pin the
# in-flight de-dupe slot forever.
MEMORY_EXTRACT_TIMEOUT_SECONDS = 60.0
MEMORY_CONSOLIDATE_TIMEOUT_SECONDS = 180.0
