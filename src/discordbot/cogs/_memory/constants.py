"""Tunable thresholds shared by the per-user memory store, extraction, and pipeline."""

# Raw entries accumulated before consolidation rewrites the main memory file.
# Deliberately eager so observations reach the injected main memory quickly;
# main-file growth is bounded by the compaction pass below, not by batching.
RAW_CONSOLIDATION_THRESHOLD = 2

# Second consolidation trigger: verbose raw extractions consolidate early even
# below the entry-count threshold.
RAW_CONSOLIDATION_MAX_BYTES = 8_192

# Hard cap for the raw file so repeated consolidation failures cannot grow it
# unbounded; the oldest entries are evicted into the archive first.
RAW_FILE_MAX_BYTES = 32_768

# The main file has no hard size clamp. Past the trigger, consolidation runs a
# deep-summarization (compaction) pass aiming at roughly the target size. The
# bound exists because consolidation rewrites the whole file in one response,
# so the ceiling is the model's output-token limit (~65k tokens on Gemini
# Flash), not the 1M-token input window. Compaction summarizes low-signal and
# stale content; it never drops durable memory outright.
MAIN_COMPACTION_TRIGGER_CHARS = 30_000
MAIN_COMPACTION_TARGET_CHARS = 15_000

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
