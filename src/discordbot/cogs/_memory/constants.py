"""Tunable thresholds shared by the per-user memory store, extraction, and pipeline."""

# Raw entries accumulated before consolidation rewrites the main memory file.
# Low values burn a consolidation LLM call too often; high values keep the
# injected main memory stale for longer.
RAW_CONSOLIDATION_THRESHOLD = 5

# Second consolidation trigger: verbose raw extractions consolidate early even
# below the entry-count threshold.
RAW_CONSOLIDATION_MAX_BYTES = 8_192

# Hard cap for the raw file so repeated consolidation failures cannot grow it
# unbounded; the oldest entries are evicted first.
RAW_FILE_MAX_BYTES = 32_768

# Upper bound for the consolidated main memory file. Keeps the prompt
# injection budget bounded and fits inside one embed description (4,096 chars)
# for `/memory show` with headroom for wrapper text.
MAIN_FILE_MAX_CHARS = 3_500

# Read-side safety truncation applied when injecting the main memory into the
# reply instructions; matches the writer-side cap so it only trips on
# hand-edited or legacy files.
MEMORY_INJECTION_MAX_CHARS = 3_500

# Phase-1 transcript truncation (keeps head and tail, drops the middle).
MEMORY_TRANSCRIPT_MAX_CHARS = 12_000

# Cap for the bot's own reply inside the transcript. The reply is secondary
# evidence and is appended last, so without this cap a long (e.g. SUMMARY)
# reply fills the entire kept tail and the middle-truncation drops the current
# user message right before it.
MEMORY_REPLY_MAX_CHARS = 2_000

# Background LLM call timeouts. `memories_model` runs with high reasoning
# effort, so these are looser than interactive paths but still bounded so a
# stuck call cannot pin the in-flight de-dupe slot forever.
MEMORY_EXTRACT_TIMEOUT_SECONDS = 30.0
MEMORY_CONSOLIDATE_TIMEOUT_SECONDS = 45.0
