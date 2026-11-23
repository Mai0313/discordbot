"""Per-request content caps that reduce what a model sees without failing the call.

A constant belongs here when its binding leaves the model seeing less content while the
request still goes out (fewer messages, attachments, or linked posts; shorter memory or
transcript). Excluded by rule:

- Write-side disk clamps (`TONE_FILE_MAX_BYTES`): stay beside the write path.
- Prompt rewrite targets/triggers (`COMPACTION_TARGET_CHARS`, `COMPACTION_TRIGGER_CHARS`).
- Reply/Discord output caps (`MAX_INLINE_IMAGES`, `DISCORD_MESSAGE_LIMIT`,
  `DISCORD_ATTACHMENT_LIMIT`).
- Provider hard limits (`FILES_API_MAX_BYTES`), UI caps (`REASONING_PREVIEW_MAX_CHARS`,
  `MEMORY_PAGE_MAX_CHARS`), concurrency, cache sizes, and deadline-derived fetch gates
  (`MAX_BILIBILI_INGEST_DURATION_SECONDS`).
- Wall-clock deadlines that fail or degrade a call (`typings/timeouts.py`).
"""

from typing import Final

# --------------------------------------------------------------------------------------
# Channel history in one reply (whichever axis binds first wins)
# --------------------------------------------------------------------------------------

# Max channel history messages per reply; `HISTORY_CHAR_BUDGET` usually binds first.
HISTORY_MESSAGE_LIMIT: Final[int] = 50

# Max rendered history chars per reply; the bot's most recent reply is kept whole past it.
HISTORY_CHAR_BUDGET: Final[int] = 4000

# Per-message char floor covering author headers and attachment-only markers.
HISTORY_PER_MESSAGE_OVERHEAD: Final[int] = 40

# Chars kept from each earlier bot reply (the most recent bot reply stays whole).
HISTORY_EARLIER_REPLY_MAX_CHARS: Final[int] = 300

# Max history attachments uploaded as media parts; older ones degrade to text markers.
MAX_HISTORY_MEDIA_PARTS: Final[int] = 5

# --------------------------------------------------------------------------------------
# Inlined attachment bytes (`InlineRenderer`; Gemini rejects base64 payloads ~77-97 MB)
# --------------------------------------------------------------------------------------

# Max raw size for a single inlined attachment (~28 MB base64; fits even if rendered
# twice in history and reference, and degrades to a marker past `MAX_HISTORY_INLINE_BYTES`).
MAX_INLINE_ATTACHMENT_BYTES: Final[int] = 20 * 1024 * 1024

# Max total base64/inlined bytes across history attachments (newest first); older ones
# degrade to markers. Current and replied-to messages render whole on top of this cap.
MAX_HISTORY_INLINE_BYTES: Final[int] = 24_000_000

# --------------------------------------------------------------------------------------
# Memory read into a request
# --------------------------------------------------------------------------------------

# Target user-memory count per reply; deterministic participants are never displaced and
# optional route picks only fill remaining slots.
MEMORY_CONTEXT_TARGET_USERS: Final[int] = 8

# Cap on one merged rendered memory document (body text only, not frontmatter); truncates
# to newest facts with a notice without deleting anything on disk.
MEMORY_INJECTION_MAX_CHARS: Final[int] = 30_000

# Post-write warning threshold for `MEMORY_INJECTION_MAX_CHARS`; does not truncate requests.
MEMORY_INJECTION_WARN_CHARS: Final[int] = 24_000

# --------------------------------------------------------------------------------------
# Memory note review and consolidation requests
# --------------------------------------------------------------------------------------

# Middle-truncation cap on transcripts sent to memory note review (keeps head and tail).
MEMORY_TRANSCRIPT_MAX_CHARS: Final[int] = 100_000

# Per-note char cap on inline memory notes (batch count is bounded by `MAX_MEMORY_NOTES`).
MEMORY_NOTE_MAX_CHARS: Final[int] = 400

# Max notes of one kind kept when merging skipped turns into one queued memory payload.
MEMORY_MERGED_NOTES_MAX: Final[int] = 10

# Cap on the bot's reply inside memory transcripts so it cannot crowd out the user turn.
MEMORY_REPLY_MAX_CHARS: Final[int] = 8_000

# Tail window of `detail.md` fed to consolidation; `DETAIL_FILE_MAX_BYTES` and its trim
# target must stay above `MEMORY_DETAIL_CONTEXT_MAX_CHARS * 4`.
MEMORY_DETAIL_CONTEXT_MAX_CHARS: Final[int] = 500_000

# Smaller `detail.md` tail used after a compartment's consolidation call fails (prompts
# >=60k tokens frequently exhaust output budget on reasoning); next success restores full.
MEMORY_DETAIL_RETRY_MAX_CHARS: Final[int] = 30_000

# --------------------------------------------------------------------------------------
# Linked posts read for one reply
# --------------------------------------------------------------------------------------

# Max media parts uploaded across a Threads block (linked post and quoted post combined).
MAX_THREADS_MEDIA_PARTS: Final[int] = 10

# Max posts rendered from a Threads reply chain (keeps the target and nearest ancestors).
MAX_THREADS_POSTS: Final[int] = 6

# Max comments rendered below a Threads target post across all branches.
MAX_THREADS_REPLIES: Final[int] = 30

# Max images uploaded from a Douyin photo post (independent of the expansion cog's cap).
MAX_DOUYIN_INGEST_IMAGES: Final[int] = 8

# Max images uploaded from a Facebook post (independent of `parse_facebook`'s display cap).
MAX_FACEBOOK_INGEST_IMAGES: Final[int] = 8

# Backstop cap on preloaded comments injected below a Facebook post.
MAX_FACEBOOK_COMMENTS: Final[int] = 10

# Max images uploaded from an Instagram carousel.
MAX_INSTAGRAM_INGEST_IMAGES: Final[int] = 8

# Max images uploaded from a Twitter post; Twitter videos stay as links and are not ingested.
MAX_TWITTER_INGEST_IMAGES: Final[int] = 4

# Max posts rendered from a Twitter thread, keeping the linked post and nearest ancestors.
MAX_TWITTER_POSTS: Final[int] = MAX_THREADS_POSTS

# Max replies rendered below a Twitter post (keeps most-liked when FxTwitter returns more).
MAX_TWITTER_REPLIES: Final[int] = MAX_THREADS_REPLIES

# Per-reply char cap below a Twitter post; truncated replies state how much was omitted.
MAX_TWITTER_REPLY_CHARS: Final[int] = 2000

# Max comments injected below an Instagram post (page carries the full comment list).
MAX_INSTAGRAM_COMMENTS: Final[int] = 20

# Head chars kept from a Bilibili video description.
MAX_BILIBILI_DESCRIPTION_CHARS: Final[int] = 1000

# --------------------------------------------------------------------------------------
# Requests that are not the reply
# --------------------------------------------------------------------------------------

# Max subject reference images shared by omni video renders and their preceding prompt refine.
MAX_VIDEO_REFERENCE_IMAGES: Final[int] = 3

# Max channel history messages fetched for `PromptGenerator.refine` on IMAGE and VIDEO routes.
MEDIA_REFINE_HISTORY_LIMIT: Final[int] = 6

# Max total characters of recent text-only conversation context passed to `PromptGenerator.refine`.
MEDIA_REFINE_CONTEXT_MAX_CHARS: Final[int] = 3000
