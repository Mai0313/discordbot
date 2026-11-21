"""Wall-clock bounds whose expiry causes a failure or degradation.

Every failure- or degradation-producing deadline lives here, along with retry counts that
multiply a bound. Excluded by rule:

- Animation pacing, cache TTLs, retry cadences, `@tasks.loop` intervals, view idle expiry,
  and `MEDIA_HOSTING_RETENTION_HOURS` (not failures): stay beside their callers.
- Deep research: intentionally unbounded at the job level; the agent settles server-side.
- Pure liveness wrappers around single `AsyncOpenAI` calls: `openai` already defaults to
  600s read/write/pool with `max_retries=2`. Direct `google-genai` calls default to
  `timeout=None` and `xai-sdk` client-streaming uploads bypass its unary timeout
  interceptors, so those paths require explicit bounds here.
"""

from typing import Final

# --------------------------------------------------------------------------------------
# Reply pipeline (`gen_reply`)
# --------------------------------------------------------------------------------------

# Max streaming answer attempts per reply; multiplies the `openai` client's own timeout.
ANSWER_STREAM_MAX_ATTEMPTS: Final[int] = 3

# Overall grace for intent-selected linked-post context (read + media fetch/upload);
# builders degrade to text before this fires.
LINK_CONTEXT_GRACE_SECONDS: Final[float] = 180.0

# Reserve kept before `LINK_CONTEXT_GRACE_SECONDS` so media stops cleanly (join, scratch
# cleanup, text render) instead of cancelling the whole link block.
LINK_MEDIA_DEGRADE_MARGIN_SECONDS: Final[float] = 10.0

# Max budget for linked-post media fetch + upload (derived so media degrades before the
# outer grace expires).
LINK_MEDIA_TIMEOUT_SECONDS: Final[float] = (
    LINK_CONTEXT_GRACE_SECONDS - LINK_MEDIA_DEGRADE_MARGIN_SECONDS
)

# Grace for a cancelled yt-dlp download thread to notice its stop flag before scratch cleanup.
DOWNLOAD_STOP_JOIN_SECONDS: Final[float] = 5.0

# Wait for a Discord attachment's Files API upload to reach `ACTIVE`; past it, cached as
# `PendingUpload` for re-polling on the next reference.
ATTACHMENT_ACTIVATION_TIMEOUT_SECONDS: Final[float] = 15.0

# Transfer bound for uploading a Discord attachment to Gemini Files API; expiry drops it.
ATTACHMENT_UPLOAD_TIMEOUT_SECONDS: Final[float] = 120.0

# Per-call bound on `files.get` during attachment activation and pending re-polls.
FILES_API_READ_TIMEOUT_SECONDS: Final[float] = 10.0

# Whole-transfer bound for xAI `Files.UploadFile` (client-streaming bypasses `xai-sdk`'s
# unary timeout interceptors); expiry drops the attachment.
GROK_FILE_UPLOAD_TIMEOUT_SECONDS: Final[float] = 30.0

# Combined transfer + `ACTIVE` poll bound for uploading a generated clip for persona reply.
GENERATED_VIDEO_ACTIVATION_TIMEOUT_SECONDS: Final[float] = 60.0

# --------------------------------------------------------------------------------------
# Generated media (VIDEO edit worst case: render + 2 * FILES_READY_TIMEOUT_SECONDS)
# --------------------------------------------------------------------------------------

# Bound for omni source-video upload/activation and generated-clip URI download; expiry
# fails the video render.
FILES_READY_TIMEOUT_SECONDS: Final[float] = 180.0

# Hard bound on direct-to-Google omni `interactions.create` (`google-genai` has no default).
VIDEO_RENDER_TIMEOUT_SECONDS: Final[float] = 600.0

# Bound on best-effort direct-to-Google Lyria inline music generation.
MUSIC_RENDER_TIMEOUT_SECONDS: Final[float] = 300.0

# Bound on best-effort inline image generation after the text reply is posted.
INLINE_IMAGE_TIMEOUT_SECONDS: Final[float] = 300.0

# Whole-call bound (including `openai` retries) on best-effort TTS synthesis after the
# text reply is posted.
VOICE_TIMEOUT_SECONDS: Final[float] = 300.0

# Bound on IMAGE/VIDEO prompt refinement; expiry falls back to the raw user prompt.
PROMPT_REFINE_TIMEOUT_SECONDS: Final[float] = 120.0

# --------------------------------------------------------------------------------------
# Product deadlines on LLM side calls
# --------------------------------------------------------------------------------------

# Bound on research-thread title generation; expiry falls back to the brief's first line.
THREAD_TITLE_TIMEOUT_SECONDS: Final[float] = 15.0

# --------------------------------------------------------------------------------------
# Background LLM work
# --------------------------------------------------------------------------------------

# Whole-fan-out liveness bound across all compartments in one consolidation pass; expiry
# releases the scope lock and global concurrency permit.
MEMORY_CONSOLIDATE_TIMEOUT_SECONDS: Final[float] = 1800.0

# --------------------------------------------------------------------------------------
# Link expansion and downloads
# --------------------------------------------------------------------------------------

# Bound on one Douyin auto-expansion, capping how long it holds the shared fetch slot.
DOUYIN_EXPAND_TIMEOUT_SECONDS: Final[float] = 120.0

# Bound on one Threads auto-expansion (full conversation walk + target video download).
THREADS_EXPAND_TIMEOUT_SECONDS: Final[float] = LINK_CONTEXT_GRACE_SECONDS

# Bound on one Facebook auto-expansion (single page fetch and JSON parse).
FACEBOOK_EXPAND_TIMEOUT_SECONDS: Final[float] = 60.0

# Bound on one Instagram auto-expansion (single page fetch and comment parse).
INSTAGRAM_EXPAND_TIMEOUT_SECONDS: Final[float] = FACEBOOK_EXPAND_TIMEOUT_SECONDS

# Bound on one Twitter auto-expansion (matches Facebook/Instagram page-read expansions).
TWITTER_EXPAND_TIMEOUT_SECONDS: Final[float] = FACEBOOK_EXPAND_TIMEOUT_SECONDS

# Per-image download bound for expansion cards; expiry leaves the image linked instead.
CARD_IMAGE_FETCH_TIMEOUT_SECONDS: Final[float] = 10.0

# Whole-download bound for an expansion post's target videos; expiry falls back to the video link.
EXPANSION_VIDEO_DOWNLOAD_TIMEOUT_SECONDS: Final[float] = 120.0

# Bound on `/download_video` (yt-dlp or Douyin); stays under Discord's 15-min token limit.
VIDEO_DOWNLOAD_TIMEOUT_SECONDS: Final[float] = 600.0

# Total retry window for empty Threads pages so retries do not starve the subsequent
# link-media budget.
THREADS_EMPTY_PAGE_RETRY_DEADLINE_SECONDS: Final[float] = 10.0

# --------------------------------------------------------------------------------------
# Network reads
# --------------------------------------------------------------------------------------

# Bound on resolving a `facebook.com/share/...` redirect.
SHARE_RESOLVE_TIMEOUT_SECONDS: Final[int] = 10

# Per-request bound on a single Threads page fetch.
THREADS_PAGE_TIMEOUT_SECONDS: Final[int] = 15

# Per-chunk read bound for streamed Threads media downloads (kept separate from page fetch).
THREADS_MEDIA_READ_TIMEOUT_SECONDS: Final[int] = 15

# Per-chunk read bound for streamed expansion video downloads (kept separate from page fetch).
EXPANSION_VIDEO_READ_TIMEOUT_SECONDS: Final[int] = 15

# Bound on fetching one Facebook post page (~950 KB HTML).
FACEBOOK_PAGE_TIMEOUT_SECONDS: Final[int] = 20

# Bound on fetching one Instagram post page (~800 KB HTML).
INSTAGRAM_PAGE_TIMEOUT_SECONDS: Final[int] = 20

# Bound on fetching one Twitter syndication JSON payload.
TWITTER_PAGE_TIMEOUT_SECONDS: Final[int] = 10

# Whole-request bound on FxTwitter conversation fetch; expiry keeps the syndication post.
FXTWITTER_CONVERSATION_TIMEOUT_SECONDS: Final[int] = 15

# Bound on downloading a history image URL during reply rendering.
IMAGE_FETCH_TIMEOUT_SECONDS: Final[int] = 10

# Bound on fetching LiteLLM's upstream price table; expiry falls back to the local mirror.
PRICE_TABLE_FETCH_TIMEOUT_SECONDS: Final[int] = 5

# Timeout in seconds for a Douyin metadata request.
DOUYIN_METADATA_TIMEOUT_SECONDS: Final[int] = 15

# Per-chunk read bound for Douyin media downloads (must not be merged with metadata timeout).
DOUYIN_DOWNLOAD_TIMEOUT_SECONDS: Final[int] = 60

# Max attempts per Douyin media download (multiplies `DOUYIN_DOWNLOAD_TIMEOUT_SECONDS`).
DOUYIN_DOWNLOAD_MAX_RETRIES: Final[int] = 3

# Per-socket yt-dlp timeout (multiplied by `YTDLP_RETRIES`; callers add an outer deadline).
YTDLP_SOCKET_TIMEOUT_SECONDS: Final[int] = 30

# Shared by yt-dlp's `retries`, `fragment_retries`, and `extractor_retries`.
YTDLP_RETRIES: Final[int] = 3

# --------------------------------------------------------------------------------------
# Discord API
# --------------------------------------------------------------------------------------

# Bound on a settled game's final `message.edit` so a hung edit cannot skip cleanup.
GAME_FINAL_EDIT_TIMEOUT_SECONDS: Final[float] = 8.0

# Reserve subtracted from `Interaction.expires_at` (15m) on `/ask` so delivery or its
# failure notice finishes before the interaction token expires (excludes media persona).
INTERACTION_DELIVERY_MARGIN_SECONDS: Final[float] = 60.0

# --------------------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------------------

# SQLite lock-wait timeout in seconds (converted to ms where `PRAGMA busy_timeout` is set).
SQLITE_BUSY_TIMEOUT_SECONDS: Final[float] = 5.0
