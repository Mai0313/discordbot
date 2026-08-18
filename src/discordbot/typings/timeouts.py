"""Every wall-clock bound whose expiry produces a failure or a degradation.

This is the single place to look up or change a deadline the bot actually enforces, so a new
bound can be picked with a view of the ones it interacts with instead of in isolation. It holds
constants only and imports nothing from `cogs/`, `services/` or `utils/`, so every layer can read
it.

**What is in.** A bound whose expiry makes something not work, or fall back to a lesser result.
**What is out**, deliberately, and stays beside the code it paces: animation timing, cache TTLs,
retry cadences, `@tasks.loop` intervals, and Discord view idle expiry. Those are the same kind of
number but a different kind of decision — a view expiring is designed behaviour, not a failure —
and pulling `PEEK_REVEAL_DELAY_SECONDS` out of the blackjack animation would make that code harder
to read, not easier. `MEDIA_HOSTING_RETENTION_HOURS` and `FEEDBACK_SUBMIT_COOLDOWN_SECONDS` are out
by the same rule and stay environment-backed.

Retry COUNTS are out too, even where they multiply an effective bound (yt-dlp's `retries` /
`fragment_retries` / `extractor_retries`, Douyin's `max_retries`): a count has no expiry, so it is
not a bound, and `DOWNLOAD_TIMEOUT_SECONDS` caps the product of all of them anyway.

**An LLM call is bounded by its provider, not by an `asyncio.timeout` of ours.** On the proxy that
needs nothing: `AsyncOpenAI` defaults to connect 5s / read 600s and raises `APITimeoutError`, which
every call site already degrades through on its broad `except`, exactly as it degrades on a proxy
`ServiceUnavailableError`. Deep research (`cogs/research/agent.py`) rides that same rule with no
number of its own, the agent settling server-side on its own budget.

**`genai.Client` is the exception, and it is not optional.** google-genai leaves
`http_options.timeout` at `None` (measured against 2.13.0; `gen_reply/files_api.py` has the same
finding from its own investigation), so a direct-to-Google call into a black-holed connection never
returns and no `except` is ever reached — a hang, not an error. Those calls therefore pass a
deadline as the SDK's own per-request `timeout=`, which is still the provider owning it rather than
a wrapper around it, and the number lives here.

The other carve-out is a **product deadline that happens to sit on an LLM call**: a bound on how
long the feature may block something ELSE, which no provider can know about. Those keep an
`asyncio.timeout` at their own call site, and this is the whole list.
"""

from typing import Final

# ----- reply pipeline ---------------------------------------------------------------------

# Optional third-party memory selection overlaps the route call for free: the QA path joins
# the speculative prep task only after the route returns, so selection runs unbounded while
# the route is still in flight. Once the route completes, a still-running selection gets only
# this grace before the reply answers with its deterministic participant memories, so a slow
# selector can never cost the author, reply-chain authors, or explicitly mentioned users.
# Tune against the `gen_reply memory selection done` latency log.
MEMORY_SELECT_GRACE_SECONDS: Final[float] = 2.0

# Effort grading runs in parallel with the route under the same `route_done` gate as
# memory selection: it runs unbounded while the route is in flight and gets only this
# grace once the route returns before the reply falls back to "high" effort. The grade
# is consumed only just before the answer model starts, so this latency hides behind the
# route. Tune against the `gen_reply effort done` latency log.
EFFORT_GRACE_SECONDS: Final[float] = 5.0

# An intent-selected linked-post context build gets this grace once the QA path resolves it.
# Far wider than memory/effort because it fetches the post's media and uploads it to the Files
# API, and because answering blind about a link the user explicitly pointed at is the failure
# this feature exists to prevent. The builder bounds its own media step just under this
# (`LINK_MEDIA_TIMEOUT_SECONDS`) and degrades to text, so the grace is a backstop rather than the
# usual exit. It starts only after routing so an incidental link never begins network work, then
# overlaps any remaining context preparation and effort grading. Tune against the
# `gen_reply link context done` latency log.
LINK_CONTEXT_GRACE_SECONDS: Final[float] = 180.0

# How far the media step finishes ahead of the grace above. It only has to cover the builder's
# own text assembly and hand-back, which is in-memory work.
LINK_MEDIA_DEGRADE_HEADROOM_SECONDS: Final[float] = 10.0

# Bound on the whole fetch + upload step for the media of a linked post, shared by every link
# context builder. It exists so the builder always returns within the pipeline's post-route grace
# and degrades to text itself, rather than being cancelled with nothing to show. DERIVED rather
# than restated so that relationship cannot drift: the two used to describe each other in prose
# from separate files, with neither able to see the other's number.
LINK_MEDIA_TIMEOUT_SECONDS: Final[float] = (
    LINK_CONTEXT_GRACE_SECONDS - LINK_MEDIA_DEGRADE_HEADROOM_SECONDS
)

# Bound on the whole Files API upload of a generated clip the persona reply then watches:
# `upload_to_files_api` covers the transfer as well as the ACTIVE poll under this one timeout,
# started once an upload slot is free. Generous relative to an image because video sits in
# PROCESSING longer, but far under the link-media bound: the clip was just produced here, so it
# is small and known-good.
GENERATED_VIDEO_ACTIVATION_TIMEOUT_SECONDS: Final[float] = 60.0

# Bound for waiting on a Files API entry to become usable: the source video uploaded for an omni
# edit (polled to ACTIVE) and the URI-delivered generated clip (download retried until it lands).
# Generous because a large clip can sit in PROCESSING a while; the render hard-fails past it, since
# video is the primary deliverable. Applies per step, so an edit that uploads and then downloads
# can spend it twice.
FILES_READY_TIMEOUT_SECONDS: Final[float] = 180.0

# How long an abandoned download gets to notice its stop signal before the scratch dir is
# removed anyway. The signal fires at the next yt-dlp progress tick, typically well under a
# second; a worker that outlives this window is stalled on the network, not downloading.
DOWNLOAD_STOP_JOIN_SECONDS: Final[float] = 5.0

# Deadline for the Threads empty-page retry: past it the post is given up on as unreadable and the
# model is told so explicitly. Kept small because it is spent INSIDE the pipeline's media step,
# which already claims almost all of `LINK_CONTEXT_GRACE_SECONDS` on its own. Two more healthy
# fetches (~3s each) fit inside this; a run of empty pages is a throttle, not a slow link.
THREADS_EMPTY_PAGE_RETRY_DEADLINE_SECONDS: Final[float] = 10.0

# ----- LLM calls: the provider's deadline, carried here only where it needs a number ---------

# Handed to `interactions.create` as its own per-request timeout on the omni video render. Without
# it the render is unbounded, and video is the primary deliverable, so a hung provider job would
# leave the VIDEO route awaiting forever: no clip, no error, the status reaction stuck.
VIDEO_RENDER_TIMEOUT_SECONDS: Final[float] = 600.0

# The same, for the inline `<generate-music>` Lyria render. Best-effort rather than primary, but an
# unbounded hang here stalls the streamer's single media-attach gather, so the reply's final
# reaction and its memory scheduling never run either.
MUSIC_RENDER_TIMEOUT_SECONDS: Final[float] = 300.0

# ----- product deadlines that sit on an LLM call ---------------------------------------------

# Not a transport bound: `ensure_due_stock_news` holds the process-wide news generation lock across
# its whole provider fan-out, and `get_stock_detail` / `get_stock_news` / `get_stock_portfolio` take
# that same lock on the path a user is waiting on. Past ~3s the 近期新聞 button fails outright,
# since it does not defer before the call. Short because the tier is flash-lite at minimal effort,
# so this is already the slow end of a healthy call; expiry writes the deterministic template.
STOCK_NEWS_AI_TIMEOUT_SECONDS: Final[float] = 4.0

# Also not a transport bound: the title is generated BEFORE the research thread exists, so the
# caller sees nothing at all until it returns. Expiry falls back to the brief's first line.
THREAD_TITLE_TIMEOUT_SECONDS: Final[float] = 15.0

# ----- link downloads ---------------------------------------------------------------------

# The outer bound on every user-facing link download: `/download_video` on both its yt-dlp and
# Douyin branches, and the Threads and Douyin auto-expansions. One constant over all four because
# they are the same promise to the user — a paste or a command either produces media or reports a
# failure — and three of them had no bound at all, which is how `/download_video` came to strand
# the caller on its progress message with no exit. Wide because the work is a real transfer over
# someone else's CDN and a healthy post finishes in seconds; it caps the stall, not the download.
# It is also what makes yt-dlp's and Douyin's retry counts stop multiplying into the tens of
# minutes. A timeout is reported as a plain failure, never as a missing post.
DOWNLOAD_TIMEOUT_SECONDS: Final[float] = 300.0

# Redirect chase for a facebook.com/share/... link. Expiry falls through to the unresolved short
# URL, which usually fails the download a step later.
SHARE_RESOLVE_TIMEOUT_SECONDS: Final[int] = 10

# Per-socket read bound handed to yt-dlp. Not a bound on the download as a whole: it applies per
# socket and yt-dlp retries, which is what `DOWNLOAD_TIMEOUT_SECONDS` is over the top of.
YTDLP_SOCKET_TIMEOUT_SECONDS: Final[int] = 30

# Douyin metadata request. Separate from the media bound below because that one covers the gap
# between chunks of a video that can run to tens of megabytes; this value is far too tight for
# that and was observed aborting an otherwise healthy transfer.
DOUYIN_METADATA_TIMEOUT_SECONDS: Final[int] = 15
DOUYIN_DOWNLOAD_TIMEOUT_SECONDS: Final[int] = 60

# Per-read bound on a Threads page fetch and on a Threads media download. Per read only, so a
# slow-drip CDN is bounded by `DOWNLOAD_TIMEOUT_SECONDS` on the expansion path and by
# `LINK_MEDIA_TIMEOUT_SECONDS` on the reply path rather than by this.
THREADS_REQUEST_TIMEOUT_SECONDS: Final[int] = 15

# ----- other network ----------------------------------------------------------------------

# Caps the history-render I/O tail on one image fetched from a URL: a URL taking longer is almost
# always a dead/slow CDN that would fail anyway, and a 30s wait let one such source dominate the
# whole render. Healthy media.discordapp.net images return well under 1s. On expiry that one image
# never reaches the model and the reply is otherwise normal.
IMAGE_FETCH_TIMEOUT_SECONDS: Final[int] = 10

# The LiteLLM price table fetch. Harmless in steady state — it falls back to the on-disk mirror,
# and `price_table_task` re-fetches every 30 minutes. It bites only on a cold start with no mirror,
# and then it costs more than a wrong footer: the usage line reads `$0.00000000` AND the modality
# baseline collapses to {"text", "image"}, so audio and video attachments are dropped before any
# model call.
MODEL_PRICE_FETCH_TIMEOUT_SECONDS: Final[int] = 5

# Per-request ceiling for the GitHub App auth calls and the issue API. Short on purpose: every
# caller is either on the `/feedback` submit path or on a panel someone is waiting for. A report is
# never lost when this fires — the retry loop owns that — but the user is told their report is
# queued rather than filed.
GITHUB_REQUEST_TIMEOUT_SECONDS: Final[float] = 15.0

# ----- Discord API ------------------------------------------------------------------------

# Bound on the settled round's last `message.edit`, shared by Blackjack and Dragon Gate. Money is
# already committed by then, so expiry leaves the table showing the previous frame while the
# balances are correct; the bound exists so a hung edit cannot skip the cleanup scheduling behind
# it.
FINAL_EDIT_TIMEOUT_SECONDS: Final[float] = 8.0

# ----- storage ----------------------------------------------------------------------------

# How long a writer waits out SQLite contention before the command fails in front of the user with
# `database is locked`. In milliseconds because that is the PRAGMA's own unit; the name carries it
# so the one odd unit in this module is visible rather than hidden behind a conversion.
SQLITE_BUSY_TIMEOUT_MS: Final[int] = 5000

__all__ = [
    "DOUYIN_DOWNLOAD_TIMEOUT_SECONDS",
    "DOUYIN_METADATA_TIMEOUT_SECONDS",
    "DOWNLOAD_STOP_JOIN_SECONDS",
    "DOWNLOAD_TIMEOUT_SECONDS",
    "EFFORT_GRACE_SECONDS",
    "FILES_READY_TIMEOUT_SECONDS",
    "FINAL_EDIT_TIMEOUT_SECONDS",
    "GENERATED_VIDEO_ACTIVATION_TIMEOUT_SECONDS",
    "GITHUB_REQUEST_TIMEOUT_SECONDS",
    "IMAGE_FETCH_TIMEOUT_SECONDS",
    "LINK_CONTEXT_GRACE_SECONDS",
    "LINK_MEDIA_DEGRADE_HEADROOM_SECONDS",
    "LINK_MEDIA_TIMEOUT_SECONDS",
    "MEMORY_SELECT_GRACE_SECONDS",
    "MODEL_PRICE_FETCH_TIMEOUT_SECONDS",
    "MUSIC_RENDER_TIMEOUT_SECONDS",
    "SHARE_RESOLVE_TIMEOUT_SECONDS",
    "SQLITE_BUSY_TIMEOUT_MS",
    "STOCK_NEWS_AI_TIMEOUT_SECONDS",
    "THREADS_EMPTY_PAGE_RETRY_DEADLINE_SECONDS",
    "THREADS_REQUEST_TIMEOUT_SECONDS",
    "THREAD_TITLE_TIMEOUT_SECONDS",
    "VIDEO_RENDER_TIMEOUT_SECONDS",
    "YTDLP_SOCKET_TIMEOUT_SECONDS",
]
