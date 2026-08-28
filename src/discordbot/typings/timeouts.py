"""Every wall-clock bound whose expiry produces a failure or a degradation.

The single place to look up what the bot's deadlines are, and the only place to change one.
What belongs here is a bound a user feels when it fires: a request that gives up, a reply
that answers without the part it was waiting for, a command that reports failure. What does
not is the same kind of number spent on a different kind of decision -- animation pacing,
cache TTLs, retry cadences, `@tasks.loop` intervals, and the idle expiry of a Discord view.
Those stay beside the code they pace, where reading them in context is worth more than
reading them next to each other, and pulling them here would make both halves harder to
read. The line cuts through `cogs/games/blackjack_views.py` and `utils/threads.py`, which
each hold one of both kinds two lines apart.

A retry count sits here when it multiplies a bound rather than being one: a 30s socket
timeout does not mean a download gives up in 30s, and reading the timeout without the count
gives the wrong answer.

Three omissions are decisions, not oversights.

`MEDIA_HOSTING_RETENTION_HOURS` stays in the environment: its expiry is not a failure, so it
falls outside the rule above, and it is the only time value a deployment was ever given to
tune.

Deep research (`cogs/research/agent.py`) is deliberately unbounded. The agent settles
server-side on its own budget and the SDK bounds each individual request, so a ceiling here
could only abandon a run that is still working.

And a bound that wraps a single LLM call purely as a liveness backstop belongs nowhere
rather than here, because the backend already owns that deadline. Measured in this project's
env: `openai` 2.46.0 defaults to `Timeout(connect=5.0, read=600, write=600, pool=600)` with
`max_retries=2`, and nothing here passes `max_retries`, so an `AsyncOpenAI` call bounds itself
at roughly half an hour rather than at 600s. That is a real ceiling but a loose one, which is
why the test for deleting such a wrapper is whether anything is WAITING, not whether the
numbers match: on background work nobody waits for, half an hour is a liveness backstop doing
its job, and a tighter wrapper would only be restating it for free.

The LLM bounds that DO appear below are the ones that survive that test. `google-genai` 2.13.0
defaults to `timeout=None`, so on a direct-to-Google path the wrapper is the only bound there
is; everything else here is a product deadline that happens to sit on an LLM call -- a number
chosen because something downstream must not wait, not because the provider might hang.
"""

from typing import Final

# --------------------------------------------------------------------------------------
# Reply pipeline (`gen_reply`)
# --------------------------------------------------------------------------------------

# Optional third-party memory selection overlaps the route call for free: the QA path joins
# the speculative prep task only after the route returns, so selection runs unbounded while
# the route is still in flight. Once the route completes, a still-running selection gets only
# this grace before the reply answers with its deterministic participant memories, so a slow
# selector can never cost the author, reply-chain authors, or explicitly mentioned users.
# Tune against the `gen_reply memory selection done` latency log.
RECALL_SELECT_GRACE_SECONDS: Final[float] = 2.0

# Effort grading runs in parallel with the route under the same `route_done` gate as
# memory selection: it runs unbounded while the route is in flight and gets only this
# grace once the route returns before the reply falls back to "high" effort. The grade
# is consumed only just before the answer model starts, so this latency hides behind the
# route. Tune against the `gen_reply effort done` latency log.
EFFORT_GRACE_SECONDS: Final[float] = 5.0

# How many times the streaming answer turn is opened before the reply gives up on a transient
# upstream failure. A count rather than a cadence, which is why it sits here while the backoff
# between the attempts stays beside the retry in `gen_reply/streaming.py`: it multiplies the
# `openai` client's own ceiling that this module's docstring measures and deliberately does not
# restate, so the worst case for one reply is this many times that, and reading either half
# alone gives the wrong answer. Deliberately small: every attempt after the first is spent with
# the user watching a thinking preview that has already stalled once.
ANSWER_STREAM_MAX_ATTEMPTS: Final[int] = 3

# An intent-selected linked-post context build gets this grace once the QA path resolves it.
# Far wider than memory/effort because it fetches the post's media and uploads it to the Files
# API, and because answering blind about a link the user explicitly pointed at is the failure
# this feature exists to prevent. The builder bounds its own media step just under this and
# degrades to text, so the grace is a backstop rather than the usual exit. It starts only after
# routing so an incidental link never begins network work, then overlaps any remaining context
# preparation and effort grading. Tune against the `gen_reply link context done` latency log.
LINK_CONTEXT_GRACE_SECONDS: Final[float] = 180.0

# How far under the grace the media step gives up, so the builder degrades to its text block
# itself instead of being cancelled with nothing to show. Two healthy fetches fit in it.
LINK_MEDIA_DEGRADE_MARGIN_SECONDS: Final[float] = 10.0

# Bound on the whole fetch + upload step for the media of a linked post, shared by every link
# context builder. Derived rather than restated: the two numbers used to live in separate files
# describing each other in prose, and this ordering is the whole reason the bound exists. Set
# well above a normal clip's cost -- watching the linked video is the point, and the text block
# is already on hand, so waiting is cheaper than answering blind.
LINK_MEDIA_TIMEOUT_SECONDS: Final[float] = (
    LINK_CONTEXT_GRACE_SECONDS - LINK_MEDIA_DEGRADE_MARGIN_SECONDS
)

# How long an abandoned link-media download gets to notice its stop signal before the scratch
# dir is removed anyway. The signal fires at the next yt-dlp progress tick, typically well under
# a second; a worker that outlives this window is stalled on the network, not downloading.
DOWNLOAD_STOP_JOIN_SECONDS: Final[float] = 5.0

# How long a Discord attachment's Files API upload is given to reach ACTIVE. Past it the
# upload becomes a `PendingUpload` the caller caches and re-polls on the next reference, so
# this reply answers without that attachment rather than waiting for it. Short because the
# poll overlaps the route and memory-selection calls: a small file is ACTIVE instantly and
# only a large one spends any of that window.
ATTACHMENT_ACTIVATION_TIMEOUT_SECONDS: Final[float] = 15.0

# Bound on one xAI Files API upload, and the only deadline that call has: `xai-sdk` registers
# timeout interceptors for unary-unary and unary-stream RPCs only, while `Files.UploadFile` is
# client-streaming, so the 27-minute default it advertises reaches every RPC except the one this
# project uses, and the call site passes none of its own. Sized by where it sits rather than by
# the other Files-API bounds: an attachment renders while the answer input is still being built,
# so expiry here is dead air in front of a user with nothing on screen, the same position that
# holds ATTACHMENT_ACTIVATION_TIMEOUT_SECONDS to 15s. Wider than that one because it covers a
# whole transfer (up to xAI's 48 MB cap) rather than a poll over an upload already sent, and
# because expiry drops the attachment outright where the Gemini path defers it to a re-poll.
GROK_FILE_UPLOAD_TIMEOUT_SECONDS: Final[float] = 30.0

# Bound on the whole Files API upload of a generated clip the persona reply then watches:
# `upload_to_files_api` covers the transfer as well as the ACTIVE poll under this one timeout,
# started once an upload slot is free. Generous relative to an image because video sits in
# PROCESSING longer, but far under the link-media bound: the clip was just produced here, so it
# is small and known-good.
GENERATED_VIDEO_ACTIVATION_TIMEOUT_SECONDS: Final[float] = 60.0

# --------------------------------------------------------------------------------------
# Generated media
#
# The VIDEO route's real worst case is a sum these two used to hide in separate files:
# VIDEO_RENDER bounds the omni call ALONE, and the source upload and the URI download each
# take FILES_READY on top of it, so an edit can legitimately run to
# VIDEO_RENDER + 2 * FILES_READY before anything is wrong.
# --------------------------------------------------------------------------------------

# Bound for waiting on a Files API entry to become usable: the source video uploaded for an omni
# edit (polled to ACTIVE) and the URI-delivered generated clip (download retried until it lands).
# Generous because a large clip can sit in PROCESSING a while; the render hard-fails past it, since
# video is the primary deliverable.
FILES_READY_TIMEOUT_SECONDS: Final[float] = 180.0

# Hard ceiling on the whole omni video render (a single blocking interactions.create) so a hung
# provider job cannot leave the message handler waiting forever. Direct to Google, where the SDK
# applies no deadline of its own, so this is the only bound the call has.
VIDEO_RENDER_TIMEOUT_SECONDS: Final[float] = 600.0

# Bound for the inline-music best-effort path, mirroring the inline-image timeout: the render
# runs after the text reply is on screen, so the wait only delays this message's own clip. Also
# direct to Google, so likewise the only bound the call has.
MUSIC_RENDER_TIMEOUT_SECONDS: Final[float] = 300.0

# Bound for the inline-image best-effort path: the render runs after the text reply is already
# on screen, so the wait only delays this message's own image, never others. Generous (mirrors
# VOICE_TIMEOUT_SECONDS) so a slower render still has room to land.
INLINE_IMAGE_TIMEOUT_SECONDS: Final[float] = 300.0

# Bound: a request timeout so a slow/hung clip cannot keep this message's own pipeline (its final
# status reaction + memory scheduling) waiting. The synthesis is per-message and runs after the text
# is already on screen, so the wait only delays its own message, never others; it is generous so a
# longer spoken reply has room to render. There is deliberately no spoken-length cap: the answer
# model decides how much to say.
VOICE_TIMEOUT_SECONDS: Final[float] = 300.0

# Bound for the prompt-refinement call: it sits SERIALLY before the image/video render on the
# IMAGE/VIDEO critical path, so a hung director must not keep the route waiting forever. On
# timeout the refine falls back to the raw user prompt like any other failure.
PROMPT_REFINE_TIMEOUT_SECONDS: Final[float] = 120.0

# --------------------------------------------------------------------------------------
# Feature deadlines that happen to sit on an LLM call
#
# Each of these is far tighter than the SDK's own 600s because a late answer is worthless
# here, not because the provider might hang. That is what keeps them from being the liveness
# backstops this module deliberately does not carry.
# --------------------------------------------------------------------------------------

# Auto-unmute replies are off the critical path; bound the call so a hung provider never
# leaves the best-effort post-timeout reply pending forever.
AUTO_UNMUTE_AI_TIMEOUT_SECONDS: Final[float] = 10.0

# Bound the small research-thread title side call; on timeout/failure the brief's first line
# is used, and the thread is named while the requester is watching it appear.
THREAD_TITLE_TIMEOUT_SECONDS: Final[float] = 15.0

# --------------------------------------------------------------------------------------
# Background LLM work
# --------------------------------------------------------------------------------------

# Ceiling on a whole consolidation fan-out, which is a LOOP of one call per compartment plus
# the disk work between them -- a shape no per-request backend deadline can bound. Kept because
# its expiry is what releases the scope lock and the global-concurrency permit: past it, that
# user or server would never get another memory update. It is a liveness backstop, not a latency
# or cost guard; a slow background rewrite is harmless.
MEMORY_CONSOLIDATE_TIMEOUT_SECONDS: Final[float] = 600.0

# --------------------------------------------------------------------------------------
# Link expansion and downloads
# --------------------------------------------------------------------------------------

# Bound on one expansion's Douyin work. It exists to cap how long a single paste can hold the
# fetch slot it shares with the reply path, not to hurry the download along: a healthy post
# finishes in seconds, while a stalling CDN would otherwise retry its way into the tens of
# minutes. A timeout is reported as a plain failure, never as a missing post.
DOUYIN_EXPAND_TIMEOUT_SECONDS: Final[float] = 120.0

# The same bound for the Threads expansion, and deliberately NOT the same number: that cog
# holds no shared fetch slot, so this caps the listener rather than a queue behind it, and one
# expansion walks a whole conversation (`MAX_THREADS_POSTS` page fetches at
# `THREADS_PAGE_TIMEOUT_SECONDS` each, plus the empty-page retry deadline, plus the target's
# video) where a Douyin expansion reads one post. Set to what the reply pipeline already allows
# the very same walk (`LINK_CONTEXT_GRACE_SECONDS`) rather than to a second guess.
THREADS_EXPAND_TIMEOUT_SECONDS: Final[float] = 180.0

# Bound on `/download_video`'s whole download step, both the yt-dlp and the Douyin branch.
# Wider than an auto-expansion because the user asked for this file by name and may be after a
# long one, and sized under Discord's 15-minute deferred-interaction token so the failure edit
# still lands on the message the command is holding.
VIDEO_DOWNLOAD_TIMEOUT_SECONDS: Final[float] = 600.0

# Ceiling on the Threads empty-page retry loop, measured from the first attempt, and sized
# against the reply pipeline rather than against the fetch: a retry that eventually succeeds is
# followed by the media step, which already claims almost all of `LINK_CONTEXT_GRACE_SECONDS` on
# its own (`LINK_MEDIA_TIMEOUT_SECONDS`). Two more healthy fetches (~3s each) fit inside this; a
# run of slow ones stops early instead of pushing the whole block past the grace. A retry that
# never succeeds costs nothing extra downstream, since an unreadable post skips the media step.
THREADS_EMPTY_PAGE_RETRY_DEADLINE_SECONDS: Final[float] = 10.0

# --------------------------------------------------------------------------------------
# Network reads
# --------------------------------------------------------------------------------------

# Redirect chases for a facebook.com/share/... link. Fixed rather than configurable: it bounds
# one HEAD/GET against Facebook and nothing has ever needed a different value.
SHARE_RESOLVE_TIMEOUT_SECONDS: Final[int] = 10

# Bound on one Threads page fetch. A conversation walk makes several of these, so what bounds
# the walk is the caller's own deadline, not this.
THREADS_PAGE_TIMEOUT_SECONDS: Final[int] = 15

# Bound on the Threads media download. Separate from the page fetch despite matching it today:
# the download is streamed, so `requests` reads this as a gap-between-chunks bound rather than
# a whole-request one, and a slow-drip CDN can hold it open indefinitely.
THREADS_MEDIA_READ_TIMEOUT_SECONDS: Final[int] = 15

# 10s caps the history-render I/O tail: a URL taking longer is almost always a dead/slow CDN
# that would fail anyway, and a 30s wait let one such source dominate the whole render. Healthy
# media.discordapp.net images return well under 1s.
IMAGE_FETCH_TIMEOUT_SECONDS: Final[int] = 10

# Short because the price table is never on a critical path: the fetch falls back to the on-disk
# mirror and `price_table_task` re-fetches later. It bites only on a cold start with no mirror,
# where it costs a `$0.00000000` footer AND a modality baseline that drops audio and video.
PRICE_TABLE_FETCH_TIMEOUT_SECONDS: Final[int] = 5

# Timeout in seconds for a Douyin metadata request.
DOUYIN_METADATA_TIMEOUT_SECONDS: Final[int] = 15

# Separate from the metadata bound because this one bounds the gap between chunks of a video that
# can run to tens of megabytes; the metadata timeout is far too tight for that and was observed
# aborting an otherwise healthy transfer.
DOUYIN_DOWNLOAD_TIMEOUT_SECONDS: Final[int] = 60

# Attempts made per Douyin media download before giving up. Multiplies the bound above.
DOUYIN_DOWNLOAD_MAX_RETRIES: Final[int] = 3

# Per-socket bound inside yt-dlp. It is NOT a ceiling on a download: it applies per socket, and
# every one of the retry counts below multiplies it, which is why the callers carry their own
# outer deadline.
YTDLP_SOCKET_TIMEOUT_SECONDS: Final[int] = 30

# Shared by yt-dlp's `retries`, `fragment_retries` and `extractor_retries`, each of which
# multiplies the socket timeout above.
YTDLP_RETRIES: Final[int] = 3

# --------------------------------------------------------------------------------------
# Discord API
# --------------------------------------------------------------------------------------

# Bound on a settled game round's final `message.edit`. Money is already committed by then, so
# expiry leaves the table showing the previous frame while the balances are correct; the bound
# exists so a hung edit cannot skip the cleanup scheduling behind it.
GAME_FINAL_EDIT_TIMEOUT_SECONDS: Final[float] = 8.0

# What a `/ask` turn keeps back from the media it is generating, so whatever it produces still has
# somewhere to land. Discord invalidates an interaction token 15 minutes after the command was
# invoked and EVERY message such a turn writes goes through it, the deferred "thinking" state
# included, so a render that finishes late costs the whole turn: the clip 404s, and so does the
# notice that would have explained it, leaving a thinking state that never resolves. This module
# owns the margin rather than the window because the window is Discord's --
# `TurnSurface.delivery_budget_seconds` reads it off nextcord's own `Interaction.expires_at` rather
# than restating it here, so the one number to change is what we hold in reserve. Sized for the
# largest attachment Discord accepts plus the error embed behind it, and deliberately NOT for the
# best-effort persona reply after that: its own Files upload can spend this whole margin, which is
# why a clip delivered on the last of the budget is handed over without the bot saying a word
# about it. That is the existing convention (a media persona reply fails silently) rather than a
# new cost.
INTERACTION_DELIVERY_MARGIN_SECONDS: Final[float] = 60.0

# --------------------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------------------

# How long a writer waits out contention before SQLite raises `database is locked` and the
# command fails in front of the user. Written in seconds like everything else here; the PRAGMA
# is the one surface that wants milliseconds, so it does the conversion at its own call site.
SQLITE_BUSY_TIMEOUT_SECONDS: Final[float] = 5.0
