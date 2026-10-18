"""Every bound on how much content one model request is allowed to carry.

The single place to look up what the bot puts in front of a model, and the only place to
change one. What belongs here is a cap that, when it binds, leaves the model seeing less while
the request goes out anyway: fewer history messages, fewer attachments, a shorter memory
document, a truncated transcript. That is the whole rule, and it is what separates this module
from `typings/timeouts.py`, where expiry produces a failure or a degradation instead.

Three neighbouring kinds of number are deliberately not here, because each answers a different
question and filing them together would imply they must agree.

**A write-side clamp trims a file, not a request.** `TONE_FILE_MAX_BYTES` is the one that looks
most like it belongs here and does not: the tone note it bounds is injected on every reply, but
the clamp fires on the write, so what it limits is what reaches disk and the reader stays
unbounded. Every clamp of that shape stays beside the write path that applies it.

**A rewrite target tells the model what to produce.** `COMPACTION_TARGET_CHARS` is an
instruction in a prompt, and `COMPACTION_TRIGGER_CHARS` is the switch that turns that
instruction on. Neither bounds what the request may contain, so a value either shares with a
budget below is a coincidence rather than a coupling.

**An output cap bounds what comes back.** `MAX_INLINE_IMAGES`, `DISCORD_MESSAGE_LIMIT` and
`DISCORD_ATTACHMENT_LIMIT` all bound the reply, so a value one shares with a budget here is the
same coincidence again.

Also absent: provider hard limits nobody here chose (`FILES_API_MAX_BYTES` is Google's 2 GB),
UI rendering caps (`REASONING_PREVIEW_MAX_CHARS`, `MEMORY_PAGE_MAX_CHARS`), concurrency, and
cache sizes.

The one bound that reads like an input budget and is not here is
`MAX_BILIBILI_INGEST_DURATION_SECONDS`, which stays beside the fetch it gates. Its whole
justification is that a longer clip cannot finish its download and upload inside
`LINK_MEDIA_TIMEOUT_SECONDS`, so it is a precomputed deadline rather than a judgement about how
much video is worth reading.
"""

from typing import Final

# --------------------------------------------------------------------------------------
# Channel history in one reply
#
# Bounded on three axes because no one of them is the right shape alone, and whichever binds
# first wins.
# --------------------------------------------------------------------------------------

# Discord chat is mostly short one-line messages, so a message count alone either starves a
# chatty channel or lets a channel of long posts blow up the input. Ordinary chat reaches this
# axis first while `HISTORY_CHAR_BUDGET` bounds long posts; anything older than the active
# conversation belongs to long-term memory.
HISTORY_MESSAGE_LIMIT: Final[int] = 100

# Paired with the message limit, this binds only once messages average over 40 characters, so
# what it bounds is a channel of long posts.
HISTORY_CHAR_BUDGET: Final[int] = 8000

# What a history message costs beyond its own text: the rendered form carries an author header,
# and an attachment-only message has empty `content` but still renders a marker standing in for
# it. Without a floor per message a run of image posts would count as free and overshoot.
HISTORY_PER_MESSAGE_OVERHEAD: Final[int] = 40

# How many history attachments ride as real uploaded files. The character budget cannot see this
# cost: an attachment-only message spends only `HISTORY_PER_MESSAGE_OVERHEAD` there while
# re-sending every file to the model on each reply, and the Files-API cache saves only the
# re-upload, never the ~1.1k input tokens per media part. Past this cap, older attachments
# degrade to `[attachment: ...]` markers so the model still knows a file was posted, and the
# answer avoids waiting on heavy history uploads that share `MEDIA_CONCURRENCY` slots across
# replies in flight.
MAX_HISTORY_MEDIA_PARTS: Final[int] = 10

# --------------------------------------------------------------------------------------
# Attachment bytes inlined into one reply
#
# Binds only where an attachment rides in the request itself (base64, or a text file's text)
# rather than as an uploaded file's handle (`InlineRenderer`). Google documents a 100 MB
# per-request limit and rejects base64 payloads well below 97 MB with a 429, while ~77 MB
# succeeds; base64 adds a third to a file's raw size.
# --------------------------------------------------------------------------------------

# Largest attachment inlined at all, checked against the size Discord reports before downloading.
# Matches Discord's non-Nitro upload limit, and encodes to ~28 MB of base64: small enough to fit
# inside `MAX_HISTORY_INLINE_BYTES` and to stay under the ~77 MB request ceiling even when
# rendered twice (in history and again as the replied-to reference message).
MAX_INLINE_ATTACHMENT_BYTES: Final[int] = 20 * 1024 * 1024

# What the history window's attachments may put in the request, counted in the bytes they render
# to, newest kept first. Past it older posts degrade to the `[attachment: ...]` markers the
# media-part cap above uses. Without it, files that fit one at a time add up past the ceiling and
# every later reply resends them until they leave the window. Room for most of the ten parts the
# cap above admits when they are downscaled photos (~3-5 MB of base64 each), with one file at
# `MAX_INLINE_ATTACHMENT_BYTES` of the turn's own beside it still under 77 MB. The current and
# replied-to messages render whole on top, so more than that one file there can still pass it.
MAX_HISTORY_INLINE_BYTES: Final[int] = 48_000_000

# --------------------------------------------------------------------------------------
# Memory read into a request
# --------------------------------------------------------------------------------------

# How many users' memories one reply may carry. Deterministic participants are never displaced:
# if they fill or exceed the target the route is offered no optional candidates, otherwise the
# ones it names can use only the remaining slots.
MEMORY_CONTEXT_TARGET_USERS: Final[int] = 8

# Ceiling on one rendered memory document (the merged compartments injected for one reply).
# Ordinary rendered documents are a few thousand characters at most, so this is a backstop
# that makes a runaway scope degrade to its newest facts plus an explicit notice instead of
# silently bloating every request. Rendering stops at the cap without deleting anything on
# disk, so it never fights the next consolidation over content it would immediately write back.
#
# Re-measure the rendered form, not the fact files on disk: a file's `---` header is routinely
# longer than the body it describes, and only the body is ever rendered.
MEMORY_INJECTION_MAX_CHARS: Final[int] = 30_000

# Not a budget of its own: nothing binds on it and no request is ever shortened by it. It is the
# operator's warning line for the cap above, logged after a write, and it lives here only because
# reading it anywhere other than beside that cap would say nothing.
MEMORY_INJECTION_WARN_CHARS: Final[int] = 24_000

# --------------------------------------------------------------------------------------
# Memory note review and consolidation requests
# --------------------------------------------------------------------------------------

# Transcript truncation for the note review (keeps head and tail, drops the middle). Large on
# purpose: the reply history window should reach the review whole, and the memory models accept
# 1M-token inputs.
MEMORY_TRANSCRIPT_MAX_CHARS: Final[int] = 100_000

# Cap for one memory note the answer model wrote inline, as it reaches the evaluator and, for a
# forget, the consolidation prompts. A note is asked for as one sentence, so this only bounds a
# model that ignored that; it is per note rather than per batch because `MAX_MEMORY_NOTES` already
# bounds the count, and because a batch cap would let one runaway note starve the rest.
MEMORY_NOTE_MAX_CHARS: Final[int] = 400

# How many notes of one kind survive a merge, when several turns were skipped while one memory
# update was in flight and their notes were folded into one payload. It bounds the request that
# carries the accumulated result, not what one answer may emit. Two replies' worth, because a
# merge that deep already means the pipeline is behind.
MEMORY_MERGED_NOTES_MAX: Final[int] = 10

# Cap for the bot's own reply inside that transcript. The reply is secondary evidence and is
# appended last, so without this cap a long (e.g. a long summary) reply fills the entire kept tail
# and the middle-truncation drops the current user message right before it.
MEMORY_REPLY_MAX_CHARS: Final[int] = 8_000

# Tail window of the detail file fed to consolidation as low-trust provenance. Effectively the
# whole evidence log for any realistic user: this bot injects memory exactly once per reply with
# no on-demand retrieval, so the stored facts must be distilled from the full evidence base in the
# background. The bound only keeps a pathological log inside the consolidation input window
# (~500k zh-TW chars stays well under the 1M-token window with the stored facts and raw batch on
# top). `DETAIL_FILE_MAX_BYTES` is sized against this and must stay above it.
MEMORY_DETAIL_CONTEXT_MAX_CHARS: Final[int] = 500_000

# --------------------------------------------------------------------------------------
# Linked posts read for one reply
# --------------------------------------------------------------------------------------

# Cap on media parts injected for a whole Threads block, shared by the linked post and the post it
# quotes rather than counted per post. It bounds what one answer pays for a huge carousel: each
# part costs a fetch and an upload as well as the tokens it spends.
MAX_THREADS_MEDIA_PARTS: Final[int] = 10

# Cap on posts rendered from a Threads reply chain, mirroring the cog's deep-chain trim: a linked
# reply deep in a long thread would otherwise render every ancestor's text and bloat or overflow
# the answer input. The chain is ordered oldest-first, so the tail keeps the target plus its
# nearest ancestors.
MAX_THREADS_POSTS: Final[int] = 6

# Cap on the comments rendered below the target, counted across every branch. Kept on its own axis
# rather than sharing `MAX_THREADS_POSTS`: the chain is the context leading up to the linked post,
# the comments are the discussion under it, and one should never squeeze the other out. As of
# 2026-09, the page ships at most ten reply branches with an unfollowed cursor for the rest, so
# this is a backstop rather than a policy.
MAX_THREADS_REPLIES: Final[int] = 30

# Cap on images ingested from a Douyin photo post. Each costs a download plus an upload, and a
# model reading eight frames of a gallery already has the gist; the cog's Discord-side cap is
# separate and larger, because attaching a file is far cheaper than tokenizing it.
MAX_DOUYIN_INGEST_IMAGES: Final[int] = 8

# Cap on images ingested from a Facebook post, sized to match the Douyin one for the same
# reason: each costs a fetch plus an upload on the reply's critical path. Deliberately larger
# than what `parse_facebook` renders into the channel, because the two bound different things —
# that one stops the expansion scrolling the channel, this one bounds what the answer pays for.
MAX_FACEBOOK_INGEST_IMAGES: Final[int] = 8

# Cap on the comments injected below a Facebook post. Logged out, the page preloads only a
# handful of comments, so this is a backstop against an unusually long preload rather than a
# policy about how much discussion to read.
MAX_FACEBOOK_COMMENTS: Final[int] = 10

# Cap on images ingested from an Instagram post, sized to match the other two for the same
# reason: each costs a fetch plus an upload on the reply's critical path. A carousel routinely
# carries nine or ten, so this one really does bind, unlike Facebook's.
MAX_INSTAGRAM_INGEST_IMAGES: Final[int] = 8

# Cap on images ingested from a Twitter post. The reader already keeps at most four media items,
# so this binds on nothing today; it is here for the reason the Instagram one is: the cost is a
# fetch plus an upload each, on the reply's critical path, and a reader that starts keeping more
# should not silently raise what one reply spends. Twitter's video is deliberately not ingested at
# all, matching Facebook and Instagram — the clip rides as a link the model is told it did not watch.
MAX_TWITTER_INGEST_IMAGES: Final[int] = 4

# Cap on the comments injected below an Instagram post. Higher than Facebook's because the page
# carries the whole list rather than a preload, so this is what keeps a viral post's discussion
# from crowding out the rest of the request.
MAX_INSTAGRAM_COMMENTS: Final[int] = 20

# Render-time cap on the Bilibili description injected as text. Descriptions can run to thousands
# of characters of tags and sponsor text; the head is where the signal lives.
MAX_BILIBILI_DESCRIPTION_CHARS: Final[int] = 1000

# --------------------------------------------------------------------------------------
# Requests that are not the reply
# --------------------------------------------------------------------------------------

# How many subject reference images ride into an omni video render, and into the prompt refine in
# front of it. Shared so the director grounds on exactly the set the render will send, rather than
# describing references omni never receives (and uploading those unused bytes on the path). omni
# accepts a handful, so the provider bounds this from above; the number here is what the two calls
# agree on.
MAX_VIDEO_REFERENCE_IMAGES: Final[int] = 3
