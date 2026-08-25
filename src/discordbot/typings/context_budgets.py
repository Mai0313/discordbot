"""Every bound on how much content one model request is allowed to carry.

The single place to look up what the bot puts in front of a model, and the only place to
change one. What belongs here is a cap that, when it binds, leaves the model seeing LESS while
the request goes out anyway: fewer history messages, fewer attachments, a shorter memory
document, a truncated transcript. That is the whole rule, and it is what separates this module
from `typings/timeouts.py`, where expiry produces a failure or a degradation instead.

Three neighbouring kinds of number are deliberately NOT here, because each answers a different
question and filing them together would imply they must agree.

**A write-side clamp trims a file, not a request.** `TONE_FILE_MAX_BYTES` is the one that looks
most like it belongs here and does not: its comment talks about the always-injected tone note,
but it fires inside `write_tone`, so what it bounds is what reaches disk. The reader is
unbounded. `RAW_FILE_MAX_BYTES`, `DETAIL_FILE_MAX_BYTES` and `DETAIL_FILE_TRIM_TARGET_BYTES` are
the same shape, and they stay in `services/memory/constants.py` beside the write paths that
apply them.

**A rewrite target tells the model what to produce.** `COMPACTION_TARGET_CHARS` is an
instruction in a prompt, and `COMPACTION_TRIGGER_CHARS` is the switch that turns that
instruction on. Neither bounds what the request may contain, and `COMPACTION_TRIGGER_CHARS`
being 30_000 like `MEMORY_INJECTION_MAX_CHARS` below is a coincidence rather than a coupling.

**An output cap bounds what comes back.** `MAX_INLINE_IMAGES`, `DISCORD_MESSAGE_LIMIT` and
`DISCORD_ATTACHMENT_LIMIT` all bound the reply. `MAX_HISTORY_MEDIA_PARTS` here is 10 and
`DISCORD_ATTACHMENT_LIMIT` is also 10, the same coincidence again.

Also absent: provider hard limits nobody here chose (`FILES_API_MAX_BYTES` is Google's 2 GB),
UI rendering caps (`REASONING_PREVIEW_MAX_CHARS`, `MEMORY_PAGE_MAX_CHARS`), concurrency, and
cache sizes.

The one bound that reads like an input budget and is not here is
`MAX_BILIBILI_INGEST_DURATION_SECONDS`, which stays in `cogs/gen_reply/link_sources/bilibili.py`
beside the fetch it gates. Its whole justification is that a longer clip cannot finish its
download and upload inside `LINK_MEDIA_TIMEOUT_SECONDS`, so it is a precomputed deadline rather
than a judgement about how much video is worth reading.
"""

from typing import Final

# --------------------------------------------------------------------------------------
# Channel history in one reply
#
# Bounded on three axes because no one of them is the right shape alone, and whichever binds
# first wins.
# --------------------------------------------------------------------------------------

# Discord conversation here is overwhelmingly one-line messages -- measured across 10M logged
# messages, the median is 6 characters and the busiest channel's last 200 come to 1.5k -- so a
# message count alone lets a chatty channel hand the model almost nothing while a channel of long
# posts blows the input up. In practice this has never been the binding axis: the char budget
# alone held history to 107-119 messages at 8000, so this is a backstop.
HISTORY_MESSAGE_LIMIT: Final[int] = 500

# Doubled from 8000 once the media cap landed. Text was never the expensive half -- the whole
# budget is worth ~4k input tokens against 1.3k to 1.9k for a single attachment -- so what made
# widening it unsafe was that more messages meant proportionally more files, which the cap below
# decouples.
HISTORY_CHAR_BUDGET: Final[int] = 16000

# What a history message costs beyond its own text: the rendered form carries an author header,
# and an attachment-only message has empty `content` but still renders a marker standing in for
# it. Without a floor per message a run of image posts would count as free and overshoot.
HISTORY_PER_MESSAGE_OVERHEAD: Final[int] = 40

# How many history attachments ride as real uploaded files. The char budget cannot see this cost
# at all: an attachment-only message spends `HISTORY_PER_MESSAGE_OVERHEAD` there while re-sending
# every one of its files to the model on every single reply, and the Files-API cache in `input.py`
# only saves the re-upload, never the tokens. A media part costs ~1.1k input tokens, measured as
# the median over consecutive replies in one channel, where the history text barely moves between
# the two; the naive slope across all replies reads 2.3k and is measuring the channels that post
# many files rather than the part. Past this many the older attachments degrade to the
# `[attachment: ...]` markers the route already reads, which keeps the model aware a file was
# posted without paying to re-read it.
#
# Ten was a judgement call when it landed and 249 post-deploy replies say to keep it, because what
# a reply WOULD send uncapped is bimodal rather than graded: just over half want five parts or
# fewer and never reach the cap, while the p90 is 92 and the worst 128. There is no bulge just
# above ten to buy, so raising the cap to 20 un-caps 21 more of them and leaves 82 still capped,
# and every further step buys less for more. Latency says the same: the answer awaits this render,
# which runs a median 6s when the cap binds against 0s when it does not, and the uploads under it
# share `MEDIA_CONCURRENCY` slots with every other reply in flight, so the cost of a raise is not
# confined to the reply that asked for it.
MAX_HISTORY_MEDIA_PARTS: Final[int] = 10

# How far up the reply chain the Reference Message block is walked. Every link past the first is
# labelled as thread context rather than as the message being answered, so this bounds background
# rather than subject. Measured over 303 logged answer turns, no chain ever rendered more than one
# link: `message.reference.resolved` is populated from Discord's cache, which rarely holds the
# grandparent, so the walk stops on its own long before this does.
MAX_REFERENCE_CHAIN_DEPTH: Final[int] = 3

# --------------------------------------------------------------------------------------
# Memory read into a request
# --------------------------------------------------------------------------------------

# How many users' memories one reply may carry. Deterministic participants are never displaced:
# if they fill or exceed the target the optional selector is skipped, otherwise it can use only
# the remaining slots.
MEMORY_CONTEXT_TARGET_USERS: Final[int] = 8

# Ceiling on one rendered memory document (the merged compartments injected for one reply).
# Measured against the live store on 2026-08-23, a rendered document runs to a median of 243
# characters, a p90 of 1.3k and a maximum of 4.5k, so this is a backstop that fires on nobody: it
# exists so a runaway scope degrades to its newest facts plus an explicit notice instead of
# silently bloating every request. Rendering stops at the cap; nothing is deleted, so the cap can
# never fight the next consolidation over content it would immediately write back.
#
# The figures this carried before (~800 B median, 25 KB max) were measuring the fact FILES on disk,
# where the `---` header is routinely longer than the body it describes. Only the body is ever
# rendered, so those numbers overstated what this cap sees by roughly four times. Re-measure the
# rendered form, not the directory.
MEMORY_INJECTION_MAX_CHARS: Final[int] = 30_000

# Not a budget of its own: nothing binds on it and no request is ever shortened by it. It is the
# operator's warning line for the cap above, logged after a write, and it lives here only because
# reading it anywhere other than beside that cap would say nothing.
MEMORY_INJECTION_WARN_CHARS: Final[int] = 24_000

# --------------------------------------------------------------------------------------
# Memory extraction and consolidation requests
# --------------------------------------------------------------------------------------

# Phase-1 transcript truncation (keeps head and tail, drops the middle). Large on purpose: the
# reply history window should reach extraction whole, and the memory models accept 1M-token
# inputs.
MEMORY_TRANSCRIPT_MAX_CHARS: Final[int] = 100_000

# Cap for one memory note the answer model wrote inline, as it reaches the evaluator and, for a
# forget, the consolidation prompts. A note is asked for as one sentence, so this only bounds a
# model that ignored that; it is per note rather than per batch because `MAX_MEMORY_NOTES` already
# bounds the count, and because a batch cap would let one runaway note starve the rest.
MEMORY_NOTE_MAX_CHARS: Final[int] = 400

# Cap for the bot's own reply inside that transcript. The reply is secondary evidence and is
# appended last, so without this cap a long (e.g. SUMMARY) reply fills the entire kept tail and
# the middle-truncation drops the current user message right before it.
MEMORY_REPLY_MAX_CHARS: Final[int] = 8_000

# Tail window of the detail file fed to consolidation as low-trust provenance. Effectively the
# whole evidence log for any realistic user: this bot injects memory exactly once per reply with
# no on-demand retrieval, so the stored facts must be distilled from the full evidence base in the
# background. The bound only keeps a pathological log inside the consolidation input window
# (~500k zh-TW chars stays well under the 1M-token window with the stored facts and raw batch on
# top). `DETAIL_FILE_MAX_BYTES` is sized against this and must stay above it, which
# `tests/test_context_budgets.py` pins.
MEMORY_DETAIL_CONTEXT_MAX_CHARS: Final[int] = 500_000

# --------------------------------------------------------------------------------------
# Linked posts read for one reply
# --------------------------------------------------------------------------------------

# Cap on media parts injected for a whole Threads block, shared by the linked post and the post it
# quotes (`_media_plan` splits it, target first). Mirrors the parse_threads cog's 10-embed ceiling
# so a huge carousel cannot bloat the answer input -- and, now that each part costs a fetch plus an
# upload, cannot blow the media budget either.
MAX_THREADS_MEDIA_PARTS: Final[int] = 10

# Cap on posts rendered from a Threads reply chain, mirroring the cog's deep-chain trim: a linked
# reply deep in a long thread would otherwise render every ancestor's text and bloat or overflow
# the answer input. The chain is ordered oldest-first, so the tail keeps the target plus its
# nearest ancestors.
MAX_THREADS_POSTS: Final[int] = 6

# Cap on the comments rendered below the target, counted across every branch. Kept on its own axis
# rather than sharing `MAX_THREADS_POSTS`: the chain is the context leading UP to the linked post,
# the comments are the discussion under it, and one should never squeeze the other out. Sized to
# roughly what one page ships (the sampled pages carry 0-46 comments, median 3), so it is a
# backstop rather than a policy; `_select_replies` decides what a trim actually drops.
MAX_THREADS_REPLIES: Final[int] = 30

# Cap on images ingested from a Douyin photo post. Each costs a download plus an upload, and a
# model reading eight frames of a gallery already has the gist; the cog's Discord-side cap is
# separate and larger, because attaching a file is far cheaper than tokenizing it.
MAX_DOUYIN_INGEST_IMAGES: Final[int] = 8

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

# How much of a feedback reporter's own words ride into the close-notice translation request.
# What they are for is telling the model which language to translate into: enough to read the
# language off, short enough not to crowd out the text being translated, and cut from the front
# because that is where people state the problem.
CLOSE_NOTICE_LANGUAGE_SAMPLE_CHARS: Final[int] = 200
