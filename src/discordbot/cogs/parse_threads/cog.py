"""Cog that expands Threads post URLs into Discord embeds and media files.

Expansion is skipped when the message is addressed to the bot (a DM, or an explicit
mention): `gen_reply` self-parses the linked post and answers about it, so expanding as
well would download the same media twice and post an embed nobody asked for. The two
paths are mutually exclusive, and `is_addressed_to_bot` is the single predicate deciding
which one runs. Keeping it to one download is the reason the rule exists: the media is the
slow half, and on the path the reply actually takes, handing the model a URL instead saves
nothing — the proxy fetches it and inlines it rather than forwarding it.

That predicate is deliberately coarser than `gen_reply`'s own guards, so a few addressed
messages get neither treatment: one typed inside an active research thread (the reply
pipeline skips those), and one the router sends to IMAGE / VIDEO (those routes discard
the link context). Both are rare enough to accept rather than couple the cogs together.

The one-download rule is per message, not per link: `gen_reply` also reads a Threads link the
user only replied to (#377), so a post expanded here is read again when someone replies to
that message and mentions the bot. That is a separate ask for the comments, which this cog
has no embed budget to show. It cannot be triggered by replying to the expansion itself.
"""

from typing import TYPE_CHECKING
import asyncio
from pathlib import Path
import tempfile

import logfire
from nextcord import Color, Embed, Message, NotFound, Forbidden, HTTPException, AllowedMentions
from pydantic import Field, BaseModel, ConfigDict
from nextcord.ext import commands

from discordbot.utils.threads import THREADS_URL_RE, ThreadsOutput, ThreadsDownloader
from discordbot.utils.mentions import is_addressed_to_bot
from discordbot.utils.reactions import update_reaction
from discordbot.typings.timeouts import DOWNLOAD_TIMEOUT_SECONDS
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.media_delivery import (
    MEDIA_ENVELOPE_MARGIN,
    MediaItem,
    upload_limit_for,
    build_media_delivery_planner,
)

if TYPE_CHECKING:
    from asyncio import Task
    from contextlib import AbstractContextManager

    from nextcord.types.embed import Embed as EmbedData

    from discordbot.utils.threads import ThreadsConversation

# Stripe for the post a quote post quotes. Deliberately off the greyscale chain gradient
# (`_gradient_color`, which spans 0x40-0xC0 and reserves pure black for "no stripe"): a quoted
# post is not a layer of the thread, so a shade from that ramp would read as one.
_QUOTED_POST_COLOR = Color.blurple()

# Appended to the target's own embed when it quotes a post Threads no longer serves. It rides on
# the target rather than taking an embed slot of its own: there is no content to show, and the
# tombstone Threads sends carries neither a username nor a shortcode, so there is not even a
# permalink to offer. Never worded as a deletion — the payload says "unavailable", nothing more.
_QUOTED_UNAVAILABLE_HINT = "\n\n🔗 *引用的貼文目前無法瀏覽(可能已刪除或改為私人)*"

_MAX_EMBEDS_PER_MESSAGE = 10
_EMBED_DESCRIPTION_LIMIT = 4096
_EMBED_TOTAL_LENGTH_LIMIT = 6000
_MESSAGE_CONTENT_LIMIT = 2000

# The chain this cog walks has no depth cap of its own (`MAX_THREADS_POSTS` is gen_reply's), so
# the permalink fallback needs one, or a deep enough thread paginates into arbitrarily many
# follow-up replies under a single link. Three pages carry roughly sixty permalinks at the
# measured line length, past any chain seen live; whatever is left is stated as a count.
_MAX_OMITTED_NOTICE_PAGES = 3
_OMITTED_NOTICE_HEADER = "-# 因 Discord embed 限制, 有 {count} 篇貼文未展開. 可從以下原始連結查看:"
_OMITTED_NOTICE_REMAINDER = "-# 其中 {count} 篇的連結因訊息長度限制未列出."


class _EmbedPlan(BaseModel):
    """Rendered embeds plus posts that need a permalink fallback."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    embeds: list[Embed] = Field(
        ..., description="Embeds selected within Discord's message limits.", examples=[[]]
    )
    omitted_posts: list[ThreadsOutput] = Field(
        ..., description="Posts represented by permalink fallbacks.", examples=[[]]
    )


def _discard_scratch_files_when_the_worker_stops(
    *,
    parse_cm: "AbstractContextManager[ThreadsConversation]",
    enter_task: "Task[object]",
    url: str,
    message_id: int,
) -> None:
    """Runs the parse's `__exit__` once a timed-out worker finally finishes.

    The listener has already given up by the time this is called, but the worker is still
    downloading into the shared system temp dir — `asyncio.to_thread` cannot be cancelled — and
    the `finally` that normally deletes those files sits on a path the timeout skipped. Waiting
    for the worker here would just move the stall back onto the listener, so the exit is deferred
    onto the worker's own completion and nothing awaits it. A worker that ends by raising wrote
    nothing to clean up.
    """

    def _on_worker_done(task: "Task[object]") -> None:
        if task.cancelled() or task.exception() is not None:
            return
        cleanup = asyncio.create_task(asyncio.to_thread(parse_cm.__exit__, None, None, None))
        cleanup.add_done_callback(
            lambda done: (
                logfire.error(
                    "Could not clean up the Threads scratch files of a timed-out parse",
                    url=url,
                    message_id=message_id,
                    _exc_info=done.exception(),
                )
                if not done.cancelled() and done.exception() is not None
                else None
            )
        )

    enter_task.add_done_callback(_on_worker_done)


def _utf16_length(value: str) -> int:
    """Counts UTF-16 code units, the conservative interpretation of Discord characters."""
    return sum(2 if ord(character) > 0xFFFF else 1 for character in value)


def _embed_text_length(embed: Embed) -> int:
    """Counts every text-bearing embed field against Discord's message-wide limit."""
    payload: EmbedData = embed.to_dict()
    text_parts = [
        value for value in (payload.get("title"), payload.get("description")) if value is not None
    ]
    if footer := payload.get("footer"):
        text_parts.append(footer["text"])
    if (author := payload.get("author")) and (name := author.get("name")):
        text_parts.append(name)
    for field in payload.get("fields", []):
        text_parts.extend((field["name"], field["value"]))

    return sum(_utf16_length(value=value) for value in text_parts)


def _omitted_post_line(post: ThreadsOutput) -> str:
    """Renders one omitted post as a permalink line."""
    if post.url:
        return f"- @{post.author_name}: <{post.url}>"
    return f"- @{post.author_name}: 原始連結無法取得"


def _closed_with_remainder(page: str, *, unlisted: int) -> str:
    """Closes the last page with the count of permalinks it could not carry.

    Trailing lines are handed back to that count until the closing line fits, so the notice
    reports what it dropped instead of growing another reply to hold it.
    """
    lines = page.split("\n")
    while True:
        closed = "\n".join([*lines, _OMITTED_NOTICE_REMAINDER.format(count=unlisted)])
        if len(lines) == 1 or _utf16_length(value=closed) <= _MESSAGE_CONTENT_LIMIT:
            return closed
        lines.pop()
        unlisted += 1


def _omitted_post_notice_pages(posts: list[ThreadsOutput]) -> list[str]:
    """Paginates permalink fallbacks for posts that could not fit in the embeds.

    Never raises and never exceeds `_MAX_OMITTED_NOTICE_PAGES`: a line no page could hold, and
    every line past the cap, is handed to the closing count instead. The notice exists so an
    over-budget chain degrades rather than dropping posts silently, so it must not be able to
    cost the expansion it reports on. The header states the true total either way.
    """
    if not posts:
        return []

    header = _OMITTED_NOTICE_HEADER.format(count=len(posts))
    pages: list[str] = []
    current = header
    listed = 0
    for post in posts:
        line = _omitted_post_line(post=post)
        if _utf16_length(value=f"{header}\n{line}") > _MESSAGE_CONTENT_LIMIT:
            continue
        candidate = f"{current}\n{line}"
        if _utf16_length(value=candidate) > _MESSAGE_CONTENT_LIMIT:
            if len(pages) + 1 >= _MAX_OMITTED_NOTICE_PAGES:
                break
            pages.append(current)
            candidate = f"{header}\n{line}"
        current = candidate
        listed += 1
    if listed < len(posts):
        current = _closed_with_remainder(page=current, unlisted=len(posts) - listed)
    pages.append(current)
    return pages


def _allocate_embed_slots(
    *, posts: list[ThreadsOutput], priority: list[int], reserved: list[int]
) -> list[int]:
    """Allocates Discord's ten embed slots in relevance order."""
    slots = [0] * len(posts)
    budget = _MAX_EMBEDS_PER_MESSAGE
    for index in reserved:
        slots[index] = 1
        budget -= 1

    for index in priority:
        take = min(max(len(posts[index].image_urls) - slots[index], 0), budget)
        slots[index] += take
        budget -= take

    for index in priority:
        if budget <= 0:
            break
        if slots[index] == 0:
            slots[index] = 1
            budget -= 1
    return slots


def _collect_omitted_posts(
    *, trimmed_posts: list[ThreadsOutput], candidate_posts: list[ThreadsOutput], slots: list[int]
) -> list[ThreadsOutput]:
    """Returns unrendered posts once each, excluding an equivalent shown permalink."""
    shown_urls = {
        post.url for index, post in enumerate(candidate_posts) if slots[index] and post.url
    }
    omitted_posts: list[ThreadsOutput] = []
    seen_urls = set(shown_urls)
    unshown_posts = (post for index, post in enumerate(candidate_posts) if not slots[index])
    for post in [*trimmed_posts, *unshown_posts]:
        if post.url and post.url in seen_urls:
            continue
        omitted_posts.append(post)
        if post.url:
            seen_urls.add(post.url)
    return omitted_posts


class ThreadsCogs(commands.Cog):
    """Expands Threads links into Discord embeds and media attachments.

    Attributes:
        bot: The Discord bot instance that owns this cog.
        output_folder: Directory where downloaded Threads media is stored.
        downloader: Downloader used to parse Threads posts and fetch media.
        media_delivery: Planner deciding whether a downloaded video is attached, hosted or dropped.
    """

    def __init__(self, bot: commands.Bot):
        """Initializes the ThreadsCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.output_folder = Path(tempfile.gettempdir())
        self.downloader = ThreadsDownloader(output_folder=str(self.output_folder))
        self.media_delivery = build_media_delivery_planner()

    @staticmethod
    def _gradient_color(index: int, total: int) -> Color:
        """Greyscale gradient — lightest at index=0 (root), darkest at index=total-1 (leaf).

        Both ends stay inside [0x40, 0xC0] so every layer renders a visible stripe; pure black
        (#000000) is reserved for "no stripe" on solo posts.
        """
        if total <= 1:
            return Color.default()
        light = 0xC0
        dark = 0x40
        shade = round(light + (dark - light) * index / (total - 1))
        return Color.from_rgb(r=shade, g=shade, b=shade)

    @staticmethod
    def _build_post_embed(output: ThreadsOutput, color: Color) -> Embed:
        """Builds an embed for a single Threads post."""
        embed = Embed(
            description=output.text, url=output.url, color=color, timestamp=output.taken_at
        )
        if output.author_name:
            embed.set_author(
                name=output.author_name, url=output.url, icon_url=output.author_icon_url
            )
        footer_parts = [
            f"❤️ {output.like_count:,}",
            f"💬 {output.reply_count:,}",
            f"🔁 {output.repost_count:,}",
            f"🔗 {output.quote_count:,}",
            f"↗️ {output.reshare_count:,}",
        ]
        embed.set_footer(text=" | ".join(footer_parts))
        return embed

    def _build_post_embeds(
        self,
        output: ThreadsOutput,
        color: Color,
        image_count: int,
        is_target: bool,
        is_quoted: bool = False,
    ) -> list[Embed]:
        """Builds the embeds for one post, showing `image_count` of its images.

        The main embed carries the post text plus its first shown image; further shown
        images become bare image embeds reusing the post URL so Discord merges them into
        one gallery. `image_count == 0` yields a single text-only context embed.
        """
        main_embed = self._build_post_embed(output=output, color=color)
        embeds = [main_embed]
        # A quoted post sits outside the chain the gradient describes, so it says what it is:
        # without the line it reads as one more post in the thread rather than as the post the
        # linked one is arguing with, and by a different author at that.
        if is_quoted:
            main_embed.description = f"🔗 **被引用的貼文**\n\n{main_embed.description or ''}"
        if image_count > 0:
            main_embed.set_image(url=output.image_urls[0])
            for img_url in output.image_urls[1:image_count]:
                extra = Embed(url=output.url)
                extra.set_image(url=img_url)
                embeds.append(extra)
        # Target videos are downloaded and attached as files; ancestor and quoted-post videos
        # are not, so surface a link hint — otherwise a video-only parent shows as an empty
        # embed, and a quoted clip would look like a quoted post with nothing in it.
        if not is_target and output.video_urls and output.url:
            hint = f"\n\n🎬 [點此觀看影片]({output.url})"
            main_embed.description = (main_embed.description or "") + hint
        # Only for the target, because the target's quote is the only one this expansion shows at
        # all. `_build_output` fills `quoted` / `quoted_unavailable` on every post it parses, so
        # without the gate an ancestor that quotes a DEAD post says so while an ancestor that
        # quotes a live one says nothing — telling the reader about a quote in exactly the case
        # where there is nothing to see.
        if is_target and output.quoted_unavailable:
            main_embed.description = (main_embed.description or "") + _QUOTED_UNAVAILABLE_HINT
        return embeds

    def _select_posts_within_text_limit(
        self,
        *,
        posts: list[ThreadsOutput],
        priority: list[int],
        chain_depth: int,
        quoted_index: int,
    ) -> set[int]:
        """Selects complete posts by relevance until the message-wide text budget is full.

        The target is kept whatever its own text costs; every other post has to fit.
        """
        selected: set[int] = set()
        text_budget = _EMBED_TOTAL_LENGTH_LIMIT
        for index in priority:
            is_quoted = index == quoted_index
            main_embed = self._build_post_embeds(
                output=posts[index],
                color=(
                    _QUOTED_POST_COLOR
                    if is_quoted
                    else self._gradient_color(index=index, total=chain_depth)
                ),
                image_count=0,
                is_target=index == chain_depth - 1,
                is_quoted=is_quoted,
            )[0]
            length = _embed_text_length(embed=main_embed)
            if index != chain_depth - 1 and length > text_budget:
                continue
            selected.add(index)
            text_budget -= length
        return selected

    def _build_embed_plan(self, results: list[ThreadsOutput]) -> _EmbedPlan:
        """Builds embeds and permalink fallbacks for a Threads reply chain.

        Args:
            results: Ordered chain `[root, ..., direct_parent, target]`.
        """
        # Discord caps a single message at 10 embeds, one image each. The posted URL is the
        # target (last item) and owns the message, so an embed for its own words is reserved
        # first, then one for the post it quotes; images then claim what is left in the same
        # order, target, quoted, direct parent, on up the chain. An ancestor that loses the
        # image race still earns a text-only context embed, but only from slots no image needed.
        # A chain deeper than the embed cap can't show every post; keep the target and its
        # nearest ancestors, which are the most relevant context, and link the rest below.
        trimmed_results: list[ThreadsOutput] = []
        if len(results) > _MAX_EMBEDS_PER_MESSAGE:
            trimmed_results = results[:-_MAX_EMBEDS_PER_MESSAGE]
            results = results[-_MAX_EMBEDS_PER_MESSAGE:]
        chain_depth = len(results)
        # The post the target quotes is not a chain member: it is what the target is talking
        # about, by someone who never joined this thread. So it is allocated alongside the chain
        # but emitted after it.
        quoted = results[-1].quoted if results else None
        posts = [*results, *([quoted] if quoted is not None else [])]
        quoted_index = chain_depth if quoted is not None else -1

        # Allocation order: the target, then the post it quotes, then up the chain towards the
        # root. Emission order is different (root first, quoted last) — see the loop below.
        priority = [chain_depth - 1, quoted_index, *reversed(range(chain_depth - 1))]
        priority = [index for index in priority if index >= 0]

        # Discord applies the 6000-character budget to every text-bearing field across the
        # message. Select complete posts in the same relevance order used for embed slots so a
        # distant ancestor can never displace the target, its quoted post or a nearer ancestor.
        # Counting UTF-16 units is conservative for emoji while Discord's documentation leaves
        # the precise meaning of "characters" unspecified.
        selected = self._select_posts_within_text_limit(
            posts=posts, priority=priority, chain_depth=chain_depth, quoted_index=quoted_index
        )
        priority = [index for index in priority if index in selected]

        # The target's embed and the quoted post's are both reserved ahead of every image,
        # including their own. Without it a quote post — a line of commentary over someone
        # else's ten-image carousel, which is the shape that motivated showing the quoted post
        # at all — spends the whole budget on that carousel and drops the commentary that owns
        # the message. The same reservation covers a text-only target under an image-heavy
        # ancestor, which the image-first pass could already starve.
        reserved = [index for index in (chain_depth - 1, quoted_index) if index in selected]
        slots = _allocate_embed_slots(posts=posts, priority=priority, reserved=reserved)

        embeds: list[Embed] = []
        for index, output in enumerate(posts):
            if slots[index] == 0:
                continue
            is_quoted = index == quoted_index
            embeds.extend(
                self._build_post_embeds(
                    output=output,
                    color=(
                        _QUOTED_POST_COLOR
                        if is_quoted
                        else self._gradient_color(index=index, total=chain_depth)
                    ),
                    image_count=min(slots[index], len(output.image_urls)),
                    is_target=index == chain_depth - 1,
                    is_quoted=is_quoted,
                )
            )

        omitted_posts = _collect_omitted_posts(
            trimmed_posts=trimmed_results, candidate_posts=posts, slots=slots
        )
        return _EmbedPlan(embeds=embeds, omitted_posts=omitted_posts)

    def _build_embeds(self, results: list[ThreadsOutput]) -> list[Embed]:
        """Builds embeds for callers that do not need permalink fallback metadata."""
        return self._build_embed_plan(results=results).embeds

    async def _mark_failed(self, *, message: Message, current_emoji: str) -> None:
        """Swaps the progress reaction for the failure cross."""
        await update_reaction(
            message=message,
            bot_user=self.bot.user,
            emoji="<:redcross:1517565100838355016>",
            previous=current_emoji,
        )

    async def _enter_parse(
        self,
        *,
        parse_cm: "AbstractContextManager[ThreadsConversation]",
        url: str,
        message: Message,
        current_emoji: str,
    ) -> "ThreadsConversation | None":
        """Enters the parse under the shared download bound; None means it never produced one.

        `parse()` blocks on an HTTP fetch plus media downloads, so its enter runs off the event
        loop; the reply then runs while the temp files still exist and the caller's matching exit
        cleans them up. Either failure marks the message ❌ and returns None, so the listener has
        one exit to check rather than three.
        """
        # Shielded so the bound releases the listener without cancelling the worker:
        # `asyncio.to_thread` cannot be cancelled anyway, and a task left cancelled here would
        # lose the handle the deferred cleanup needs.
        enter_task = asyncio.create_task(asyncio.to_thread(parse_cm.__enter__))
        try:
            async with asyncio.timeout(delay=DOWNLOAD_TIMEOUT_SECONDS):
                return await asyncio.shield(enter_task)
        except TimeoutError:
            logfire.warn(
                "Threads parse timed out",
                url=url,
                message_id=message.id,
                timeout_seconds=DOWNLOAD_TIMEOUT_SECONDS,
            )
            _discard_scratch_files_when_the_worker_stops(
                parse_cm=parse_cm, enter_task=enter_task, url=url, message_id=message.id
            )
        # Broad on purpose: a fetch failure must not escape into the listener; the ❌ reaction is
        # the user-visible outcome.
        except Exception as error:
            logfire.warn(
                "Threads parse failed",
                url=url,
                message_id=message.id,
                error_type=type(error).__name__,
                _exc_info=error,
            )
        await self._mark_failed(message=message, current_emoji=current_emoji)
        return None

    async def _deliver(
        self,
        *,
        message: Message,
        url: str,
        results: list[ThreadsOutput],
        embed_plan: _EmbedPlan,
        current_emoji: str,
    ) -> None:
        """Plans the target's media, posts the expansion and marks the source done.

        The embed plan arrives already built rather than being rebuilt here, so the
        description-length guard in `on_message` measured the very list that gets sent.

        Only the expansion itself can fail this step. The permalink fallbacks are built and sent
        afterwards, past the ✅ and behind their own guard, because they exist to describe what
        the expansion left out and must never be able to take the expansion down with them.
        """
        target = results[-1]
        embeds = embed_plan.embeds
        # Broad on purpose: the delivery step must never escape into the listener, and its
        # failures split three ways — the source message went away, the bot lacks a permission,
        # or something unexpected lost the expansion.
        try:
            # Videos too big to attach are hosted on the external static server and linked
            # instead of refusing the whole post; the rest attach natively. The planner
            # reserves 1 MiB for the multipart envelope + embeds JSON and pulls the per-guild
            # limit from nextcord (boost tier raises it to 50/100 MiB; a DM has the 10 MiB base).
            items = [
                MediaItem(source=path, filename=path.name)
                for path in target.video_paths
                if path.exists()
            ]
            plan = await self.media_delivery.plan(
                items=items,
                upload_limit=upload_limit_for(guild=message.guild),
                envelope_margin=MEDIA_ENVELOPE_MARGIN,
            )
            if plan.dropped_items:
                # An oversize video that could not be hosted (hosting off / failed) keeps
                # today's whole-post ⚠️ refusal rather than posting a partial chain. Kept simple
                # on purpose: in the near-unreachable hosting-on partial-failure case (one
                # sibling video already moved into the serve dir, another's write fails) this
                # leaves the moved file orphaned at an unposted URL; both serve-dir writes
                # realistically succeed or fail together, so it is not worth a cleanup branch.
                logfire.warn(
                    "Threads videos could not be hosted; refusing the whole post",
                    url=url,
                    message_id=message.id,
                    dropped=len(plan.dropped_items),
                )
                await update_reaction(
                    message=message, bot_user=self.bot.user, emoji="⚠️", previous=current_emoji
                )
                return

            files = [item.to_file() for item in plan.native]

            try:
                await message.edit(suppress=True)
            # Broad on purpose: hiding Discord's own preview is cosmetic and must not abort the
            # expansion. A persistent Forbidden means the guild lacks Manage Messages.
            except Exception as error:
                logfire.warn(
                    "Could not suppress the source message embed",
                    message_id=message.id,
                    guild_id=message.guild.id if message.guild else None,
                    error_type=type(error).__name__,
                    _exc_info=error,
                )

            await message.reply(
                content="\n".join(plan.hosted_urls) if plan.hosted_urls else None,
                embeds=embeds,
                mention_author=False,
                allowed_mentions=AllowedMentions.none(),
                **embed_spacer_payload(
                    embeds=embeds, is_edit=False, target=message, extra_files=files
                ),
            )
        except Exception as error:
            # A reply to a deleted source comes back as HTTP 50035, not only as NotFound.
            gone = isinstance(error, NotFound) or (
                isinstance(error, HTTPException) and error.code == 50035
            )
            if gone:
                logfire.info(
                    "Threads expansion target is gone",
                    url=url,
                    message_id=message.id,
                    channel_id=message.channel.id,
                )
            elif isinstance(error, Forbidden):
                logfire.warn(
                    "Missing permission to post the Threads expansion",
                    url=url,
                    message_id=message.id,
                    channel_id=message.channel.id,
                    error_type=type(error).__name__,
                    _exc_info=error,
                )
            else:
                logfire.error(
                    "Failed to send Threads expansion",
                    url=url,
                    message_id=message.id,
                    channel_id=message.channel.id,
                    error_type=type(error).__name__,
                    _exc_info=error,
                )
            await self._mark_failed(message=message, current_emoji=current_emoji)
            return

        # The expansion is on screen, so it is marked done before the permalink fallbacks are
        # posted. Those only report what the embeds could not carry, and nothing that happens
        # to them may reduce what the user already has or relabel it as a failure.
        await update_reaction(
            message=message,
            bot_user=self.bot.user,
            emoji="<:greencheck:1517565102424068226>",
            previous=current_emoji,
        )
        await self._post_omitted_notices(message=message, url=url, posts=embed_plan.omitted_posts)

    async def _post_omitted_notices(
        self, *, message: Message, url: str, posts: list[ThreadsOutput]
    ) -> None:
        """Builds and posts the permalink fallbacks as follow-up replies, best effort.

        The two steps are guarded separately so the log names the one that failed: building is
        pure and should not be able to fail at all, while a send meets a channel that may have
        changed under it. Both are broad on purpose — the expansion is already delivered and
        marked done, so a failure here costs only the permalink list and must never travel back
        to the delivery's failure path.
        """
        try:
            notices = _omitted_post_notice_pages(posts=posts)
        except Exception as error:
            logfire.warn(
                "Could not build the Threads permalink fallbacks",
                url=url,
                message_id=message.id,
                error_type=type(error).__name__,
                _exc_info=error,
            )
            return
        try:
            for notice in notices:
                await message.reply(
                    content=notice, mention_author=False, allowed_mentions=AllowedMentions.none()
                )
        except Exception as error:
            logfire.warn(
                "Could not post the Threads permalink fallbacks",
                url=url,
                message_id=message.id,
                error_type=type(error).__name__,
                _exc_info=error,
            )

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Listens for messages and parses Threads links.

        Args:
            message: The message that was sent.
        """
        if message.author.bot:
            return

        match = THREADS_URL_RE.search(string=message.content)
        if not match:
            return

        # A link addressed to the bot is gen_reply's to answer about, not ours to expand; see
        # the module docstring for why the two never both fire. Checked after the URL match so
        # the common no-link message costs one regex, not two.
        if is_addressed_to_bot(message=message, bot_user=self.bot.user):
            return

        url = match.group(0)
        # Persistent marker (added directly, not through the status chain, which replaces its own
        # reaction) saying a Threads post was read. `gen_reply` adds the same one on the path it
        # takes instead of this one, so every read is marked the same way whichever cog did it.
        await update_reaction(
            message=message, bot_user=self.bot.user, emoji="<:threads:1535657820668559380>"
        )
        current_emoji = await update_reaction(message=message, bot_user=self.bot.user, emoji="🔗")

        parse_cm = self.downloader.parse(url=url)
        conversation = await self._enter_parse(
            parse_cm=parse_cm, url=url, message=message, current_emoji=current_emoji
        )
        if conversation is None:
            return

        try:
            try:
                # The expansion shows the reply chain only; the comments the parse also carries
                # are gen_reply's to read, and there is no embed budget left for them here.
                results = conversation.chain
                if not results:
                    logfire.info(
                        "Threads parse returned no post; treating as unavailable", url=url
                    )
                    await update_reaction(
                        message=message, bot_user=self.bot.user, emoji="⚠️", previous=current_emoji
                    )
                    return

                target = results[-1]
                if target.quoted_unavailable:
                    # A routine user-driven outcome (a removed remote post), so info, not warn.
                    # Logged because it is common — measured at 15 of 96 live quote relations —
                    # and otherwise invisible in `data/logs`.
                    logfire.info("A Threads post quotes a post Threads no longer serves", url=url)
                embed_plan = self._build_embed_plan(results=results)
                embeds = embed_plan.embeds
                # Measured on the RENDERED descriptions rather than on `target.text`: the quoted
                # post's marker prefix, an ancestor's video hint and the unavailable hint are all
                # appended by `_build_post_embeds` AFTER any check on the raw body, so a text
                # sitting just under the limit crossed it and turned a ⚠️ skip into a Discord 400
                # and a ❌. A body past the limit cannot be rescued by hosting, so it stays the ⚠️
                # refusal. (Image count is not guarded: _build_embed_plan caps the message at 10
                # embeds and shows as many images as fit.)
                longest_text = max(
                    (_utf16_length(value=embed.description or "") for embed in embeds), default=0
                )
                if longest_text > _EMBED_DESCRIPTION_LIMIT:
                    logfire.info(
                        "Threads post exceeds the embed description limit; skipping expansion",
                        url=url,
                        text_length=longest_text,
                    )
                    await update_reaction(
                        message=message, bot_user=self.bot.user, emoji="⚠️", previous=current_emoji
                    )
                    return

                await self._deliver(
                    message=message,
                    url=url,
                    results=results,
                    embed_plan=embed_plan,
                    current_emoji=current_emoji,
                )
            finally:
                # Guarded here rather than by the outer handler, which would relabel a delivered
                # expansion ❌ over scratch files the user cannot see; swallowing it also stops a
                # failing cleanup from replacing whatever the body raised. Broad on purpose: it
                # is the last step and nothing downstream can act on its failure. Still an error
                # because these files are deleted nowhere else, so a failure leaks the temp dir
                # and names an environment someone must look at (read-only or full filesystem).
                try:
                    await asyncio.to_thread(parse_cm.__exit__, None, None, None)
                except Exception as error:
                    logfire.error(
                        "Could not clean up the Threads scratch files",
                        url=url,
                        message_id=message.id,
                        error_type=type(error).__name__,
                        _exc_info=error,
                    )
        # Broad on purpose: the listener's last line of defence, covering the steps between the
        # parse and the delivery so nothing escapes into the dispatcher.
        except Exception as error:
            logfire.error(
                "Threads expansion failed outside the parse and delivery steps",
                url=url,
                message_id=message.id,
                error_type=type(error).__name__,
                _exc_info=error,
            )
            await self._mark_failed(message=message, current_emoji=current_emoji)


def setup(bot: commands.Bot) -> None:
    """Adds the ThreadsCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(ThreadsCogs(bot), override=True)
