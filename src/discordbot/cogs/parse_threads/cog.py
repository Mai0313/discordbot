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

None of the above reaches `/clean_threads_url`, the cog's slash command, which reads no post
and downloads nothing: it resolves a share link to the post's own URL and answers the caller
alone. Nothing is expanded, so there is no second read for the mention rule to prevent.

Threads, Facebook, Instagram and Douyin are one feature four times over, and what makes them
one is `utils/expansion_placeholder.py`: the reply slot claimed before the read starts, the
five reactions and what each of them means, the restart sweep, and the rule that a failed
expansion says nothing in the channel at all. Read that module before changing anything here
that a reader would notice, and `tests/test_expansion_contract.py` before adding a fifth
source. What stays per platform is the card: this one walks a whole conversation and has ten
embed slots to allocate over it.
"""

from typing import TYPE_CHECKING
import asyncio

import logfire
import nextcord
from nextcord import Color, Embed, Locale, Message, NotFound, Interaction, SlashOption
from nextcord.ext import commands

from discordbot.utils.threads import THREADS_URL_RE, ThreadsOutput, ThreadsDownloader
from discordbot.typings.emojis import THREADS_EMOJI
from discordbot.utils.mentions import is_addressed_to_bot
from discordbot.utils.reactions import update_reaction
from discordbot.typings.commands import INSTALL_CONTEXTS, INTERACTION_CONTEXTS
from discordbot.typings.timeouts import THREADS_EXPAND_TIMEOUT_SECONDS
from discordbot.utils.scratch_dir import scratch_directory
from discordbot.utils.media_delivery import (
    MEDIA_ENVELOPE_MARGIN,
    MediaItem,
    upload_limit_for,
    build_media_delivery_planner,
)
from discordbot.utils.expansion_placeholder import (
    EXPANSION_DONE_EMOJI,
    EXPANSION_FAILED_EMOJI,
    EXPANSION_WORKING_EMOJI,
    EXPANSION_UNREADABLE_EMOJI,
    EXPANSION_RETRY_LATER_EMOJI,
    ExpansionPlaceholder,
    send_expansion_placeholder,
    resume_expansion_placeholders,
    report_expansion_delivery_failure,
)

if TYPE_CHECKING:
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

# Held back from the message-wide budget for the remainder notes, which are appended to the
# target's footer after selection has already measured it: what a post gave up is not known
# until every slot is spent. Sixty-four UTF-16 units is well past the longest pair of notes
# (emoji count double), and `parse_facebook` holds back `_BUDGET_SLACK` for the same reason.
_REMAINDER_RESERVE = 64

_PLACEHOLDER_TEXT = "-# 正在讀取 Threads 貼文⋯"
# Owns this cog's rows in the pending-expansion table. Keyed as in `LINK_SOURCE_EMOJIS`, which
# `tests/test_link_source_emojis.py` pins, so the four cogs and the reply path name a platform
# the same way rather than each inventing a spelling.
_SOURCE = "threads"


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


def _remainder_notes(*, omitted_posts: int, omitted_images: int) -> list[str]:
    """States what the ten embed slots could not carry, the way the other cogs state it.

    An over-budget expansion shows the ten most relevant slots and says how much it left
    behind, rather than growing follow-up replies to hold the rest: those land wherever the
    channel has got to, several messages under the card describing them. `parse_facebook` and
    `parse_instagram` say the same thing about their own image caps in the same place, so a
    reader learns what is missing from the card itself whichever platform it came from.
    """
    notes = []
    if omitted_images > 0:
        notes.append(f"🖼️ 另有 {omitted_images} 張")
    if omitted_posts > 0:
        notes.append(f"📝 另有 {omitted_posts} 篇未展開")
    return notes


class ThreadsCogs(commands.Cog):
    """Expands Threads links into Discord embeds and media attachments.

    Attributes:
        bot: The Discord bot instance that owns this cog.
        downloader_factory: Builds the per-invocation downloader, one per scratch directory;
            the seam a test replaces to keep an expansion off the network.
        media_delivery: Planner deciding whether a downloaded video is attached, hosted or dropped.
    """

    def __init__(self, bot: commands.Bot):
        """Initializes the ThreadsCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.downloader_factory = ThreadsDownloader
        self.media_delivery = build_media_delivery_planner()
        self._resume_started = False

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Runs again the expansions a restart interrupted (once per process).

        `on_ready` fires on every gateway reconnect, so the flag guards it to one sweep.
        """
        if self._resume_started:
            return
        self._resume_started = True
        await resume_expansion_placeholders(bot=self.bot, source=_SOURCE, expand=self._expand)

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
            f"💬 {output.comment_count:,}",
            f"🔁 {output.repost_count:,}",
            f"🔗 {output.quote_count:,}",
            f"↗️ {output.share_count:,}",
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
        text_budget = _EMBED_TOTAL_LENGTH_LIMIT - _REMAINDER_RESERVE
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

    def _build_embeds(self, results: list[ThreadsOutput]) -> list[Embed]:
        """Builds the whole expansion for a Threads reply chain, ten embeds at most.

        Args:
            results: Ordered chain `[root, ..., direct_parent, target]`.
        """
        # Discord caps a single message at 10 embeds, one image each. The posted URL is the
        # target (last item) and owns the message, so an embed for its own words is reserved
        # first, then one for the post it quotes; images then claim what is left in the same
        # order, target, quoted, direct parent, on up the chain. An ancestor that loses the
        # image race still earns a text-only context embed, but only from slots no image needed.
        # A chain deeper than the embed cap can't show every post; keep the target and its
        # nearest ancestors, which are the most relevant context, and count the rest in the
        # target's footer.
        trimmed = 0
        if len(results) > _MAX_EMBEDS_PER_MESSAGE:
            trimmed = len(results) - _MAX_EMBEDS_PER_MESSAGE
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
        target_index = chain_depth - 1
        target_embed: Embed | None = None
        for index, output in enumerate(posts):
            if slots[index] == 0:
                continue
            is_quoted = index == quoted_index
            built = self._build_post_embeds(
                output=output,
                color=(
                    _QUOTED_POST_COLOR
                    if is_quoted
                    else self._gradient_color(index=index, total=chain_depth)
                ),
                image_count=min(slots[index], len(output.image_urls)),
                is_target=index == target_index,
                is_quoted=is_quoted,
            )
            if index == target_index:
                target_embed = built[0]
            embeds.extend(built)

        # Counted after allocation rather than during it: what a post gave up is only known
        # once every slot is spent. The count rides the target's footer because the target is
        # the post that was linked, so it is the card a reader is looking at. Only its own
        # images are counted — an ancestor's are context the expansion never promised.
        if target_embed is not None:
            notes = _remainder_notes(
                omitted_posts=trimmed + sum(1 for slot in slots if slot == 0),
                omitted_images=len(posts[target_index].image_urls) - slots[target_index],
            )
            if notes:
                carried = [part for part in (target_embed.footer.text, *notes) if part]
                target_embed.set_footer(text=" | ".join(carried))
        return embeds

    async def _mark_failed(self, *, message: Message, current_emoji: str | None) -> None:
        """Paints the failure cross, naming the platform when nothing else on the message does.

        `current_emoji` is None only when claiming the reply slot failed, before either mark
        went on. The cross alone cannot say WHICH link died, and on a message carrying two
        that is the whole of what someone needs, so the platform marker goes on first.
        """
        if current_emoji is None:
            await update_reaction(message=message, bot_user=self.bot.user, emoji=THREADS_EMOJI)
        await update_reaction(
            message=message,
            bot_user=self.bot.user,
            emoji=EXPANSION_FAILED_EMOJI,
            previous=current_emoji,
        )

    async def _deliver(  # noqa: PLR0913 -- the walk, its rendered embeds, and both channel handles
        self,
        *,
        message: Message,
        url: str,
        results: list[ThreadsOutput],
        embeds: list[Embed],
        current_emoji: str,
        placeholder: ExpansionPlaceholder,
    ) -> None:
        """Plans the target's media, delivers the expansion and marks the source done.

        The embeds arrive already built rather than being rebuilt here, so the
        description-length guard in `_expand_conversation` measured the very list that gets sent.
        """
        target = results[-1]
        # Broad on purpose: the delivery step must never escape into the listener, and its
        # failures split three ways — the placeholder went away, the bot lacks a permission,
        # or something unexpected lost the expansion.
        try:
            # Videos too big to attach are hosted on the external static server and linked
            # instead of refusing the whole post; the rest attach natively. The planner
            # reserves 1 MiB for the multipart envelope + embeds JSON and pulls the per-guild
            # limit from nextcord (boost tier raises it to 50/100 MiB; a DM has the 20 MiB base).
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
                    message=message,
                    bot_user=self.bot.user,
                    emoji=EXPANSION_UNREADABLE_EMOJI,
                    previous=current_emoji,
                )
                return

            files = [item.to_file() for item in plan.native]

            try:
                await message.edit(suppress=True)
            # Deleting the link while the post was being read is a withdrawal: before the
            # placeholder existed the late reply was simply refused, and this keeps that,
            # since a reply outlives the message it answers.
            except NotFound:
                logfire.info(
                    "Threads expansion target is gone",
                    url=url,
                    message_id=message.id,
                    channel_id=message.channel.id,
                )
                return
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

            await placeholder.deliver(
                content="\n".join(plan.hosted_urls) if plan.hosted_urls else None,
                embeds=embeds,
                files=files,
            )
        except Exception as error:
            report_expansion_delivery_failure(
                error=error,
                platform="Threads",
                url=url,
                message_id=message.id,
                channel_id=message.channel.id,
            )
            await self._mark_failed(message=message, current_emoji=current_emoji)
            return

        await update_reaction(
            message=message,
            bot_user=self.bot.user,
            emoji=EXPANSION_DONE_EMOJI,
            previous=current_emoji,
        )

    @nextcord.slash_command(
        name="clean_threads_url",
        description="Turn a Threads share link into the post's own URL.",
        name_localizations={Locale.zh_TW: "清理串文連結", Locale.ja: "スレッズリンク整理"},
        description_localizations={
            Locale.zh_TW: "把 Threads 分享連結還原成貼文本身的網址",
            Locale.ja: "Threads の共有リンクを投稿本来の URL に戻します。",
        },
        nsfw=False,
        integration_types=INSTALL_CONTEXTS,
        contexts=INTERACTION_CONTEXTS,
    )
    async def clean_threads_url(
        self,
        interaction: Interaction[commands.Bot],
        url: str = SlashOption(
            description="Threads post link, or the share text containing it",
            description_localizations={
                Locale.zh_TW: "Threads 貼文連結,或含有連結的分享文字",
                Locale.ja: "Threads の投稿リンク、またはそれを含む共有テキスト",
            },
            required=True,
        ),
    ) -> None:
        """Answers with the canonical URL of the Threads post a link names.

        Ephemeral throughout: what the caller pasted names whoever shared it, so the answer is
        for them to copy rather than something the channel needs to see. A share link costs the
        one fetch its redirect needs, read no further than the URL that came back; no media is
        fetched and no post is parsed, which is why a post nobody can read still gets an answer.

        Args:
            interaction: The interaction that triggered the command.
            url: The Threads link to clean, or the share text carrying it.
        """
        await interaction.response.defer(ephemeral=True)
        # The same regex the listener matches on, run over the whole option so pasting a share
        # blob works here exactly as pasting it into the channel does.
        match = THREADS_URL_RE.search(string=url)
        if not match:
            await interaction.followup.send(
                content="這裡面沒有 Threads 貼文連結。", ephemeral=True
            )
            return

        # No output folder because nothing is ever written: `resolve_clean_url` reads the
        # redirect and stops there, so the field the downloader keeps for media never gets read.
        downloader = self.downloader_factory(output_folder="")
        try:
            clean_url = await asyncio.to_thread(downloader.resolve_clean_url, url=match.group(0))
        except RuntimeError as error:
            logfire.warn(
                "Could not resolve a Threads share link",
                url=match.group(0),
                error_type=type(error).__name__,
                _exc_info=error,
            )
            # Not worded as a temporary failure: `_fetch_page` raises the same way for a network
            # error as for a landing page Threads refused, and only the first is worth retrying.
            await interaction.followup.send(content="這個連結現在拿不到。", ephemeral=True)
            return

        if not clean_url:
            await interaction.followup.send(
                content="這個分享連結沒有指向任何貼文。", ephemeral=True
            )
            return
        await interaction.followup.send(content=clean_url, ephemeral=True)

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
        # Nothing is on the message yet, and the outer handler below is reachable before
        # anything is: claiming the reply slot is itself a step that can fail. Left unset
        # rather than pre-filled so a failure mark removes a reaction only when one is there.
        current_emoji: str | None = None
        try:
            placeholder = await send_expansion_placeholder(
                message=message, text=_PLACEHOLDER_TEXT, source=_SOURCE, url=url
            )
            if placeholder is None:
                # A channel that refused the placeholder will refuse the card too, so the
                # read is never started.
                await self._mark_failed(message=message, current_emoji=current_emoji)
                return
            # Persistent marker (added directly, not through the status chain, which replaces
            # its own reaction) saying a Threads post was read, and the working ring under it.
            # Both go on AFTER the reply slot is claimed: they share one per-channel rate-limit
            # bucket that a message send does not, so claiming first is what stops the card
            # queueing behind them. `gen_reply` adds the same marker on the path it takes
            # instead of this one, so every read is marked the same way whichever cog did it.
            await update_reaction(message=message, bot_user=self.bot.user, emoji=THREADS_EMOJI)
            current_emoji = await update_reaction(
                message=message, bot_user=self.bot.user, emoji=EXPANSION_WORKING_EMOJI
            )
            try:
                await self._expand(
                    message=message, url=url, current_emoji=current_emoji, placeholder=placeholder
                )
            finally:
                # Every failure and refusal below returns rather than raising, so this one
                # line covers all of them: an expansion that delivered nothing leaves nothing
                # behind. Once delivered it is a no-op, which is what lets the permalink
                # notices run past it.
                await placeholder.discard()
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

    async def _expand(
        self, *, message: Message, url: str, current_emoji: str, placeholder: ExpansionPlaceholder
    ) -> None:
        """Parses the post inside a scratch directory and hands the walk to `_expand_conversation`.

        Owns the parse lifecycle alone: opening the walk under the wall-clock bound, reporting a
        fetch failure, and closing the generator afterwards. What the parsed conversation is worth
        showing is decided a level down, so the delivery decision does not sit under this cleanup.
        """
        # A private directory per invocation, so two people expanding the same post cannot
        # write each other's paths, and so the bound below has something to abandon into:
        # the `requests` calls under `parse` are per-read only, so a slow-drip CDN can stream
        # for as long as it likes, and `asyncio.to_thread` cannot cancel the walk it holds.
        # Removing the directory is the only stop signal that reaches it, and only once it
        # tries to write — a text-only post runs its fetches out either way, costing the
        # page timeout rather than the stall this bound exists for.
        with scratch_directory(prefix="parse-threads-") as download_dir:
            downloader = self.downloader_factory(output_folder=download_dir)
            # parse() blocks on HTTP fetch + media downloads, so run its enter
            # off the event loop; the reply runs while the temp files still exist
            # and the matching exit cleans them up afterwards.
            parse_cm = downloader.parse(url=url)
            try:
                async with asyncio.timeout(delay=THREADS_EXPAND_TIMEOUT_SECONDS):
                    conversation = await asyncio.to_thread(parse_cm.__enter__)
            # Broad on purpose: a fetch failure must not escape into the listener; the
            # reaction is the user-visible outcome. A stall never reads as a missing post —
            # under the shared vocabulary it is the retryable mark, since the link is fine.
            except Exception as error:
                # No exit call here: the walk is still driving that generator on its own
                # thread, so throwing into it would be a second driver. Returning removes
                # the directory instead, which both deletes whatever it wrote and fails
                # its next write.
                logfire.warn(
                    "Threads parse failed",
                    url=url,
                    message_id=message.id,
                    error_type=type(error).__name__,
                    _exc_info=error,
                )
                if isinstance(error, TimeoutError):
                    await update_reaction(
                        message=message,
                        bot_user=self.bot.user,
                        emoji=EXPANSION_RETRY_LATER_EMOJI,
                        previous=current_emoji,
                    )
                    return
                await self._mark_failed(message=message, current_emoji=current_emoji)
                return
            try:
                await self._expand_conversation(
                    message=message,
                    url=url,
                    conversation=conversation,
                    current_emoji=current_emoji,
                    placeholder=placeholder,
                )
            finally:
                # Closes the walk's generator and unlinks its media before the directory
                # goes, which is the ordinary path. Guarded here rather than by the outer
                # handler, which would relabel a delivered expansion ❌ over scratch files
                # the user cannot see; swallowing it also stops a failing cleanup from
                # replacing whatever the body raised. Broad on purpose: it is the last step
                # and nothing downstream can act on its failure. A warning rather than an
                # error now that the enclosing scratch directory removes what this missed.
                try:
                    await asyncio.to_thread(parse_cm.__exit__, None, None, None)
                except Exception as error:
                    logfire.warn(
                        "Could not clean up the Threads scratch files",
                        url=url,
                        message_id=message.id,
                        error_type=type(error).__name__,
                        _exc_info=error,
                    )

    async def _expand_conversation(
        self,
        *,
        message: Message,
        url: str,
        conversation: "ThreadsConversation",
        current_emoji: str,
        placeholder: ExpansionPlaceholder,
    ) -> None:
        """Decides whether a parsed conversation can be shown, and delivers it when it can.

        The two ⚠️ outcomes are refusals rather than failures: the post could be read, it just
        cannot be rendered as embeds, so neither takes the ❌ the listener's handler paints.
        """
        # The expansion shows the reply chain only; the comments the parse also carries
        # are gen_reply's to read, and there is no embed budget left for them here.
        results = conversation.chain
        if not results:
            logfire.info("Threads parse returned no post; treating as unavailable", url=url)
            await update_reaction(
                message=message,
                bot_user=self.bot.user,
                emoji=EXPANSION_UNREADABLE_EMOJI,
                previous=current_emoji,
            )
            return

        target = results[-1]
        if target.quoted_unavailable:
            # A routine user-driven outcome (a removed remote post), so info, not warn.
            # Logged because it is common — measured at 15 of 96 live quote relations —
            # and otherwise invisible in `data/logs`.
            logfire.info("A Threads post quotes a post Threads no longer serves", url=url)
        embeds = self._build_embeds(results=results)
        # Measured on the RENDERED descriptions rather than on `target.text`: the quoted
        # post's marker prefix, an ancestor's video hint and the unavailable hint are all
        # appended by `_build_post_embeds` AFTER any check on the raw body, so a text
        # sitting just under the limit crossed it and turned a ⚠️ skip into a Discord 400
        # and a ❌. A body past the limit cannot be rescued by hosting, so it stays the ⚠️
        # refusal. (Image count is not guarded: _build_embeds caps the message at 10
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
                message=message,
                bot_user=self.bot.user,
                emoji=EXPANSION_UNREADABLE_EMOJI,
                previous=current_emoji,
            )
            return

        await self._deliver(
            message=message,
            url=url,
            results=results,
            embeds=embeds,
            current_emoji=current_emoji,
            placeholder=placeholder,
        )


def setup(bot: commands.Bot) -> None:
    """Adds the ThreadsCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(ThreadsCogs(bot), override=True)
