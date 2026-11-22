"""Expands Threads post URLs into Discord embeds/media and provides `/clean_threads_url`."""

import asyncio
import contextlib

import logfire
import nextcord
from nextcord import Color, Embed, Locale, Message, Interaction, SlashOption
from nextcord.ext import commands

from discordbot.typings.commands import INSTALL_CONTEXTS, INTERACTION_CONTEXTS
from discordbot.typings.timeouts import (
    THREADS_PAGE_TIMEOUT_SECONDS,
    THREADS_EXPAND_TIMEOUT_SECONDS,
)
from discordbot.utils.scratch_dir import scratch_directory
from discordbot.utils.expansion_cog import (
    VIDEO_HINT,
    ExpansionCog,
    ExpansionDelivery,
    with_gallery,
    omitted_posts_note,
    omitted_images_note,
)
from discordbot.utils.discord_embeds import (
    DISCORD_ATTACHMENT_LIMIT,
    DISCORD_EMBED_COUNT_LIMIT,
    DISCORD_EMBED_TOTAL_LIMIT,
    DISCORD_EMBED_DESCRIPTION_LIMIT,
    utf16_length,
    embed_text_length,
    target_allows_file_uploads,
)
from discordbot.utils.media_delivery import (
    MEDIA_ENVELOPE_MARGIN,
    MediaItem,
    upload_limit_for,
    build_media_delivery_planner,
)
from discordbot.services.platforms.threads import (
    THREADS_POST_LINK,
    ThreadsURL,
    ThreadsOutput,
    ThreadsDownloader,
    ThreadsConversation,
)

# Off the greyscale chain gradient (`0x40`-`0xC0`) so quoted posts read as out-of-chain.
_QUOTED_POST_COLOR = Color.blurple()

# Appended to the target embed; tombstones lack username/shortcode and deletion-vs-privacy info.
_QUOTED_UNAVAILABLE_HINT = "\n\n🔗 *引用的貼文目前無法瀏覽(可能已刪除或改為私人)*"

# UTF-16 budget reserved for remainder footer notes appended after slot allocation.
_REMAINDER_RESERVE = 64


def _allocate_embed_slots(
    posts: list[ThreadsOutput], priority: list[int], reserved: list[int]
) -> list[int]:
    """Allocates Discord's ten embed slots in relevance order."""
    slots = [0] * len(posts)
    budget = DISCORD_EMBED_COUNT_LIMIT
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


def _remainder_notes(omitted_posts: int, omitted_images: int) -> list[str]:
    """Returns footer notes counting posts and target images that did not fit the ten embed slots."""
    notes = []
    if omitted_images > 0:
        notes.append(omitted_images_note(count=omitted_images))
    if omitted_posts > 0:
        notes.append(omitted_posts_note(count=omitted_posts))
    return notes


class ThreadsCogs(ExpansionCog[ThreadsConversation]):
    """Expands Threads links into Discord embeds and media attachments.

    Attributes:
        downloader_factory: Per-invocation downloader constructor, one per scratch directory.
        media_delivery: Planner deciding whether a downloaded video is attached, hosted or dropped.
    """

    SOURCE = "threads"
    PLATFORM = "Threads"
    POST_LINK = THREADS_POST_LINK
    PLACEHOLDER_TEXT = "-# 正在讀取 Threads 貼文⋯"

    def __init__(self, bot: commands.Bot):
        """Initializes the ThreadsCogs instance.

        Args:
            bot (commands.Bot): The Discord bot instance.
        """
        super().__init__(bot=bot)
        self.downloader_factory = ThreadsDownloader
        self.media_delivery = build_media_delivery_planner()

    async def read(
        self, message: Message, url: str, stack: contextlib.AsyncExitStack
    ) -> ThreadsConversation:
        """Returns the parsed conversation with the target's media downloaded into a scratch directory.

        Uses a per-invocation scratch directory on `stack`; removing it on timeout aborts a
        stalled `asyncio.to_thread` download on its next write.

        Args:
            message (Message): Unused; required by the `ExpansionCog` hook signature.
            url (str): The post URL to read.
            stack (contextlib.AsyncExitStack): Holds the scratch directory until delivery finishes.
        """
        del message
        download_dir = stack.enter_context(scratch_directory(prefix="parse-threads-"))
        downloader = self.downloader_factory(output_folder=download_dir)
        async with asyncio.timeout(delay=THREADS_EXPAND_TIMEOUT_SECONDS):
            conversation = await asyncio.to_thread(downloader.parse, url=url)
        return conversation

    async def build_delivery(
        self, message: Message, url: str, parsed: ThreadsConversation
    ) -> ExpansionDelivery | None:
        """Returns the expansion embeds and media files, or `None` when nothing can be delivered.

        Args:
            message (Message): The message carrying the link.
            url (str): The post URL that was read.
            parsed (ThreadsConversation): The parsed conversation.
        """
        # Expand the ancestor chain only; replies belong to `gen_reply`.
        results = parsed.chain
        if not results:
            logfire.info("Threads parse returned no post; treating as unavailable", url=url)
            return None

        target = results[-1]
        if target.quoted_unavailable:
            logfire.info("A Threads post quotes a post Threads no longer serves", url=url)
        embeds = self._build_embeds(results=results)
        # Check rendered descriptions (with suffixes) against Discord's 4096-char limit.
        longest_text = max(
            (utf16_length(value=embed.description or "") for embed in embeds), default=0
        )
        if longest_text > DISCORD_EMBED_DESCRIPTION_LIMIT:
            logfire.info(
                "Threads post exceeds the embed description limit; skipping expansion",
                url=url,
                text_length=longest_text,
            )
            return None

        upload_limit = (
            upload_limit_for(guild=message.guild)
            if target_allows_file_uploads(target=message)
            else 0
        )
        items = [MediaItem(source=path, filename=path.name) for path in target.video_paths]
        plan = await self.media_delivery.plan(
            items=items, upload_limit=upload_limit, envelope_margin=MEDIA_ENVELOPE_MARGIN
        )
        if plan.dropped_items:
            # Refuse the whole post if any video cannot be attached or hosted.
            logfire.warn(
                "Threads videos could not be hosted; refusing the whole post",
                url=url,
                message_id=message.id,
                dropped=len(plan.dropped_items),
            )
            return None

        lines: list[str] = []
        if plan.clamped_items:
            lines.append(
                f"-# 已省略 {len(plan.clamped_items)} 部影片 (Discord 單則訊息最多 "
                f"{DISCORD_ATTACHMENT_LIMIT} 個附件)"
            )
        lines.extend(plan.hosted_urls)
        return ExpansionDelivery(
            content="\n".join(lines) if lines else None, embeds=embeds, media=plan.native
        )

    @staticmethod
    def _gradient_color(index: int, total: int) -> Color:
        """Returns a greyscale shade from root (lightest) to leaf (darkest), or default solo."""
        if total <= 1:
            return Color.default()
        # Stay inside [0x40, 0xC0]; default/black is reserved for solo posts.
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
                name=output.author_name, url=output.url, icon_url=output.author_icon_url or None
            )
        embed.set_footer(text=output.counters_line())
        return embed

    def _build_post_embeds(
        self,
        output: ThreadsOutput,
        color: Color,
        image_count: int,
        is_target: bool,
        is_quoted: bool = False,
    ) -> list[Embed]:
        """Builds the main embed plus gallery embeds for one post, showing `image_count` images."""
        main_embed = self._build_post_embed(output=output, color=color)
        # Label quoted posts explicitly since they sit outside the thread gradient.
        if is_quoted:
            main_embed.description = f"🔗 **被引用的貼文**\n\n{main_embed.description or ''}"
        embeds = with_gallery(card=main_embed, images=output.image_urls[:image_count])
        # Ancestor/quoted videos are not downloaded; add a link hint so embeds stay non-empty.
        if not is_target and output.video_urls and output.url:
            hint = VIDEO_HINT.format(url=output.url)
            main_embed.description = (main_embed.description or "") + hint
        # Only the target's quote is rendered, so only the target reports an unavailable quote.
        if is_target and output.quoted_unavailable:
            main_embed.description = (main_embed.description or "") + _QUOTED_UNAVAILABLE_HINT
        return embeds

    def _select_posts_within_text_limit(
        self, posts: list[ThreadsOutput], priority: list[int], chain_depth: int, quoted_index: int
    ) -> set[int]:
        """Selects complete posts within the message text budget; always keeps the target."""
        selected: set[int] = set()
        text_budget = DISCORD_EMBED_TOTAL_LIMIT - _REMAINDER_RESERVE
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
            length = embed_text_length(embed=main_embed)
            if index != chain_depth - 1 and length > text_budget:
                continue
            selected.add(index)
            text_budget -= length
        return selected

    def _build_embeds(self, results: list[ThreadsOutput]) -> list[Embed]:
        """Builds at most ten embeds for the ordered reply chain `[root, ..., direct_parent, target]`."""
        # Keep at most 10 newest posts; prioritize target, quoted post, then nearest ancestors.
        trimmed = 0
        if len(results) > DISCORD_EMBED_COUNT_LIMIT:
            trimmed = len(results) - DISCORD_EMBED_COUNT_LIMIT
            results = results[-DISCORD_EMBED_COUNT_LIMIT:]
        chain_depth = len(results)
        # Allocated alongside the chain, emitted after it.
        quoted = results[-1].quoted if results else None
        posts = [*results, *([quoted] if quoted is not None else [])]
        quoted_index = chain_depth if quoted is not None else -1

        # Allocation order (target, quoted, nearest ancestors) differs from emission order.
        priority = [chain_depth - 1, quoted_index, *reversed(range(chain_depth - 1))]
        priority = [index for index in priority if index >= 0]

        selected = self._select_posts_within_text_limit(
            posts=posts, priority=priority, chain_depth=chain_depth, quoted_index=quoted_index
        )
        priority = [index for index in priority if index in selected]

        # Reserve one slot each for target and quoted post before gallery images.
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

        # Count only omitted posts and omitted target images (ancestor images are best-effort).
        if target_embed is not None:
            notes = _remainder_notes(
                omitted_posts=trimmed + sum(1 for slot in slots if slot == 0),
                omitted_images=len(posts[target_index].image_urls) - slots[target_index],
            )
            if notes:
                carried = [part for part in (target_embed.footer.text, *notes) if part]
                target_embed.set_footer(text=" | ".join(carried))
        return embeds

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
        """Replies ephemerally with the canonical URL of a Threads post or share link.

        Resolves only the share redirect without fetching post JSON or media, stripping share
        tracking parameters.

        Args:
            interaction (Interaction[commands.Bot]): The interaction that triggered the command.
            url (str): The Threads link to clean, or the share text carrying it.
        """
        await interaction.response.defer(ephemeral=True)
        # Search the full option text so pasted share blobs work.
        raw_url = self.POST_LINK.first_in(texts=(url,))
        if raw_url is None:
            await interaction.followup.send(
                content="這裡面沒有 Threads 貼文連結。", ephemeral=True
            )
            return

        # `resolve_clean_url` writes nothing to disk.
        downloader = self.downloader_factory(output_folder="")
        try:
            async with asyncio.timeout(delay=THREADS_PAGE_TIMEOUT_SECONDS):
                clean_url = await asyncio.to_thread(downloader.resolve_clean_url, url=raw_url)
        except (RuntimeError, TimeoutError) as error:
            logfire.warn(
                "Could not resolve a Threads share link",
                # Strip `?xmt=` share-tracking token before logging.
                url=ThreadsURL(raw_url=raw_url).clean_url,
                error_type=type(error).__name__,
                _exc_info=error,
            )
            await interaction.followup.send(content="這個連結現在拿不到。", ephemeral=True)
            return

        if not clean_url:
            await interaction.followup.send(
                content="這個分享連結沒有指向任何貼文。", ephemeral=True
            )
            return
        await interaction.followup.send(content=clean_url, ephemeral=True)


def setup(bot: commands.Bot) -> None:
    """Adds the ThreadsCogs to the bot.

    Args:
        bot (commands.Bot): The Discord bot instance.
    """
    bot.add_cog(ThreadsCogs(bot), override=True)
