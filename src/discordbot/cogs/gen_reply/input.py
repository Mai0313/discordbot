"""Builds Responses API input messages from Discord messages."""

from typing import TYPE_CHECKING, Literal, cast
import asyncio
from datetime import UTC, datetime, timedelta
from mimetypes import guess_type
from collections import OrderedDict

import logfire
from nextcord import Embed, Message, Attachment, StickerItem, MessageSnapshot
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation
from nextcord.ext import commands
from openai.types.responses.response_input_param import EasyInputMessageParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam

from discordbot.typings.media import LoadedMedia, RenderedPart, RenderedAttachment
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.utils.mentions import bot_mention_pattern
from discordbot.utils.model_pricing import get_supported_modalities
from discordbot.utils.llm_transcript import (
    USAGE_FOOTER_RE,
    FORWARDED_MESSAGE_MARKER,
    sanitize_identity,
)
from discordbot.cogs.gen_reply.generation import VOICE_REPLY_FILENAME
from discordbot.cogs.gen_reply.attachment.base import AttachmentRenderer, loggable_cache_key
from discordbot.cogs.gen_reply.attachment.loaders import (
    attachment_mime,
    load_image_bytes,
    load_attachment_bytes,
)

if TYPE_CHECKING:
    from collections.abc import Sequence, Coroutine


class AttachmentSource(BaseModel):
    """One renderable attachment source classified from message metadata.

    Collected once per message and shared by every render of it. Carries only metadata (no
    bytes, no network) so it is safe to build on the route critical path.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    handle: SkipValidation[Attachment | StickerItem | str] = Field(
        ..., description="The attachment, sticker, or image URL the loaders consume."
    )
    kind: Literal["image", "file"] = Field(
        ..., description="Whether the source renders as an image or a generic file."
    )
    is_sticker: bool = Field(
        default=False,
        description=(
            "Whether the source is a Discord sticker. Separate from `kind`, which stays the "
            "render path a sticker shares with every other image; this is what the text-only "
            "marker names it."
        ),
    )
    content_type: str = Field(
        ...,
        description=(
            "Resolved MIME type; empty when the sent type is blank or parameter-only, or when "
            "none is sent and the filename is unguessable."
        ),
    )
    cache_key: int | str = Field(
        ...,
        description="Stable identity (attachment/sticker id or chosen embed URL) for the cache.",
    )
    size: int | None = Field(
        default=None,
        description=(
            "Bytes Discord reports for an attachment; None for a sticker or an embed image, "
            "which report none."
        ),
    )


class MessageInputBuilder(BaseModel):
    """Converts Discord messages into Responses API input parts."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bot: SkipValidation[commands.Bot] = Field(
        ..., description="The Discord bot instance, used to detect the bot's own messages."
    )
    runtime_models: RuntimeModelCatalog = Field(
        ..., description="Catalog whose slow model gates attachment modalities."
    )
    attachment_handler: SkipValidation[AttachmentRenderer] = Field(
        ..., description="Strategy that renders each attachment source to a content part."
    )
    # Rendered attachment parts per message, so replying repeatedly in the same channel
    # does not re-upload the same history attachments every time. Keyed on the exact
    # sources rendered (attachment + sticker ids, embed image/thumbnail URLs) plus edit
    # time, so an edit or a late embed unfurl that swaps a URL without changing the
    # source count still re-renders. Each entry pairs the parts with the earliest expiry among
    # them (the provider's own for an uploaded file), so a handle is re-uploaded just before it
    # actually expires instead of on a guessed fixed TTL.
    _attachment_cache: OrderedDict[
        tuple[int, datetime | None, tuple[int | str, ...]], tuple[datetime, list[RenderedPart]]
    ] = PrivateAttr(default_factory=OrderedDict)

    async def get_user_prompt(self, content: str) -> str:
        """Returns `content` with bot mention syntax removed and surrounding whitespace stripped.

        Args:
            content (str): Raw message text.
        """
        if self.bot.user:
            content = bot_mention_pattern(bot_id=self.bot.user.id).sub(repl="", string=content)
        return content.strip()

    @staticmethod
    def extract_embed_text(embeds: list[Embed]) -> str:
        """Returns author, title, link, description, fields, and footer text joined from `embeds`.

        Includes each embed's `url` so the answer model and URL detectors see the same link
        target.

        Args:
            embeds (list[Embed]): Discord embeds to extract text from.
        """
        embed_parts: list[str] = []
        for embed in embeds:
            parts: list[str] = []
            if embed.author and embed.author.name:
                parts.append(f"Author: {embed.author.name}")
            if embed.title:
                parts.append(f"Title: {embed.title}")
            if embed.url:
                parts.append(f"Link: {embed.url}")
            if embed.description:
                parts.append(embed.description)
            for field in embed.fields:
                parts.append(f"{field.name}: {field.value}")
            if embed.footer and embed.footer.text:
                parts.append(f"Footer: {embed.footer.text}")
            if parts:
                embed_parts.append("\n".join(parts))
        return "\n\n".join(embed_parts)

    @staticmethod
    def snapshot_text(snapshot: MessageSnapshot) -> str:
        """Returns the rendered text of `snapshot`: its content (minus usage footer) or embed text.

        Strips `USAGE_FOOTER_RE` directly because a forwarded bot reply carries no `author` on
        its snapshot.

        Args:
            snapshot (MessageSnapshot): Forwarded message snapshot to render.
        """
        content_present = bool(snapshot.content.strip())
        text = USAGE_FOOTER_RE.sub("", snapshot.content).strip()
        if not content_present and snapshot.embeds:
            text = MessageInputBuilder.extract_embed_text(embeds=list(snapshot.embeds))
        return text

    def _forwardedsnapshot_text(self, message: Message) -> str:
        """Renders `message.snapshots` with each snapshot prefixed by `[forwarded message]`."""
        blocks: list[str] = []
        for snapshot in message.snapshots:
            text = self.snapshot_text(snapshot=snapshot)
            blocks.append(
                f"{FORWARDED_MESSAGE_MARKER}: {text}" if text else FORWARDED_MESSAGE_MARKER
            )
        return "\n".join(blocks)

    def forwarded_request_text(self, message: Message) -> str:
        """Returns the concatenated raw forwarded text across `message.snapshots` without tags.

        Args:
            message (Message): Message whose forwarded snapshots are read.
        """
        texts = [
            text
            for snapshot in message.snapshots
            if (text := self.snapshot_text(snapshot=snapshot))
        ]
        return "\n".join(texts)

    async def get_cleaned_content(self, message: Message) -> str:
        """Returns `message`'s cleaned text, with embed/system fallbacks and forwarded snapshots.

        Strips the usage footer from the bot's own messages so model/token/cost lines are not
        fed back into the conversation.

        Args:
            message (Message): Message whose text content is extracted.
        """
        content = message.content
        content_present = bool(content.strip())
        if content and self.bot.user and message.author.id == self.bot.user.id:
            content = USAGE_FOOTER_RE.sub("", content)
        content = content.strip()
        if not content_present and message.embeds:
            content = self.extract_embed_text(embeds=list(message.embeds))
        if not content and message.is_system():
            content = message.system_content
        # A forward can also carry the forwarder's own comment, so append rather than replace.
        forwarded = self._forwardedsnapshot_text(message=message)
        if forwarded:
            content = f"{content}\n{forwarded}".strip() if content else forwarded
        return content

    def collect_attachment_sources(self, message: Message) -> list[AttachmentSource]:
        """Returns every renderable attachment source classified from `message` and its snapshots.

        Performs metadata-only classification with no network I/O, preferring embed `proxy_url`
        over origin URLs.

        Args:
            message (Message): Message whose attachments, stickers, embeds, and snapshots are
                inspected.
        """
        is_own_message = bool(self.bot.user and message.author.id == self.bot.user.id)
        sources = self._sources_from_parts(
            attachments=list(message.attachments),
            stickers=list(message.stickers),
            embeds=list(message.embeds),
            drop_own_voice=is_own_message,
        )
        # A forward's real payload lives in `message.snapshots`; the original author is not the
        # bot, so the own-voice skip never applies. The snapshot carries stickers as
        # `sticker_items`, and its attachment/embed ids stay unique so the render cache never
        # collides. Empty for a normal message.
        for snapshot in message.snapshots:
            sources.extend(
                self._sources_from_parts(
                    attachments=list(snapshot.attachments),
                    stickers=list(snapshot.sticker_items),
                    embeds=list(snapshot.embeds),
                    drop_own_voice=False,
                )
            )
        return sources

    def _sources_from_parts(
        self,
        attachments: list[Attachment],
        stickers: list[StickerItem],
        embeds: list[Embed],
        drop_own_voice: bool,
    ) -> list[AttachmentSource]:
        """Classifies one carrier's attachments / stickers / embed images into sources.

        Shared by the message body and each forwarded snapshot. `drop_own_voice` skips the bot's
        own generated voice clip (only meaningful on the bot's own message body).
        """
        sources: list[AttachmentSource] = []
        for attachment in attachments:
            # Skip the bot's own generated voice clip: its text is already in the transcript,
            # so re-uploading the WAV only adds latency and feeds the model duplicate self-output.
            if drop_own_voice and attachment.filename == VOICE_REPLY_FILENAME:
                continue
            content_type = attachment_mime(attachment=attachment)
            sources.append(
                AttachmentSource(
                    handle=attachment,
                    kind="image" if content_type.startswith("image/") else "file",
                    content_type=content_type,
                    cache_key=attachment.id,
                    size=attachment.size,
                )
            )
        for sticker in stickers:
            sources.append(
                AttachmentSource(
                    handle=sticker,
                    kind="image",
                    is_sticker=True,
                    content_type=guess_type(sticker.url)[0] or "image/png",
                    cache_key=sticker.id,
                )
            )
        for embed in embeds:
            if embed.image and (url := embed.image.proxy_url or embed.image.url):
                sources.append(
                    AttachmentSource(
                        handle=url,
                        kind="image",
                        content_type=guess_type(url)[0] or "image/png",
                        cache_key=url,
                    )
                )
            if embed.thumbnail and (url := embed.thumbnail.proxy_url or embed.thumbnail.url):
                sources.append(
                    AttachmentSource(
                        handle=url,
                        kind="image",
                        content_type=guess_type(url)[0] or "image/png",
                        cache_key=url,
                    )
                )
        return sources

    def _split_on_modality(
        self, sources: list[AttachmentSource], model_name: str
    ) -> tuple[list[AttachmentSource], list[tuple[AttachmentSource, str, bool]]]:
        """Splits `sources` into accepted sources and rejected `(source, modality, carried)` tuples.

        Takes `model_name` explicitly so the gate and its caller log against the same snapshot.
        """
        modalities = (
            get_supported_modalities(model_name=model_name)
            - self.attachment_handler.dropped_modalities
        )
        accepted: list[AttachmentSource] = []
        rejected: list[tuple[AttachmentSource, str, bool]] = []
        for source in sources:
            required = self.required_modality(content_type=source.content_type)
            carried = self.attachment_handler.carries(
                content_type=source.content_type, cache_key=source.cache_key, size=source.size
            )
            if required in modalities and carried:
                accepted.append(source)
            else:
                rejected.append((source, required, carried))
        return accepted, rejected

    def count_supported_sources(self, message: Message) -> int:
        """Returns the number of attachment sources on `message` accepted by the slow model and renderer.

        Args:
            message (Message): Message whose supported attachment sources are counted.
        """
        sources = self.collect_attachment_sources(message=message)
        if not sources:
            return 0
        accepted, _ = self._split_on_modality(
            sources=sources, model_name=self.runtime_models.slow_model.name
        )
        return len(accepted)

    def _supported_sources(
        self, sources: list[AttachmentSource], message_id: int
    ) -> list[AttachmentSource]:
        """Returns the subset of `sources` accepted by the slow model and renderer, logging skips."""
        if not sources:
            return []
        model_name = self.runtime_models.slow_model.name
        accepted, rejected = self._split_on_modality(sources=sources, model_name=model_name)
        for source, required, carried in rejected:
            logfire.info(
                "gen_reply skipping unsupported attachment",
                modality=required,
                carried=carried,
                model=model_name,
                renderer=type(self.attachment_handler).__name__,
                cache_key=loggable_cache_key(cache_key=source.cache_key),
                content_type=source.content_type,
                size_bytes=source.size,
                message_id=message_id,
            )
        return accepted

    async def get_image_sources_with_mime(self, message: Message) -> list[LoadedMedia]:
        """Returns downscaled `(bytes, mime_type)` `LoadedMedia` items for `message`'s image sources.

        Loads raw pixels directly for image/video generation and editing without applying the
        slow model's modality gate.

        Args:
            message (Message): Message whose image sources are downloaded and downscaled.
        """
        tasks: list[Coroutine[object, object, LoadedMedia]] = []
        for source in self.collect_attachment_sources(message=message):
            if source.kind == "image":
                tasks.append(load_image_bytes(source=source.handle))
        loaded = await asyncio.gather(*tasks, return_exceptions=True)
        for item in loaded:
            if isinstance(item, BaseException):
                logfire.warn(
                    "failed to load a source image; generating without it",
                    message_id=message.id,
                    error_type=type(item).__name__,
                    _exc_info=item,
                )
        return [item for item in loaded if isinstance(item, LoadedMedia)]

    async def get_turn_image_sources(
        self, message: Message, replied_to: Message | None
    ) -> list[LoadedMedia]:
        """Returns a turn's source images: `message`'s own followed by `replied_to`'s.

        Args:
            message (Message): Current message being answered.
            replied_to (Message | None): Referenced message being replied to, or None.
        """
        messages = [message] if replied_to is None else [message, replied_to]
        groups = await asyncio.gather(
            *(self.get_image_sources_with_mime(message=source) for source in messages)
        )
        return [loaded for group in groups for loaded in group]

    async def get_video_sources(self, message: Message) -> list[LoadedMedia]:
        """Returns a single-element list with the first readable video attachment on `message`, or `[]`.

        Downloads at most the first usable direct video attachment for video editing, skipping
        failed reads and trying the next attachment.

        Args:
            message (Message): Message to inspect for direct video attachments.
        """
        for source in self.collect_attachment_sources(message=message):
            if source.content_type.startswith("video/") and isinstance(source.handle, Attachment):
                try:
                    return [await load_attachment_bytes(attachment=source.handle)]
                except Exception as exc:
                    # Broad on purpose: a Discord/transport read failure must degrade to the
                    # next clip, never escape into the VIDEO route's hard-fail path.
                    logfire.warn(
                        "failed to read a source video attachment; trying the next",
                        message_id=message.id,
                        filename=source.handle.filename,
                        error_type=type(exc).__name__,
                        _exc_info=exc,
                    )
        return []

    @staticmethod
    def required_modality(content_type: str) -> Literal["image", "video", "audio", "unknown"]:
        """Returns the input modality required by `content_type`, or `"unknown"` for unsupported binaries.

        Maps supported documents and text to `"image"` (the LiteLLM proxy modality for
        `input_file`), and maps empty types, archives, executables, and Office/OpenDocument
        binaries to `"unknown"`.

        Args:
            content_type (str): Normalized MIME type of the attachment source.
        """
        unsupported_binary_mimes = frozenset({
            "application/octet-stream",
            "application/zip",
            "application/x-zip-compressed",
            "application/x-rar-compressed",
            "application/vnd.rar",
            "application/x-7z-compressed",
            "application/x-tar",
            "application/gzip",
            "application/x-gzip",
            "application/x-bzip",
            "application/x-bzip2",
            "application/x-xz",
            "application/java-archive",
            "application/x-msdownload",
            "application/x-dosexec",
            "application/x-executable",
            "application/x-mach-binary",
            "application/x-sharedlib",
            "application/wasm",
            # Office / OpenDocument binaries: the Files API stores them but the Gemini model
            # 400s "Unsupported MIME type" at generation time, so drop them here rather than
            # let one leak into the answer request (which has no per-attachment retry).
            "application/msword",
            "application/vnd.ms-excel",
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.oasis.opendocument.text",
            "application/vnd.oasis.opendocument.spreadsheet",
            "application/vnd.oasis.opendocument.presentation",
        })
        if not content_type or content_type in unsupported_binary_mimes:
            return "unknown"
        if content_type.startswith("video/"):
            return "video"
        if content_type.startswith("audio/"):
            return "audio"
        if content_type.startswith("image/"):
            return "image"
        # Everything else (documents, source code, structured text, an unlisted application
        # type) proxies as `image` / `input_file`. MIME cannot reliably tell an unlisted binary
        # apart from unlisted text/code, and a positive allowlist silently drops legitimate code
        # attachments, so the check above stays the only drop rule every renderer shares; the
        # renderers make the final call (the inline path keeps only PDF + UTF-8, refusing at the
        # gate what it knows it cannot carry; Gemini ingests the rest).
        return "image"

    async def _render_attachment_parts(
        self, sources: list[AttachmentSource], allow_dead_cache: bool = False
    ) -> list[RenderedAttachment | None]:
        """Renders every supported source to a content part + expiry; failures stay None.

        Each source renders concurrently, so a message with several attachments pays
        roughly one upload's latency (Gemini) or one download's latency (inline), not the sum.
        """
        tasks: list[Coroutine[object, object, RenderedAttachment | None]] = []
        for source in sources:
            if source.kind == "image":
                tasks.append(
                    self.attachment_handler.render_image(
                        source=source.handle,
                        cache_key=source.cache_key,
                        allow_dead_cache=allow_dead_cache,
                    )
                )
            else:
                # Only attachments are ever classified as files; stickers and embeds are images.
                tasks.append(
                    self.attachment_handler.render_file(
                        attachment=cast("Attachment", source.handle),
                        cache_key=source.cache_key,
                        allow_dead_cache=allow_dead_cache,
                    )
                )
        return list(await asyncio.gather(*tasks))

    async def get_attachment_parts(
        self, message: Message, sources: list[AttachmentSource], allow_dead_cache: bool = False
    ) -> list[RenderedPart]:
        """Returns rendered attachment content parts for `message`, using the per-message cache when fresh.

        Args:
            message (Message): Message whose attachment sources are rendered.
            sources (list[AttachmentSource]): Modality-filtered attachment sources for `message`.
            allow_dead_cache (bool): Whether permanently dead CDN URLs may be skipped via cache.
        """
        if not sources:
            return []
        # Key on the exact sources rendered plus the edit time, so a late embed unfurl
        # or an edit that swaps a source URL re-renders even when the count is unchanged.
        source_keys = tuple(source.cache_key for source in sources)
        cache_key = (message.id, message.edited_at, source_keys)
        # Reuse the cached handles until shortly before the files actually expire; the
        # margin keeps a borderline-expired URI from reaching the answer request, which
        # has no per-attachment retry and would 400 the whole reply.
        cache_safety_margin = timedelta(hours=2)
        cached = self._attachment_cache.get(cache_key)
        if cached is not None and datetime.now(tz=UTC) < cached[0] - cache_safety_margin:
            self._attachment_cache.move_to_end(cache_key)
            logfire.debug(
                "gen_reply attachment cache hit", message_id=message.id, source_count=len(sources)
            )
            # Hand out per-part copies so no caller ever holds the cached dicts; the
            # values are immutable strings, so the copies stay cheap.
            return [part.copy() for part in cached[1]]

        logfire.debug(
            "gen_reply attachment render", message_id=message.id, source_count=len(sources)
        )
        rendered = await self._render_attachment_parts(
            sources=sources, allow_dead_cache=allow_dead_cache
        )
        resolved = [item.part for item in rendered if item is not None]
        logfire.debug(
            "gen_reply attachment render done",
            message_id=message.id,
            resolved=len(resolved),
            dropped=len(sources) - len(resolved),
        )
        # A None entry means a download/convert or upload failed; skip caching so the
        # next reply retries instead of pinning a degraded render. The entry's expiry is
        # the earliest across its files, so the whole entry re-renders before any handle
        # in it expires.
        if None not in rendered:
            expires_at = min(item.expires_at for item in rendered if item is not None)
            self._attachment_cache[cache_key] = (expires_at, [part.copy() for part in resolved])
            if len(self._attachment_cache) > 128:
                self._attachment_cache.popitem(last=False)
        return resolved

    def _assemble_input_message(
        self,
        message: Message,
        content: str,
        parts: "Sequence[RenderedPart]",
        has_attachments: bool,
    ) -> EasyInputMessageParam:
        """Assembles one input message, sharing role and prefix rules across renders.

        `has_attachments` is decided from the message's sources, not from `parts`, so
        the text-only render (markers) and the full render (uploaded files) agree on
        role and message shape even when every upload is dropped.
        """
        is_bot = bool(self.bot.user and message.author.id == self.bot.user.id)

        # Bot's own history without attachments → role=assistant carries identity, so the
        # sender-prefix is dropped here. Without this, the model sees its own past replies
        # prefixed with `Bot (bot) [id: ...]:` and learns to mimic that header.
        if is_bot and not has_attachments:
            return EasyInputMessageParam(role="assistant", content=content)

        prefixed = (
            f"{sanitize_identity(value=message.author.display_name)} "
            f"({sanitize_identity(value=message.author.name)}) "
            f"[id: {message.author.id}]: {content}"
        )

        # No attachments → use EasyInputMessageParam's string-content shorthand. The SDK
        # serializes it as `input_text` for role=user, which satisfies the strict rule
        # about content-part types per role.
        if not has_attachments:
            return EasyInputMessageParam(role="user", content=prefixed)

        # Has attachments → must use a content list. role=assistant cannot carry
        # `input_file` (only output_text/refusal), so bot replies that include generated
        # images fall back to role=user; the author prefix above preserves bot identity.
        return EasyInputMessageParam(
            role="user", content=[ResponseInputTextParam(text=prefixed, type="input_text"), *parts]
        )

    async def render_text_only(
        self, message: Message, sources: list[AttachmentSource]
    ) -> EasyInputMessageParam:
        """Returns `message` rendered as cleaned text plus `[attachment: ...]` markers without uploading.

        Names stickers `[attachment: sticker]` rather than `[attachment: image]` so a
        sticker-only reaction is distinguishable from a screenshot during routing.

        Args:
            message (Message): Message to render.
            sources (list[AttachmentSource]): Modality-filtered attachment sources for `message`.
        """
        content = await self.get_cleaned_content(message=message)
        markers: list[ResponseInputTextParam] = [
            ResponseInputTextParam(
                text=f"[attachment: {'sticker' if source.is_sticker else source.kind}]",
                type="input_text",
            )
            for source in sources
        ]
        return self._assemble_input_message(
            message=message, content=content, parts=markers, has_attachments=bool(sources)
        )

    async def process_single_message(
        self, message: Message, text_only: bool = False, allow_dead_cache: bool = False
    ) -> EasyInputMessageParam:
        """Returns `message` rendered as a Responses API input message, degrading to empty text on error.

        Args:
            message (Message): Discord message to render.
            text_only (bool): Whether to emit `[attachment: ...]` markers without uploading.
            allow_dead_cache (bool): Whether permanently dead CDN URLs may be skipped via cache
                (set on history scrollback only).
        """
        try:
            if text_only:
                sources = self._supported_sources(
                    sources=self.collect_attachment_sources(message=message), message_id=message.id
                )
                return await self.render_text_only(message=message, sources=sources)
            content = await self.get_cleaned_content(message=message)
            sources = self._supported_sources(
                sources=self.collect_attachment_sources(message=message), message_id=message.id
            )
            attachment_parts = await self.get_attachment_parts(
                message=message, sources=sources, allow_dead_cache=allow_dead_cache
            )
            return self._assemble_input_message(
                message=message,
                content=content,
                parts=attachment_parts,
                has_attachments=bool(sources),
            )
        except Exception as exc:
            # Broad on purpose: the route awaits the text-only render before dispatching and the
            # answer request carries the full one, so any failure across the collect, gate and
            # render steps (an unexpected nextcord shape, say) must degrade this one message to
            # empty text rather than abort the whole reply through the generic error path.
            # Per-attachment download/upload failures never reach here; the handlers drop them
            # to None with their own step-named warn.
            if text_only:
                logfire.warn(
                    "gen_reply failed to render message for routing",
                    message_id=message.id,
                    error_type=type(exc).__name__,
                    _exc_info=exc,
                )
            else:
                logfire.warn(
                    "gen_reply failed to process message",
                    message_id=message.id,
                    error_type=type(exc).__name__,
                    _exc_info=exc,
                )
            return EasyInputMessageParam(role="user", content="")
