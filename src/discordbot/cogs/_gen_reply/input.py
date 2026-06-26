"""Builds Responses API input messages from Discord messages."""

import re
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

from discordbot.typings.models import RuntimeModelCatalog
from discordbot.utils.model_pricing import get_supported_modalities
from discordbot.cogs._gen_reply.generation import VOICE_REPLY_FILENAME
from discordbot.cogs._gen_reply.attachment.base import (
    RenderedPart,
    AttachmentRenderer,
    loggable_cache_key,
)
from discordbot.cogs._gen_reply.attachment.loaders import load_image_bytes

if TYPE_CHECKING:
    from collections.abc import Sequence, Coroutine

# Strips the usage_footer appended by `streaming.ResponseStreamer.stream` from
# bot-authored messages before feeding them back as `role=assistant` history.
# Without this, the model performs in-context learning on its own past footers
# and starts hallucinating fake "-# model · ⬆ ... ⬇ ... · $... · ..." lines into
# fresh replies. Anchored on the `\n\n-# ` separator plus the ⬆/⬇ token-count
# icons, which never appear together in user-authored content. The optional
# trailing `\n-# ...` line matches the second subtext line that credits looked-up
# memory owners, so the whole footer is stripped as one unit.
USAGE_FOOTER_RE = re.compile(r"\n\n-#[^\n]*⬆[^\n]*⬇[^\n]*(?:\n-#[^\n]*)?$")

# A display name (or legacy username) containing an `[id: ...]`-shaped string
# could forge the sender-identity prefix this module prepends, which the reply
# persona prompt and the memory extraction prompt both treat as the trusted
# authorship signal. Neutralize the lookalike before rendering.
_ID_PREFIX_LOOKALIKE_RE = re.compile(r"\[\s*id\s*:", flags=re.IGNORECASE)

# Marker prefixing each forwarded snapshot span appended to a rendered message body. The
# answer model uses it to attribute forwarded content; the memory transcript strips from it
# to end-of-body so a forward of someone else's words is never recorded as the forwarder's
# own fact (forwarded text is always appended last, so the marker is the suffix boundary).
FORWARDED_MESSAGE_MARKER = "[forwarded message]"


def sanitize_identity(value: str) -> str:
    """Neutralizes authorship-prefix lookalikes in user-controlled identity fields."""
    return _ID_PREFIX_LOOKALIKE_RE.sub("[id-", value)


def render_author_identity(display_name: str, username: str, user_id: int) -> str:
    """Renders the single-line author identity stamped into memory files.

    Whitespace runs (including any newline that slips past Discord's name
    rules) collapse to single spaces so the identity can never break the
    one-line header formats the memory store relies on.
    """
    safe_display = " ".join(sanitize_identity(value=display_name).split())
    safe_username = " ".join(sanitize_identity(value=username).split())
    return f"{safe_display} ({safe_username}) [id: {user_id}]"


def render_server_identity(server_name: str, server_id: int) -> str:
    """Renders the single-line server identity stamped into per-server memory files.

    Mirrors `render_author_identity`: the guild name is user-controlled, so it
    is sanitized against `[id:` lookalikes and collapsed to one line before the
    `[id: <server_id>]` suffix the memory store's identity regex expects.
    """
    safe_name = " ".join(sanitize_identity(value=server_name).split())
    return f"{safe_name} [id: {server_id}]"


class AttachmentSource(BaseModel):
    """One renderable attachment source classified from message metadata.

    Collected once per message and shared by the text-only marker render, the
    Files-API upload, the per-message render cache key, and the IMAGE route's
    raw-bytes path. Carries only metadata (no bytes, no network) so it is safe to
    build on the route critical path.

    Attributes:
        handle: The attachment, sticker, or image URL the loaders consume.
        kind: Whether the source renders as an image or a generic file.
        content_type: Resolved MIME type, empty only for unguessable sources.
        cache_key: Stable identity (attachment/sticker id or chosen embed URL).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    handle: SkipValidation[Attachment | StickerItem | str] = Field(
        ..., description="The attachment, sticker, or image URL the loaders consume."
    )
    kind: Literal["image", "file"] = Field(
        ..., description="Whether the source renders as an image or a generic file."
    )
    content_type: str = Field(
        ..., description="Resolved MIME type, empty only for unguessable sources."
    )
    cache_key: int | str = Field(
        ...,
        description="Stable identity (attachment/sticker id or chosen embed URL) for the cache.",
    )


class MessageInputBuilder(BaseModel):
    """Converts Discord messages into Responses API input parts.

    Attributes:
        bot: The Discord bot instance, used to detect the bot's own messages.
        runtime_models: Catalog whose slow model gates attachment modalities.
        attachment_handler: Strategy that renders each attachment source to a content part.
    """

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
    # source count still re-renders. Each entry pairs the files' real expiry (the earliest
    # Gemini `expiration_time` across the rendered parts) with the parts, so a handle is
    # re-uploaded just before it actually expires instead of on a guessed fixed TTL.
    _attachment_cache: OrderedDict[
        tuple[int, datetime | None, tuple[int | str, ...]], tuple[datetime, list[RenderedPart]]
    ] = PrivateAttr(default_factory=OrderedDict)

    async def get_user_prompt(self, content: str) -> str:
        """Removes bot mention syntax from image/video generation prompts."""
        if self.bot.user:
            bot_id = re.escape(str(self.bot.user.id))
            content = re.sub(rf"<@!?{bot_id}>", "", content)
        return content.strip()

    def has_bot_mention(self, content: str) -> bool:
        """Returns whether the content mentions the bot directly."""
        if not self.bot.user:
            return False
        bot_id = re.escape(str(self.bot.user.id))
        return re.search(rf"<@!?{bot_id}>", content) is not None

    @staticmethod
    def extract_embed_text(embeds: list[Embed]) -> str:
        """Joins author / title / link / description / fields / footer text from embeds.

        The embed's own `url` is included so the answer model actually sees a link card's
        target (e.g. a forwarded link with no caption) and the URL detectors stay aligned with
        the rendered text instead of reacting to a link the model was never shown.
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
    def _snapshot_text(snapshot: MessageSnapshot) -> str:
        """Raw forwarded text of one snapshot: its content, or its embeds as a fallback.

        A forward of the bot's own reply carries no `author`, so the usage-footer strip
        `get_cleaned_content` applies to the bot's own messages cannot fire here; the same regex
        is applied directly so a forwarded reply never re-injects the `-# ... ⬆ ... ⬇ ...` footer
        the model would otherwise learn to mimic. The pattern is bot-footer-shaped, so it is a
        no-op on forwarded user text.
        """
        text = USAGE_FOOTER_RE.sub("", snapshot.content).strip()
        if not text and snapshot.embeds:
            text = MessageInputBuilder.extract_embed_text(embeds=list(snapshot.embeds))
        return text

    def _forwarded_snapshot_text(self, message: Message) -> str:
        """Renders a forwarded message's snapshots, each tagged `[forwarded message]`.

        A Discord forward leaves `content`/`embeds`/`attachments` empty and puts the original
        payload in `message.snapshots`; the tag tells the model the span is forwarded so it does
        not credit the forwarder with the words. Empty for a normal message (no snapshots); a
        media-only forward still emits the bare tag since its attachment rides separately via
        `collect_attachment_sources`.
        """
        blocks: list[str] = []
        for snapshot in message.snapshots:
            text = self._snapshot_text(snapshot=snapshot)
            blocks.append(
                f"{FORWARDED_MESSAGE_MARKER}: {text}" if text else FORWARDED_MESSAGE_MARKER
            )
        return "\n".join(blocks)

    def forwarded_request_text(self, message: Message) -> str:
        """Concatenated raw forwarded text (no `[forwarded message]` tag) across snapshots.

        Used as the media prompt when a pure forward carries no comment of its own, so a
        forwarded "draw a cat" reaches the IMAGE/VIDEO handler as its actual request instead
        of the generic fallback. Empty for a normal message or a media-only forward.
        """
        texts = [
            text
            for snapshot in message.snapshots
            if (text := self._snapshot_text(snapshot=snapshot))
        ]
        return "\n".join(texts)

    async def get_cleaned_content(self, message: Message) -> str:
        """Returns the textual content of a message without the author prefix."""
        content = message.content.strip()
        if content and self.bot.user and message.author.id == self.bot.user.id:
            content = USAGE_FOOTER_RE.sub("", content)
        if not content and message.embeds:
            content = self.extract_embed_text(embeds=list(message.embeds))
        if not content and message.is_system():
            content = message.system_content
        # A forward can also carry the forwarder's own comment, so append rather than replace.
        forwarded = self._forwarded_snapshot_text(message=message)
        if forwarded:
            content = f"{content}\n{forwarded}".strip() if content else forwarded
        return content

    def collect_attachment_sources(self, message: Message) -> list[AttachmentSource]:
        """Classifies every renderable attachment source on a message from metadata.

        One metadata-only pass shared by the text-only marker render, the Files-API
        upload, the render cache key, and the IMAGE route; does no network or upload
        work so it is safe to call on the route critical path. Embeds prefer Discord's
        `proxy_url` (media.discordapp.net) over the origin URL, since sources like the
        Threads CDN expire and reject requests without specific headers. A forwarded
        message's media is folded in from `message.snapshots` so a forward is not blank.
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
        *,
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
            content_type = attachment.content_type or guess_type(attachment.filename)[0] or ""
            sources.append(
                AttachmentSource(
                    handle=attachment,
                    kind="image" if content_type.startswith("image/") else "file",
                    content_type=content_type,
                    cache_key=attachment.id,
                )
            )
        for sticker in stickers:
            sources.append(
                AttachmentSource(
                    handle=sticker,
                    kind="image",
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

    def _supported_sources(self, sources: list[AttachmentSource]) -> list[AttachmentSource]:
        """Drops sources whose required modality the slow model cannot accept.

        Gating once on the shared source list keeps the text-only marker render and the
        Files-API upload render in agreement: the route never marks an attachment the
        answer would silently drop, and vice versa.
        """
        if not sources:
            return []
        model_name = self.runtime_models.slow_model.name
        modalities = get_supported_modalities(model_name=model_name)
        supported: list[AttachmentSource] = []
        for source in sources:
            required = self.required_modality(content_type=source.content_type)
            if required not in modalities:
                logfire.warn(
                    "gen_reply skipping unsupported attachment",
                    modality=required,
                    model=model_name,
                    cache_key=loggable_cache_key(cache_key=source.cache_key),
                    content_type=source.content_type,
                )
                continue
            supported.append(source)
        return supported

    async def get_image_sources_with_mime(self, message: Message) -> list[tuple[bytes, str]]:
        """Returns downscaled (bytes, MIME) pairs of a message's image sources.

        Image editing feeds raw pixels to `images.edit`, so it loads bytes directly
        rather than reusing the Files-API handles `get_attachment_parts` produces. Only
        image sources are collected; non-image files are not editable as images. The
        IMAGE/VIDEO routes run on the image/video model, so the slow model's modality gate
        is not applied here. The MIME is kept because the native Veo `types.Image` requires
        it; the IMAGE route drops it via `get_image_source_bytes`.
        """
        tasks: list[Coroutine[object, object, tuple[bytes, str]]] = []
        for source in self.collect_attachment_sources(message=message):
            if source.kind == "image":
                tasks.append(load_image_bytes(source=source.handle))
        loaded = await asyncio.gather(*tasks, return_exceptions=True)
        return [item for item in loaded if isinstance(item, tuple)]

    async def get_image_source_bytes(self, message: Message) -> list[bytes]:
        """Returns downscaled bytes of a message's image sources for the IMAGE route."""
        return [raw for raw, _ in await self.get_image_sources_with_mime(message=message)]

    async def get_video_thumbnail_sources(self, message: Message) -> list[tuple[bytes, str]]:
        """Best-effort (bytes, MIME) poster frames of a message's video attachments.

        Veo cannot ingest a raw video, so a referenced/attached video contributes its poster
        frame as a reference image instead: Discord's media proxy renders one from the
        attachment's `proxy_url` via `?format=jpeg`, and `load_image_bytes` decodes it, so a
        proxy that returns no frame is simply skipped. Video links Discord unfurled into an
        embed thumbnail are already collected as image sources by `collect_attachment_sources`,
        so only direct video attachments are handled here.
        """
        tasks: list[Coroutine[object, object, tuple[bytes, str]]] = []
        for source in self.collect_attachment_sources(message=message):
            if source.content_type.startswith("video/") and isinstance(source.handle, Attachment):
                tasks.append(load_image_bytes(source=f"{source.handle.proxy_url}?format=jpeg"))
        loaded = await asyncio.gather(*tasks, return_exceptions=True)
        return [item for item in loaded if isinstance(item, tuple)]

    @staticmethod
    def required_modality(content_type: str) -> Literal["image", "video", "audio", "unknown"]:
        """Maps a MIME type to the input modality the model must accept.

        Documents (PDF / Office / text / code) fall through to `image` as a
        proxy: LiteLLM only reports text/image/audio/video, and image-capable
        models in practice also accept `input_file`. Known binaries (archives,
        executables, octet-stream) are checked first and return `unknown` so
        they are dropped before reaching the API.
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
        })
        if content_type in unsupported_binary_mimes:
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
        # attachments, so the denylist above stays the only drop rule; the renderers make the
        # final call (the inline path keeps only PDF + UTF-8, Gemini ingests the rest).
        return "image"

    async def _render_attachment_parts(
        self, sources: list[AttachmentSource], allow_dead_cache: bool = False
    ) -> list[tuple[RenderedPart, datetime] | None]:
        """Renders every supported source to a content part + expiry; failures stay None.

        Each source renders concurrently, so a message with several attachments pays
        roughly one upload's latency (Gemini) or one download's latency (inline), not the sum.
        """
        tasks: list[Coroutine[object, object, tuple[RenderedPart, datetime] | None]] = []
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
        self,
        message: Message,
        sources: list[AttachmentSource] | None = None,
        allow_dead_cache: bool = False,
    ) -> list[RenderedPart]:
        """Extracts attachment content parts from a message, with a per-message cache.

        Pass the pre-collected supported `sources` to avoid re-collecting; when omitted
        they are collected and gated here so direct callers keep working. `allow_dead_cache`
        is opt-in for history scrollback only (see `GeminiFileUploader._resolve_file_upload`).
        """
        if sources is None:
            sources = self._supported_sources(
                sources=self.collect_attachment_sources(message=message)
            )
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
        resolved = [item[0] for item in rendered if item is not None]
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
            expires_at = min(item[1] for item in rendered if item is not None)
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
        """Renders a message as cleaned text plus `[attachment: kind]` markers.

        Pure metadata plus the already-cheap cleaned content; performs no upload, so the
        route and memory-selection calls never wait on the Files API. Mirrors
        `process_single_message`'s role and prefix rules so the route sees the same shape
        the answer will, minus the payload bytes.
        """
        content = await self.get_cleaned_content(message=message)
        markers: list[ResponseInputTextParam] = [
            ResponseInputTextParam(text=f"[attachment: {source.kind}]", type="input_text")
            for source in sources
        ]
        return self._assemble_input_message(
            message=message, content=content, parts=markers, has_attachments=bool(sources)
        )

    async def process_single_message_text_only(self, message: Message) -> EasyInputMessageParam:
        """Renders a message for the route and memory-selection calls without uploading."""
        try:
            sources = self._supported_sources(
                sources=self.collect_attachment_sources(message=message)
            )
            return await self.render_text_only(message=message, sources=sources)
        except Exception:
            # The route awaits this before dispatching, so a cold-start modality lookup
            # (or render) failure must degrade to empty text like process_single_message
            # does, not abort the whole reply through the generic error path.
            logfire.warn(
                "gen_reply failed to render message for routing",
                message_id=message.id,
                _exc_info=True,
            )
            return EasyInputMessageParam(role="user", content="")

    async def process_single_message(
        self, message: Message, allow_dead_cache: bool = False
    ) -> EasyInputMessageParam:
        """Processes a single Discord message into a Responses API input message.

        `allow_dead_cache` is set only for history scrollback, where an expired CDN source
        re-fails every turn; current/reference renders leave it off so a transient failure
        on a just-posted attachment is retried on the next reply.
        """
        try:
            content = await self.get_cleaned_content(message=message)
            sources = self._supported_sources(
                sources=self.collect_attachment_sources(message=message)
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
        except Exception:
            logfire.warn(
                "gen_reply failed to process message", message_id=message.id, _exc_info=True
            )
            return EasyInputMessageParam(role="user", content="")
