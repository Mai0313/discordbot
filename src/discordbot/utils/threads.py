"""Threads URL parsing, API models, and media download helpers."""

import re
import json
import time
from typing import Any
from pathlib import Path
from datetime import UTC, datetime
from functools import cached_property
import contextlib
from urllib.parse import urlparse
from collections.abc import Generator

import logfire
from pydantic import (
    Field,
    BaseModel,
    ValidationInfo,
    ValidationError,
    computed_field,
    field_validator,
)
import requests
from pydantic_core.core_schema import ValidatorFunctionWrapHandler

from discordbot.typings.timeouts import (
    THREADS_PAGE_TIMEOUT_SECONDS,
    THREADS_MEDIA_READ_TIMEOUT_SECONDS,
    THREADS_EMPTY_PAGE_RETRY_DEADLINE_SECONDS,
)

# Single source of truth for detecting a Threads post URL, shared by the parse_threads
# cog (which expands it into embeds) and gen_reply (which self-parses it into answer
# context). Matches the two shapes that name a post on both threads.net and threads.com: the
# canonical `@user/post/<code>`, and the `share/<code>` form the app's share button copies. Both
# paths are anchored, so a profile or any other Threads page still matches nothing. The shortcode
# + query tail is matched as ASCII URL characters only and must END on `[A-Za-z0-9_-]` (the only
# characters a valid Threads code or query value ends in). Restricting to ASCII stops the match
# at any non-ASCII terminator, and the trailing class strips ASCII sentence punctuation, so a
# link written mid-sentence is matched cleanly in both English (`.../post/ABC123.`) and zh/ja
# (`...ABC123。`, `...ABC123】super`) text instead of swallowing the terminator into the code,
# which would otherwise make the parse fail on an otherwise valid link.
THREADS_URL_RE = re.compile(
    r"https?://(?:www\.)?threads\.(?:net|com)/(?:@[^/]+/post|share)/"
    r"[A-Za-z0-9_.?=&%-]*[A-Za-z0-9_-]"
)

# The canonical post path, and the only shape that names its own post. A `share/<code>` link
# carries a code unrelated to the post's (`DfX81RWN8` for a post whose own code is `DZZImVsCWU-`)
# and the page's JSON never carries it, so the share form names its post only through the redirect
# it answers with; `ThreadsURL.post_code` and `ThreadsDownloader.extract_post_data` have the rest.
_POST_PATH_RE = re.compile(r"^/@[^/]+/post/([^/]+)/?$")

# Where a Threads fetch has to be aimed, and every spelling of the host that is aimed there.
# Threads lives on `https://www.threads.com` and 301s everything else to it, so normalising the
# other way costs a redirect on every single read. Measured logged out with this module's own
# headers (#394): `www.threads.net`, `threads.net` and bare `threads.com` each answer 301 to the
# canonical form, `http://` answers 301 to `https://`, and the canonical form answers 200 with
# the same page payload the `.net` host redirected to. Only these hosts are rewritten, so a
# redirect that ever lands somewhere else is fetched where it actually landed rather than
# silently re-pointed at Threads.
_THREADS_HOSTS = frozenset({"threads.com", "www.threads.com", "threads.net", "www.threads.net"})
_CANONICAL_THREADS_ORIGIN = "https://www.threads.com"


class _ThreadsModel(BaseModel):
    """Base model tolerating an explicit JSON null on plain string fields.

    Threads sometimes serialises an optional string field (e.g. a link
    preview's image_url) as an explicit null instead of omitting it. Those
    fields are declared as str with an empty default, so the null would raise a
    ValidationError, which the parser treats as a corrupt block and silently
    drops the whole post. Coercing null to the empty string keeps the default
    semantics for both absent and null values.
    """

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_null_string(cls, value: object, info: ValidationInfo) -> object:
        """Maps a null value to an empty string for str-typed fields."""
        field_name = info.field_name
        if value is None and field_name and cls.model_fields[field_name].annotation is str:
            return ""
        return value


class ThreadsURL(BaseModel):
    """Parses and normalises a Threads post URL.

    Handles both shapes `THREADS_URL_RE` accepts, but only the canonical one names its post:
    `post_code` is empty for a `share/<code>` link, which is the signal to resolve it by fetching
    (see `ThreadsDownloader.extract_post_data`).

    Attributes:
        raw_url: Original Threads URL provided by the caller.
    """

    raw_url: str = Field(..., description="Original Threads URL provided by the caller")

    @computed_field
    @cached_property
    def clean_url(self) -> str:
        """The cleaned and normalised URL, aimed where Threads actually answers.

        Every accepted spelling of the host collapses onto `https://www.threads.com`, which is
        the one form the platform serves without a redirect; see `_CANONICAL_THREADS_ORIGIN` for
        what was measured. A host this module does not own keeps its origin untouched.

        Returns:
            The URL on the canonical origin, with query parameters removed.
        """
        parsed = urlparse(self.raw_url)
        if parsed.netloc.lower() in _THREADS_HOSTS:
            return f"{_CANONICAL_THREADS_ORIGIN}{parsed.path}"
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    @computed_field
    @cached_property
    def post_code(self) -> str:
        """The post short code the URL names.

        Read off the canonical `/@<user>/post/<code>` path rather than taken as the last path
        segment, so a URL that names no post yields nothing instead of a segment of whatever it
        does name. That distinction is the whole share-link handling: the share form's own code
        is not the post's, so an empty result here means "fetch it and read the redirect".

        Returns:
            The short code, or an empty string when the URL is not a canonical post URL.
        """
        match = _POST_PATH_RE.match(string=urlparse(self.raw_url).path)
        return match.group(1) if match else ""


class User(_ThreadsModel):
    """Represents a Threads user.

    Attributes:
        username: User handle.
        profile_pic_url: Profile picture URL.
    """

    username: str = Field(default="", description="Username handle")
    profile_pic_url: str = Field(default="", description="Profile picture URL")


class Caption(_ThreadsModel):
    """Represents caption text attached to a Threads post.

    Attributes:
        text: Caption text content.
    """

    text: str = Field(default="", description="Caption text content")


class VideoVersion(_ThreadsModel):
    """Represents an available video rendition.

    Attributes:
        url: Video file URL.
    """

    url: str = Field(default="", description="Video file URL")


class ImageCandidate(_ThreadsModel):
    """Represents an available image rendition.

    Attributes:
        url: Image URL.
    """

    url: str = Field(default="", description="Image URL")


class ImageVersions2(_ThreadsModel):
    """Holds available image renditions for a media object.

    Attributes:
        candidates: Available image resolutions.
    """

    candidates: list[ImageCandidate] = Field(
        default_factory=list, description="Available image resolutions"
    )


class CarouselMedia(_ThreadsModel):
    """Represents one media item in a Threads carousel.

    Attributes:
        video_versions: Available video renditions.
        image_versions2: Available image renditions.
    """

    video_versions: list[VideoVersion] | None = Field(
        default=None, description="Available video versions"
    )
    image_versions2: ImageVersions2 | None = Field(
        default=None, description="Available image versions"
    )


class MediaContainer(_ThreadsModel):
    """Contains media fields shared by posts and linked inline media.

    Attributes:
        carousel_media: Carousel media items.
        video_versions: Available video renditions.
        image_versions2: Available image renditions.
    """

    carousel_media: list[CarouselMedia] | None = Field(
        default=None, description="Carousel media items"
    )
    video_versions: list[VideoVersion] | None = Field(
        default=None, description="Available video versions"
    )
    image_versions2: ImageVersions2 | None = Field(
        default=None, description="Available image versions"
    )

    @property
    def media_urls(self) -> list[str]:
        """The list of media URLs extracted from the container.

        Returns:
            First media URL from each carousel item, or the first standalone
            video or image URL, with empty values removed.
        """
        urls: list[str] = []
        if self.carousel_media:
            for item in self.carousel_media:
                if item.video_versions:
                    urls.append(item.video_versions[0].url)
                elif item.image_versions2 and item.image_versions2.candidates:
                    urls.append(item.image_versions2.candidates[0].url)
        elif self.video_versions:
            urls.append(self.video_versions[0].url)
        elif self.image_versions2 and self.image_versions2.candidates:
            urls.append(self.image_versions2.candidates[0].url)
        return [u for u in urls if u]


class Fragment(_ThreadsModel):
    """Represents one text fragment from Threads structured text.

    Attributes:
        plaintext: Plain text content of the fragment.
    """

    plaintext: str = Field(default="", description="Plain text content of the fragment")


class TextFragments(_ThreadsModel):
    """Holds ordered structured text fragments.

    Attributes:
        fragments: Ordered list of text fragments.
    """

    fragments: list[Fragment] = Field(
        default_factory=list, description="Ordered list of text fragments"
    )


class LinkPreviewAttachment(_ThreadsModel):
    """Represents metadata for a link preview attachment.

    Attributes:
        title: Title shown in the link preview.
        image_url: Image shown in the link preview.
        url: Original link preview URL.
    """

    title: str = Field(default="", description="Title shown in the link preview")
    image_url: str = Field(default="", description="Image shown in the link preview")
    url: str = Field(default="", description="Original link preview URL")


class LinkedInlineMedia(MediaContainer):
    """Represents media attached through a link preview.

    Attributes:
        code: Linked media short code.
        caption: Linked media caption.
    """

    code: str = Field(default="", description="Linked media short code")
    caption: Caption | None = Field(default=None, description="Linked media caption")


class ShareInfo(_ThreadsModel):
    """Represents what a post quotes or reposts.

    `quoted_post` is a WHOLE post payload, not a reference: measured live across 96 quote
    relations it carries `user`, `caption` / `text_fragments`, `image_versions2` /
    `carousel_media` / `video_versions`, `like_count`, `taken_at` and its own
    `text_post_app_info`, and it arrives at full depth whenever the quoting post is the page target
    (it can still be the tombstone `Post.is_unavailable` describes) — which is the only case
    either caller reads. So it is typed as `Post`, which makes
    `Post -> TextPostAppInfo -> ShareInfo -> Post` a real cycle, hence the forward reference. No
    `model_rebuild()` is needed for it: measured in a cold process, `Post` is already
    `__pydantic_complete__` at import because pydantic resolves the reference as a self-reference,
    and `ShareInfo`'s own standalone schema heals on first use. What the cycle DOES cost is blast
    radius, which `_isolate_quoted_post` below contains.

    Two sibling fields are deliberately NOT modelled, each because a guess would be worse than
    the omission. `quoted_attachment_post` is a separate, mutually exclusive carrier (3 of 404
    posts, never non-null alongside `quoted_post`, and the only one shipping a ready-made
    `permalink`) whose relationship to the quoting post was never established, so rendering it
    as "the post it quotes" could mislabel it. `reposted_post` — a reshare carrying no comment
    of its own — was never non-null across the 351 posts that carry the key, so its shape is
    simply unknown. `quoted_attachment_post_unavailable` is modelled by neither: see
    `Post.is_unavailable` for why the flag is not the tell it looks like.

    Attributes:
        quoted_post: The post this one quotes, absent when it quotes nothing.
    """

    quoted_post: "Post | None" = Field(
        default=None, description="The post this one quotes, absent when it quotes nothing"
    )

    @field_validator("quoted_post", mode="wrap")
    @classmethod
    def _isolate_quoted_post(
        cls, value: object, handler: ValidatorFunctionWrapHandler
    ) -> object | None:
        """Drops an unparsable quoted post instead of failing the post that quotes it.

        Typing this field as a whole `Post` pulls every model in this module into the validation
        of the node that also holds the TARGET, and `_collect_threads` discards a thread node on
        any `ValidationError` — so without this, one unmodelled shape anywhere inside a quoted
        payload costs the linked post itself. Measured on the way in: a `quoted_post` carrying
        `image_versions2: {"candidates": null}`, `text_fragments: {"fragments": null}`,
        `carousel_media: [null]` or a non-numeric `like_count` each took the target down with it,
        and it did not even have to be the TARGET's own quoted post — an ancestor's was enough.

        This is the same isolation `_collect_threads` gives a reply branch, one level lower: the
        quote is the most disposable thing in the payload, and losing only it degrades to exactly
        the pre-quote-post behaviour.
        """
        try:
            return handler(value)
        except ValidationError:
            logfire.warn(
                "A quoted Threads post no longer matches the parser schema; dropping just it",
                _exc_info=True,
            )
            return None


class TextPostAppInfo(_ThreadsModel):
    """Represents Threads-specific post metadata and engagement fields.

    Attributes:
        direct_reply_count: Number of direct replies.
        repost_count: Number of reposts.
        quote_count: Number of quote posts.
        reshare_count: Total reshare count.
        text_fragments: Structured text fragments with links or mentions.
        link_preview_attachment: Preview metadata for shared links.
        linked_inline_media: Inline media attached through a link preview.
        is_reply: Whether this post is a reply to another post.
        is_post_unavailable: Whether the post this info belongs to is deleted or private.
        reply_to_author: User this post is directly replying to, if any.
        root_post_author: Author of the post at the top of this post's thread.
        share_info: What this post quotes or reposts.
    """

    direct_reply_count: int | None = Field(default=None, description="Number of direct replies")
    repost_count: int | None = Field(default=None, description="Number of reposts")
    quote_count: int | None = Field(default=None, description="Number of quote posts")
    reshare_count: int | None = Field(default=None, description="Total reshare count")
    text_fragments: TextFragments | None = Field(
        default=None, description="Structured text fragments with links/mentions"
    )
    link_preview_attachment: LinkPreviewAttachment | None = Field(
        default=None, description="Preview metadata for shared links"
    )
    linked_inline_media: LinkedInlineMedia | None = Field(
        default=None, description="Inline media attached through a link preview"
    )
    is_reply: bool | None = Field(
        default=None, description="True when this post is a reply to another post"
    )
    # Declared nullable like every other optional scalar here rather than `bool` with a False
    # default: Threads serialises an absent optional as an explicit null, and `_ThreadsModel`
    # coerces one only on `str` fields, so a plain `bool` would raise and take the whole thread
    # node down with it (`_collect_threads` drops a node that fails validation).
    is_post_unavailable: bool | None = Field(
        default=None, description="True when this post is deleted or private"
    )
    reply_to_author: User | None = Field(
        default=None, description="User this post is directly replying to"
    )
    root_post_author: User | None = Field(
        default=None, description="Author of the post at the top of this post's thread"
    )
    share_info: ShareInfo | None = Field(
        default=None, description="What this post quotes or reposts"
    )


class Post(MediaContainer):
    """Represents a single Threads post parsed from the API JSON.

    Attributes:
        code: Post short code used in URLs.
        caption: Post caption.
        user: Post author.
        text_post_app_info: Threads-specific post info and engagement metrics.
        like_count: Number of likes.
        taken_at: Post creation timestamp as a Unix epoch.
    """

    code: str = Field(default="", description="Post short code used in URLs")
    caption: Caption | None = Field(default=None, description="Post caption")
    user: User | None = Field(default=None, description="Post author")
    text_post_app_info: TextPostAppInfo | None = Field(
        default=None, description="Threads-specific post info and engagement metrics"
    )
    like_count: int | None = Field(default=None, description="Number of likes")
    taken_at: int | None = Field(default=None, description="Post creation timestamp (Unix epoch)")

    @property
    def caption_text(self) -> str:
        """The extracted caption text or fallback link preview title.

        Returns:
            Structured text fragments, caption text, link preview title, or an
            empty string in that priority order.
        """
        if self.text_post_app_info and self.text_post_app_info.text_fragments:
            fragments_text = "".join(
                f.plaintext for f in self.text_post_app_info.text_fragments.fragments
            )
            if fragments_text:
                return fragments_text
        if self.caption and self.caption.text:
            return self.caption.text
        if self.text_post_app_info and self.text_post_app_info.link_preview_attachment:
            title = self.text_post_app_info.link_preview_attachment.title
            if title:
                return title
        return ""

    @property
    def author_name(self) -> str:
        """The username of the post author.

        Returns:
            Author username, or an empty string when author data is missing.
        """
        return self.user.username if self.user else ""

    @property
    def author_icon_url(self) -> str:
        """The profile picture URL of the post author.

        Returns:
            Author profile picture URL, or an empty string when author data is
            missing.
        """
        return self.user.profile_pic_url if self.user else ""

    @property
    def reply_count(self) -> int:
        """The number of direct replies to the post.

        Returns:
            Direct reply count, or 0 when engagement data is missing.
        """
        return (self.text_post_app_info.direct_reply_count or 0) if self.text_post_app_info else 0

    @property
    def repost_count(self) -> int:
        """The number of reposts.

        Returns:
            Repost count, or 0 when engagement data is missing.
        """
        return (self.text_post_app_info.repost_count or 0) if self.text_post_app_info else 0

    @property
    def quote_count(self) -> int:
        """The number of quote posts.

        Returns:
            Quote post count, or 0 when engagement data is missing.
        """
        return (self.text_post_app_info.quote_count or 0) if self.text_post_app_info else 0

    @property
    def reshare_count(self) -> int:
        """The total reshare count.

        Returns:
            Reshare count, or 0 when engagement data is missing.
        """
        return (self.text_post_app_info.reshare_count or 0) if self.text_post_app_info else 0

    @property
    def is_reply(self) -> bool:
        """Whether this post is a reply to another post.

        Returns:
            True when `text_post_app_info.is_reply` is set; False otherwise.
        """
        return bool(self.text_post_app_info and self.text_post_app_info.is_reply)

    @property
    def reply_to_username(self) -> str:
        """The username of the post being directly replied to.

        Returns:
            Username from `text_post_app_info.reply_to_author`, or an empty
            string when this post is not a reply or the field is missing.
        """
        if self.text_post_app_info and self.text_post_app_info.reply_to_author:
            return self.text_post_app_info.reply_to_author.username
        return ""

    @property
    def root_post_username(self) -> str:
        """The username of the author of the post at the top of this post's thread.

        Returns:
            Username from `text_post_app_info.root_post_author`, or an empty string when the
            field is missing (as it is on a post that is not itself a reply).
        """
        if self.text_post_app_info and self.text_post_app_info.root_post_author:
            return self.text_post_app_info.root_post_author.username
        return ""

    @property
    def quoted_post(self) -> "Post | None":
        """The post this one quotes, as the whole payload Threads ships for it.

        Returns:
            The quoted post, or None when this post quotes nothing.
        """
        if self.text_post_app_info and self.text_post_app_info.share_info:
            return self.text_post_app_info.share_info.quoted_post
        return None

    @property
    def is_quote_post(self) -> bool:
        """Whether this post embeds another post as a quote.

        Returns:
            True when `text_post_app_info.share_info.quoted_post` is present.
        """
        return self.quoted_post is not None

    @property
    def is_unavailable(self) -> bool:
        """Whether Threads reports this post itself as deleted or private.

        This is the ONLY tell for a quoted post that is gone, and it is not the one the field
        names suggest. Measured live: `share_info.quoted_attachment_post_unavailable` was False
        in all 465 nodes sampled, INCLUDING all 15 whose quoted post was genuinely gone — it
        belongs to the mutually exclusive `quoted_attachment_post` family, not to `quoted_post`.
        The gone state instead arrives as a tombstone inside `quoted_post` itself: every field
        null bar `id`, `pk` and a `text_post_app_info` holding only this flag. 15 of 96 quote
        relations were that shape, so it is an ordinary outcome, not an exotic one.

        Returns:
            True when `text_post_app_info.is_post_unavailable` is set.
        """
        return bool(self.text_post_app_info and self.text_post_app_info.is_post_unavailable)

    @property
    def is_readable(self) -> bool:
        """Whether this payload holds enough of a post to render at all.

        Both halves earn their keep on the tombstone above: the flag is the platform's own
        statement, and the content test catches a payload that carries nothing to show whether
        or not the flag came with it. A tombstone yields no username and no shortcode either, so
        there is not even a permalink to fall back on.

        Returns:
            True when the post is not reported unavailable and has an author, code, text or
            media to render.
        """
        if self.is_unavailable:
            return False
        return bool(self.author_name or self.code or self.caption_text or self.media_urls)

    @property
    def media_urls(self) -> list[str]:
        """The list of media URLs, including inline media and link preview images.

        Returns:
            Deduplicated non-empty media URLs from post media, linked inline
            media, or the link preview image fallback.
        """
        urls = super().media_urls
        app_info = self.text_post_app_info
        if app_info and app_info.linked_inline_media:
            urls.extend(app_info.linked_inline_media.media_urls)
        if not urls and app_info and app_info.link_preview_attachment:
            urls.append(app_info.link_preview_attachment.image_url)
        return [u for u in dict.fromkeys(urls) if u]


class ThreadItem(_ThreadsModel):
    """Represents one item in a Threads reply chain.

    Attributes:
        post: Parsed post for this thread item.
    """

    post: Post | None = Field(default=None, description="Parsed post for this thread item")


class ThreadData(_ThreadsModel):
    """Represents one entry of a Threads post page: a thread, or a section header between them.

    Attributes:
        thread_items: Ordered thread items from the embedded JSON.
        header: Section label; only a section-marker entry carries one.
        thread_type: `thread` for a real thread, `header` for a section marker.
    """

    thread_items: list[ThreadItem] = Field(
        default_factory=list, description="Ordered thread items from the embedded JSON"
    )
    header: str = Field(
        default="", description="Section label, e.g. 'More replies to <user>'", examples=[""]
    )
    thread_type: str = Field(
        default="", description="'thread' for a real thread, 'header' for a section marker"
    )

    @property
    def is_section_header(self) -> bool:
        """Whether this entry marks the start of a new section rather than holding a thread.

        Returns:
            True when the entry carries a section label or a non-thread type.
        """
        return bool(self.header) or (self.thread_type not in ("", "thread"))

    @property
    def posts(self) -> list[Post]:
        """The parsed posts of this thread, oldest first.

        Returns:
            Every item's post with the empty items dropped.
        """
        return [item.post for item in self.thread_items if item.post]

    def find_post_with_parents(self, post_code: str) -> tuple[Post | None, list[Post]]:
        """Returns the matching post and the chronologically-ordered ancestors before it.

        Threads stores an entire reply chain (root → direct parent → target) in a single
        `thread_items` list, oldest first. Everything appearing before the target item is
        therefore an ancestor of it.

        Args:
            post_code: The short code of the target post.

        Returns:
            A tuple containing:
                - The matching Post instance if found, else None.
                - A list of ancestor Post instances, ordered oldest to newest.
        """
        for index, item in enumerate(self.thread_items):
            if item.post and item.post.code == post_code:
                parents = [t.post for t in self.thread_items[:index] if t.post]
                return item.post, parents
        return None, []


class ThreadsPage(BaseModel):
    """Everything one Threads post page yielded, as raw posts.

    A post page embeds several threads in the same JSON block: one holding the chain that ends
    at the target, and one per branch of replies below it. This is that block, split into the
    two parts the callers actually want.

    Attributes:
        chain: The chain ending at the target, ordered `[root, ..., parent, target]`; empty
            when the page carried no such post.
        reply_branches: One list per reply branch under the target, each ordered from the
            direct reply outward, so an item's index in its branch is its nesting depth.
    """

    chain: list[Post] = Field(
        default_factory=list, description="The chain ending at the target, root first"
    )
    reply_branches: list[list[Post]] = Field(
        default_factory=list,
        description="One list per reply branch under the target, direct reply first",
    )

    @property
    def target(self) -> Post | None:
        """The post the URL pointed at.

        Returns:
            The last chain entry, or None when the page carried no such post.
        """
        return self.chain[-1] if self.chain else None


class FetchedPage(BaseModel):
    """One page as it came back: its HTML, and the URL the request actually ended on.

    `final_url` is what makes a `share/<code>` link readable. That form names its post nowhere
    else — its code is unrelated to the post's and appears nowhere in the page — so the redirect
    the fetch already followed is the only thing that names it. Reading it off the response costs
    nothing, while resolving it separately would spend another round trip on the reply pipeline's
    critical path.

    Attributes:
        html: The fetched page's HTML body.
        final_url: The URL the request ended on, after every redirect it followed.
    """

    html: str = Field(..., description="The fetched page's HTML body")
    final_url: str = Field(
        ..., description="The URL the request ended on, after every redirect it followed"
    )


class ParsedPage(BaseModel):
    """One fetched page's outcome: what it yielded, and whether it was an answer at all.

    `carried_post_json` is what separates the platform's soft throttle from a real answer that
    simply does not hold the post. A throttled fetch comes back 200 with a few hundred KB of
    shell carrying no post JSON anywhere; a private, deleted, or mistyped post comes back with
    a full payload (the recommendation rail Threads pads the page with) that just does not
    contain the requested code. Only the first is worth fetching again.

    Attributes:
        page: The posts this page yielded; empty when it held no such post.
        carried_post_json: Whether any script block on the page held post JSON at all.
    """

    page: ThreadsPage = Field(
        ..., description="The posts this page yielded; empty when it held no such post"
    )
    carried_post_json: bool = Field(
        ...,
        description="Whether any script block on the page held post JSON at all",
        examples=[True],
    )


class ThreadsOutput(BaseModel):
    """Output model for a single Threads post.

    Attributes:
        text: Extracted post text.
        url: Source Threads URL.
        image_urls: Image URLs extracted from the post.
        video_urls: Video URLs extracted from the post.
        video_paths: Local paths of downloaded videos.
        author_name: Post author username.
        author_icon_url: Post author profile picture URL.
        reply_to_username: Username this post replies to, if any.
        like_count: Number of likes.
        reply_count: Number of direct replies.
        repost_count: Number of reposts.
        quote_count: Number of quote posts.
        reshare_count: Total reshare count.
        taken_at: Post creation time.
        quoted: The post this one quotes, when it quotes a readable one.
        quoted_unavailable: Whether this post quotes a post Threads reports as gone.
    """

    text: str = Field(default="", description="Extracted post text")
    url: str = Field(default="", description="Source Threads URL")
    image_urls: list[str] = Field(
        default_factory=list, description="Image URLs extracted from the post"
    )
    video_urls: list[str] = Field(
        default_factory=list, description="Video URLs extracted from the post"
    )
    video_paths: list[Path] = Field(
        default_factory=list, description="Local paths of downloaded videos"
    )
    author_name: str = Field(default="", description="Post author username")
    author_icon_url: str = Field(default="", description="Post author profile picture URL")
    reply_to_username: str = Field(
        default="", description="Username this post replies to, empty when it replies to nobody"
    )
    like_count: int = Field(default=0, description="Number of likes")
    reply_count: int = Field(default=0, description="Number of direct replies")
    repost_count: int = Field(default=0, description="Number of reposts")
    quote_count: int = Field(default=0, description="Number of quote posts")
    reshare_count: int = Field(default=0, description="Total reshare count")
    taken_at: datetime | None = Field(default=None, description="Post creation time")
    # Its own media stays URL-only in this walk: `video_paths` is always empty here because
    # `_build_output` never downloads for a quoted post, which is also what keeps `parse` and
    # `parse_metadata` producing equal trees. The expansion links a quoted clip instead of
    # attaching it; the reply pipeline fetches what it wants from these URLs itself, images
    # straight to bytes and a clip into a scratch dir of its own.
    quoted: "ThreadsOutput | None" = Field(
        default=None,
        description="The post this one quotes, absent when it quotes nothing readable",
    )
    quoted_unavailable: bool = Field(
        default=False,
        description="Whether this post quotes a post Threads reports as deleted or private",
        examples=[False],
    )

    def unlink(self) -> None:
        """Deletes downloaded video files for this post."""
        for path in self.video_paths:
            path.unlink(missing_ok=True)


class ThreadsConversation(BaseModel):
    """A parsed Threads post: its reply chain plus the comments underneath it.

    Attributes:
        chain: The chain ending at the linked post, ordered `[root, ..., parent, target]`.
        reply_branches: One list per reply branch under the target, each ordered from the
            direct reply outward, so an item's index in its branch is its nesting depth.
    """

    chain: list[ThreadsOutput] = Field(
        default_factory=list, description="The chain ending at the linked post, root first"
    )
    reply_branches: list[list[ThreadsOutput]] = Field(
        default_factory=list,
        description="One list per reply branch under the target, direct reply first",
    )

    @property
    def target(self) -> ThreadsOutput | None:
        """The linked post itself.

        Returns:
            The last chain entry, or None when the post could not be read.
        """
        return self.chain[-1] if self.chain else None

    @property
    def posts(self) -> list[ThreadsOutput]:
        """Every post the page yielded.

        Returns:
            The chain oldest first, then the replies in page order.
        """
        return [*self.chain, *(post for branch in self.reply_branches for post in branch)]

    def unlink(self) -> None:
        """Deletes every downloaded video file this conversation owns."""
        for post in self.posts:
            post.unlink()


_SJS_PATTERN = re.compile(
    r'<script type="application/json"[^>]*data-sjs>(.*?)</script>', re.DOTALL
)

# Extra attempts spent on a page that came back carrying no post JSON at all, which is the
# platform's soft throttle rather than an answer about the post (see `ParsedPage`). Both entry
# points are one-shot — the user pastes a link or mentions the bot once — so a throttle that a
# second fetch usually clears would otherwise spend a real user-visible failure on nothing. A
# page that DID answer, without the post in it, is never retried: that is what keeps a private
# or deleted post a fast failure instead of a slow one.
THREADS_EMPTY_PAGE_RETRIES = 2

# Pause between those attempts. Short on purpose: the throttle is transient, and this sits on
# the reply pipeline's critical path, ahead of the media fetch.
THREADS_EMPTY_PAGE_RETRY_DELAY_SECONDS = 0.8


class ThreadsDownloader(BaseModel):
    """A downloader for extracting text and media from Threads posts.

    Attributes:
        output_folder: Directory where downloaded media files are written.
    """

    output_folder: str = Field(
        ..., description="Directory where downloaded media files are written"
    )

    def _fetch_page(self, url: str) -> FetchedPage:
        """Fetches the given URL, returning its HTML and the URL the request ended on.

        Redirects are followed, as they always were, but where they land is now part of the
        result: a `share/<code>` link names its post only there. See `FetchedPage`.
        """
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/html"}
        try:
            response = requests.get(url=url, headers=headers, timeout=THREADS_PAGE_TIMEOUT_SECONDS)
            response.raise_for_status()
            return FetchedPage(html=response.text, final_url=response.url)
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch HTML from {url}: {e}") from e

    @staticmethod
    def _find_thread_nodes(
        obj: dict[str, Any] | list[Any] | str | float | None,
    ) -> list[dict[str, Any]]:
        """Recursively collects every node carrying a `thread_items` list, in document order.

        The enclosing node is what is collected, not the bare list: the page's section markers
        ("More replies to <user>") are nodes with an empty `thread_items` and a `header`, and
        that header is the only signal separating the target's own replies from the unrelated
        posts Threads pads the page with. Document order is the page's own ordering, which is
        what makes the section boundary meaningful.
        """
        results: list[dict[str, Any]] = []
        if isinstance(obj, dict):
            if isinstance(obj.get("thread_items"), list):
                results.append(obj)
            for key, value in obj.items():
                # The items themselves are posts, never nested nodes; descending into them
                # would walk every post payload for nothing.
                if key != "thread_items":
                    results.extend(ThreadsDownloader._find_thread_nodes(obj=value))
        elif isinstance(obj, list):
            for item in obj:
                results.extend(ThreadsDownloader._find_thread_nodes(obj=item))
        return results

    @staticmethod
    def _collect_threads(data: dict[str, Any] | list[Any], post_code: str) -> list[ThreadData]:
        """Builds a ThreadData for every thread node in one parsed SJS payload, in page order.

        Each node is validated on its own so a single malformed branch costs only that branch;
        validating them together would let one unexpected reply payload discard the target too.
        """
        threads: list[ThreadData] = []
        for node in ThreadsDownloader._find_thread_nodes(obj=data):
            try:
                threads.append(ThreadData.model_validate(obj=node))
            except ValidationError:
                logfire.warn(
                    "Threads payload no longer matches the parser schema; skipping one thread",
                    post_code=post_code,
                    _exc_info=True,
                )
        return threads

    @staticmethod
    def _answers_the_target(*, head: Post, target_author: str, root_author: str) -> bool:
        """Whether a branch's first post is a comment on the target rather than page filler.

        The author it answers is the primary test. Threads omits that field on at least one
        real shape though — a reply that is ALSO a quote post comes back with a null
        `reply_to_author` (and a misleading item-level `parent_post_unavailable_reason:
        "default"`, while the parent is alive) — so a second, narrower test catches exactly
        that shape: the post says it is a reply, it quotes another post, and it names the
        target's own thread root as its root. Every one of those has to hold, which is what
        keeps the relaxation from re-opening the filler section the author test guards: a
        recommended post is not a reply, an ordinary comment carries the author it answers,
        and a thread from elsewhere on the page names a different root.

        Args:
            head: The branch's first post.
            target_author: Username of the target post's author.
            root_author: Username of the author of the post at the top of the target's chain.

        Returns:
            True when the branch hangs off the target.
        """
        if head.reply_to_username:
            return head.reply_to_username == target_author
        return bool(
            head.is_reply
            and head.is_quote_post
            and root_author
            and head.root_post_username == root_author
        )

    @staticmethod
    def _collect_reply_branches(
        threads: list[ThreadData], chain_index: int, target_author: str, root_author: str
    ) -> list[list[Post]]:
        """Returns the reply branches under the target, in the order the page ranked them.

        Threads serialises the whole post page into one JSON block: the chain ending at the
        target, then one thread per branch of replies, and then — on a post whose replies do not
        fill the page — a `More replies to <user>` section header followed by replies to a
        DIFFERENT post, the one at the top of the chain. Two independent tells keep those out:

        - The section header ends the target's own replies, so the scan stops at the first one.
          It is the only tell that works when the target is its author's own reply to their own
          post, because the filler then answers the same username the target does.
        - A branch's first post has to answer the target, which rejects the filler whenever those
          two authors differ, plus a sibling reply to the target's own parent and (if Threads ever
          moves them into this block) the recommended posts it keeps in a separate one today.
          `_answers_the_target` owns that second test, including the one real shape Threads
          serialises without naming the author it answers.

        Args:
            threads: Every thread parsed out of the SJS block holding the target, in page order.
            chain_index: Index of the thread holding the target's own chain.
            target_author: Username of the target post's author.
            root_author: Username of the author of the post at the top of the target's chain.

        Returns:
            One list per reply branch, each ordered from the direct reply outward.
        """
        # Without an author there is nothing to match a reply against, and an empty username
        # would match every post whose `reply_to_author` is missing.
        if not target_author:
            return []
        branches: list[list[Post]] = []
        for index, thread in enumerate(threads):
            if thread.is_section_header:
                break
            if index == chain_index:
                continue
            posts = thread.posts
            if posts and ThreadsDownloader._answers_the_target(
                head=posts[0], target_author=target_author, root_author=root_author
            ):
                branches.append(posts)
        return branches

    def _parse_page_from_html(self, html: str, post_code: str) -> ParsedPage:
        """Parses the target post, its ancestors, and its replies from the SJS script tags."""
        carried_post_json = False
        for match in _SJS_PATTERN.finditer(string=html):
            text = match.group(1)
            if "thread_items" not in text:
                continue

            try:
                data = json.loads(s=text)
            except json.JSONDecodeError:
                logfire.debug(
                    "Skipped a non-JSON Threads SJS block", post_code=post_code, _exc_info=True
                )
                continue
            except ValueError:
                # json.loads can also raise a plain ValueError (e.g. the int-string conversion
                # limit); keep the skip so a later SJS block can still yield the post.
                logfire.warn(
                    "Skipped an unparsable Threads SJS block", post_code=post_code, _exc_info=True
                )
                continue

            # Set only once a block actually parsed: the flag means "the server sent a payload we
            # could read", so a truncated block that carries the substring and nothing usable is
            # retried like the throttle it resembles rather than reported as an answer.
            carried_post_json = True
            threads = self._collect_threads(data=data, post_code=post_code)
            for index, thread in enumerate(threads):
                post, parents = thread.find_post_with_parents(post_code=post_code)
                if not post:
                    continue
                chain = [*parents, post]
                return ParsedPage(
                    page=ThreadsPage(
                        chain=chain,
                        reply_branches=self._collect_reply_branches(
                            threads=threads,
                            chain_index=index,
                            target_author=post.author_name,
                            root_author=chain[0].author_name,
                        ),
                    ),
                    carried_post_json=True,
                )

        return ParsedPage(page=ThreadsPage(), carried_post_json=carried_post_json)

    @staticmethod
    def _determine_extension(media_url: str) -> str:
        """Determines the file extension from a media URL."""
        path_lower = urlparse(media_url).path.lower()
        if ".jpg" in path_lower or ".jpeg" in path_lower:
            return "jpg"
        if ".webp" in path_lower:
            return "webp"
        if ".png" in path_lower:
            return "png"
        if ".mp4" in path_lower:
            return "mp4"
        if "video" not in media_url and "mp4" not in media_url:
            return "jpg"
        return "mp4"

    def download_media(self, url: str, filename: str) -> Path | None:
        """Downloads media from the given URL to the output folder.

        Args:
            url: The URL of the media to download.
            filename: The name to save the file as.

        Returns:
            The Path to the downloaded file.

        Raises:
            RuntimeError: If the HTTP fetch fails.
            OSError: If the file cannot be written. A caller that removed the scratch dir gets
                `FileNotFoundError` here, deliberately: see below.
        """
        # `output_folder` is the caller's to create and this never recreates it, which is what
        # turns its removal into the stop signal a cancellation could not deliver: a caller that
        # gives up mid-walk cannot stop the worker thread (`asyncio.to_thread` abandons it), so
        # it removes the directory instead and the open below fails. A `mkdir` here would undo
        # that between two files and quietly rebuild a directory nobody will clean up.
        filepath = Path(self.output_folder) / filename
        try:
            # The CDN serves these signed URLs with any Referer or none (measured), so this
            # only has to stop naming a host the fetch no longer visits.
            headers = {"User-Agent": "Mozilla/5.0", "Referer": f"{_CANONICAL_THREADS_ORIGIN}/"}
            response = requests.get(
                url=url, headers=headers, stream=True, timeout=THREADS_MEDIA_READ_TIMEOUT_SECONDS
            )
            response.raise_for_status()

            with filepath.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return filepath
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download media from {url}: {e}") from e

    def extract_post_data(self, url: str) -> ThreadsPage:
        """Extracts the target post, its parents, and its replies from a Threads URL.

        A `share/<code>` link names no post of its own, so its code comes from where the fetch
        landed instead of from the URL the user pasted; the remaining attempts then go straight
        to that canonical URL rather than through the share hop again. A share link that lands
        anywhere but a post URL is reported unreadable rather than parsed for: the post code
        would be empty, and an empty code matches any post payload serialised without one.

        A fetch that comes back with no post JSON at all is retried, bounded by
        `THREADS_EMPTY_PAGE_RETRIES` and `THREADS_EMPTY_PAGE_RETRY_DEADLINE_SECONDS`: that shape
        is the platform throttling this fingerprint, and a second attempt usually clears it. A
        page that answered without holding the post is returned as-is, so a private or deleted
        post still fails on the first attempt.

        Args:
            url: The raw Threads post URL, canonical or share form.

        Returns:
            The parsed page; its `target` is None when the post could not be found.
        """
        threads_url = ThreadsURL(raw_url=url)
        fetch_url = threads_url.clean_url
        post_code = threads_url.post_code
        deadline = time.monotonic() + THREADS_EMPTY_PAGE_RETRY_DEADLINE_SECONDS
        attempts = 0
        for attempt in range(THREADS_EMPTY_PAGE_RETRIES + 1):
            attempts = attempt + 1
            fetched = self._fetch_page(url=fetch_url)
            if not post_code:
                resolved = ThreadsURL(raw_url=fetched.final_url)
                post_code = resolved.post_code
                if not post_code:
                    logfire.info(
                        "A Threads share link did not lead to a post; treating it as unreadable",
                        url=threads_url.clean_url,
                        # Query stripped: where it landed is the whole signal, and the `?xmt=`
                        # token a share redirect answers with names whoever shared the post.
                        final_url=resolved.clean_url,
                    )
                    return ThreadsPage()
                fetch_url = resolved.clean_url
            parsed = self._parse_page_from_html(html=fetched.html, post_code=post_code)
            if parsed.page.chain or parsed.carried_post_json:
                return parsed.page
            if attempt == THREADS_EMPTY_PAGE_RETRIES or time.monotonic() >= deadline:
                break
            logfire.info(
                "Threads answered without any post JSON; fetching the page again",
                post_code=post_code,
                attempt=attempts,
                html_length=len(fetched.html),
            )
            time.sleep(THREADS_EMPTY_PAGE_RETRY_DELAY_SECONDS)
        logfire.warn(
            "Threads kept answering without any post JSON; treating the post as unreadable",
            post_code=post_code,
            attempts=attempts,
        )
        return ThreadsPage()

    @staticmethod
    def _post_url(post: Post) -> str:
        """Reconstructs a canonical Threads URL from a post's author handle and code."""
        username = post.author_name
        code = post.code
        if username and code:
            return f"{_CANONICAL_THREADS_ORIGIN}/@{username}/post/{code}"
        return ""

    def _build_output(
        self, post: Post, url: str, download: bool, include_quoted: bool = True
    ) -> ThreadsOutput:
        """Builds a ThreadsOutput object from a Post object.

        `include_quoted` is how the quoted post is bounded to ONE level, and one level is what
        the platform serialises rather than a limit chosen here. Measured live: a quote-of-a-quote
        (2 of 96 relations) comes back as a username-only stub — no code, no caption, no counters,
        no media — and the same post was observed full at depth 0 and stubbed at depth 1 on one
        page, so the degradation is a serialisation-depth artifact. Threads does supply a
        `quoted_post_caption` preview string ("<username>: <text>") exactly where that stub
        appears, if a second level is ever worth rendering.
        """
        post_code = post.code or "unknown"
        image_urls: list[str] = []
        video_urls: list[str] = []
        video_paths: list[Path] = []

        for i, media_url in enumerate(post.media_urls):
            ext = self._determine_extension(media_url=media_url)
            if ext == "mp4":
                video_urls.append(media_url)
                if download:
                    filename = f"threads_{post_code}_{i}.{ext}"
                    filepath = self.download_media(url=media_url, filename=filename)
                    if filepath:
                        video_paths.append(filepath)
            else:
                image_urls.append(media_url)

        taken_at = datetime.fromtimestamp(post.taken_at, tz=UTC) if post.taken_at else None

        quoted_post = post.quoted_post if include_quoted else None
        quoted = (
            self._build_output(
                post=quoted_post,
                url=self._post_url(post=quoted_post),
                download=False,
                include_quoted=False,
            )
            if quoted_post is not None and quoted_post.is_readable
            else None
        )

        return ThreadsOutput(
            text=post.caption_text,
            url=url,
            image_urls=image_urls,
            video_urls=video_urls,
            video_paths=video_paths,
            author_name=post.author_name,
            author_icon_url=post.author_icon_url,
            reply_to_username=post.reply_to_username,
            like_count=post.like_count or 0,
            reply_count=post.reply_count,
            repost_count=post.repost_count,
            quote_count=post.quote_count,
            reshare_count=post.reshare_count,
            taken_at=taken_at,
            quoted=quoted,
            # Only ever set when the post DOES quote something and that something came back
            # unreadable, so a post quoting nothing is never reported as quoting a dead post.
            quoted_unavailable=quoted_post is not None and quoted is None,
        )

    def _build_conversation(self, *, url: str, download: bool) -> ThreadsConversation:
        """Fetches the page once and builds every post it yielded.

        The single walk both public entry points share; `download` is the only thing that ever
        differed between them, and it applies to the target alone. Ancestors and replies stay
        metadata-only in both modes: their media is never part of what the callers deliver.

        Args:
            url: The Threads post URL.
            download: Whether to write the target post's videos to `output_folder`.

        Returns:
            The parsed conversation; empty when the post could not be found.
        """
        page = self.extract_post_data(url=url)
        if not page.chain:
            return ThreadsConversation()
        target_index = len(page.chain) - 1
        # Every post gets a reconstructed canonical URL, the target included, and the caller's own
        # URL survives only as its fallback with the query stripped. What lands here is posted in
        # the expansion's embeds and quoted into the reply prompt, and a pasted link carries the
        # sharer with it: a `share/<code>` link and the `?xmt=` token its redirect answers with are
        # both minted per share, so echoing either names whoever sent it to the channel.
        caller_url = ThreadsURL(raw_url=url).clean_url
        chain = [
            self._build_output(
                post=post,
                url=self._post_url(post=post) or (caller_url if index == target_index else ""),
                download=download and index == target_index,
            )
            for index, post in enumerate(page.chain)
        ]
        reply_branches = [
            [
                self._build_output(post=reply, url=self._post_url(post=reply), download=False)
                for reply in branch
            ]
            for branch in page.reply_branches
        ]
        return ThreadsConversation(chain=chain, reply_branches=reply_branches)

    @contextlib.contextmanager
    def parse(self, url: str) -> Generator[ThreadsConversation]:
        """Parses a Threads post URL and yields the conversation, target media included.

        The target post (the chain's last element) has its videos downloaded into
        `output_folder`; nothing else does. Downloaded video files are removed when the
        context manager exits.

        Args:
            url: The Threads post URL.

        Yields:
            The parsed conversation. Its `chain` is empty when no post is found.
        """
        # Once, here, ahead of every fetch, rather than per file inside `download_media`: see
        # that method's comment for why rebuilding it mid-walk would cost a caller who gave up
        # its only way to stop this. A caller already handing over a scratch directory of its
        # own gets a no-op; the standalone use gets a folder it did not have to make.
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        conversation = self._build_conversation(url=url, download=True)
        try:
            yield conversation
        finally:
            conversation.unlink()

    def parse_metadata(self, *, url: str) -> ThreadsConversation:
        """Parses a Threads post URL into the conversation WITHOUT downloading media.

        Mirrors `parse` with `download=False`, so no video is written to disk and there is
        nothing to clean up (not a context manager). The reply pipeline uses this: it fetches the
        media it wants (the target's, plus that of the post it quotes) from the returned URLs
        itself, straight to the answer model.

        Args:
            url: The Threads post URL.

        Returns:
            The parsed conversation; empty when no post is found.
        """
        return self._build_conversation(url=url, download=False)


if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    downloader = ThreadsDownloader(output_folder="./tmp")
    url = "https://www.threads.com/share/DwqmnLALg/"
    with downloader.parse(url=url) as parsed:
        console.print(parsed.chain)
        console.print(parsed.reply_branches)
