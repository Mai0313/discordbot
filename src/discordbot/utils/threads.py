"""Threads URL parsing, API models, and media download helpers."""

import re
import json
from typing import Any
from pathlib import Path
from datetime import UTC, datetime
from functools import cached_property
import contextlib
from urllib.parse import urlparse
from collections.abc import Iterator

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

# Single source of truth for detecting a Threads post URL, shared by the parse_threads
# cog (which expands it into embeds) and gen_reply (which self-parses it into answer
# context). Matches `@user/post/<code>` on both threads.net and threads.com. The shortcode +
# query tail is matched as ASCII URL characters only and must END on `[A-Za-z0-9_-]` (the only
# characters a valid Threads code or query value ends in). Restricting to ASCII stops the match
# at any non-ASCII terminator, and the trailing class strips ASCII sentence punctuation, so a
# link written mid-sentence is matched cleanly in both English (`.../post/ABC123.`) and zh/ja
# (`...ABC123。`, `...ABC123】super`) text instead of swallowing the terminator into the code,
# which would otherwise make the parse fail on an otherwise valid link.
THREADS_URL_RE = re.compile(
    r"https?://(?:www\.)?threads\.(?:net|com)/@[^/]+/post/[A-Za-z0-9_.?=&%-]*[A-Za-z0-9_-]"
)


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

    Attributes:
        raw_url: Original Threads URL provided by the caller.
    """

    raw_url: str = Field(..., description="Original Threads URL provided by the caller")

    @computed_field
    @cached_property
    def clean_url(self) -> str:
        """The cleaned and normalised URL.

        Returns:
            URL with `threads.com` hosts normalised to `www.threads.net` and
            query parameters removed.
        """
        parsed = urlparse(self.raw_url)
        netloc = parsed.netloc
        if netloc in ("www.threads.com", "threads.com"):
            netloc = "www.threads.net"
        return f"{parsed.scheme}://{netloc}{parsed.path}"

    @computed_field
    @cached_property
    def post_code(self) -> str:
        """The post short code extracted from the URL.

        Returns:
            The last path segment from the raw URL, or an empty string for an
            empty path.
        """
        parsed = urlparse(self.raw_url)
        path_parts = parsed.path.strip("/").split("/")
        return path_parts[-1] if path_parts else ""


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
        reply_to_author: User this post is directly replying to, if any.
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
    reply_to_author: User | None = Field(
        default=None, description="User this post is directly replying to"
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


class ThreadsDownloader(BaseModel):
    """A downloader for extracting text and media from Threads.net posts.

    Attributes:
        output_folder: Directory where downloaded media files are written.
    """

    output_folder: str = Field(
        ..., description="Directory where downloaded media files are written"
    )

    def _fetch_html(self, url: str) -> str:
        """Fetches the HTML content of the given URL."""
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/html"}
        try:
            response = requests.get(url=url, headers=headers, timeout=15)
            response.raise_for_status()
            return response.text
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
    def _collect_reply_branches(
        threads: list[ThreadData], chain_index: int, target_author: str
    ) -> list[list[Post]]:
        """Returns the reply branches under the target, in the order the page ranked them.

        Threads serialises the whole post page into one JSON block: the chain ending at the
        target, then one thread per branch of replies, and then — on a post whose replies do not
        fill the page — a `More replies to <user>` section header followed by replies to a
        DIFFERENT post, the one at the top of the chain. Two independent tells keep those out:

        - The section header ends the target's own replies, so the scan stops at the first one.
          It is the only tell that works when the target is its author's own reply to their own
          post, because the filler then answers the same username the target does.
        - A branch's first post carries the author it answers, which rejects the filler whenever
          those two authors differ, plus a sibling reply to the target's own parent and (if
          Threads ever moves them into this block) the recommended posts it keeps in a separate
          one today.

        The known cost of the author test is a direct reply that Threads serialises with a null
        `reply_to_author` — observed on a reply that is also a quote post. Dropping one real
        comment is the safe side of that trade: attributing a stranger's comment to the wrong
        post is a mistake the model would then repeat as fact.

        Args:
            threads: Every thread parsed out of the SJS block holding the target, in page order.
            chain_index: Index of the thread holding the target's own chain.
            target_author: Username of the target post's author.

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
            if posts and posts[0].reply_to_username == target_author:
                branches.append(posts)
        return branches

    def _parse_page_from_html(self, html: str, post_code: str) -> ThreadsPage:
        """Parses the target post, its ancestors, and its replies from the SJS script tags."""
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

            threads = self._collect_threads(data=data, post_code=post_code)
            for index, thread in enumerate(threads):
                post, parents = thread.find_post_with_parents(post_code=post_code)
                if not post:
                    continue
                return ThreadsPage(
                    chain=[*parents, post],
                    reply_branches=self._collect_reply_branches(
                        threads=threads, chain_index=index, target_author=post.author_name
                    ),
                )

        return ThreadsPage()

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
            RuntimeError: If the download fails.
        """
        # Created before the request, never after: a caller that cancels mid-download cannot stop
        # the worker thread (`asyncio.to_thread` abandons it), so it may remove the scratch dir
        # underneath this. Creating it up front lets the open fail instead of quietly rebuilding
        # a directory nobody will clean up, which turns the removal into the stop signal the
        # cancellation could not deliver.
        filepath = Path(self.output_folder) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.threads.net/"}
            response = requests.get(url=url, headers=headers, stream=True, timeout=15)
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

        Args:
            url: The raw Threads post URL.

        Returns:
            The parsed page; its `target` is None when the post could not be found.
        """
        threads_url = ThreadsURL(raw_url=url)
        html = self._fetch_html(url=threads_url.clean_url)
        return self._parse_page_from_html(html=html, post_code=threads_url.post_code)

    @staticmethod
    def _post_url(post: Post) -> str:
        """Reconstructs a canonical Threads URL from a post's author handle and code."""
        username = post.author_name
        code = post.code
        if username and code:
            return f"https://www.threads.com/@{username}/post/{code}"
        return ""

    def _build_output(self, post: Post, url: str, download: bool) -> ThreadsOutput:
        """Builds a ThreadsOutput object from a Post object."""
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
        chain = [
            self._build_output(
                post=post,
                # The caller's own URL for the target; a reconstructed one for the ancestors.
                url=url if index == target_index else self._post_url(post=post),
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
    def parse(self, url: str) -> Iterator[ThreadsConversation]:
        """Parses a Threads post URL and yields the conversation, target media included.

        The target post (the chain's last element) has its videos downloaded into
        `output_folder`; nothing else does. Downloaded video files are removed when the
        context manager exits.

        Args:
            url: The Threads post URL.

        Yields:
            The parsed conversation. Its `chain` is empty when no post is found.
        """
        conversation = self._build_conversation(url=url, download=True)
        try:
            yield conversation
        finally:
            conversation.unlink()

    def parse_metadata(self, *, url: str) -> ThreadsConversation:
        """Parses a Threads post URL into the conversation WITHOUT downloading media.

        Mirrors `parse` with `download=False`, so no video is written to disk and there is
        nothing to clean up (not a context manager). The reply pipeline uses this: it fetches
        the target's media itself, straight to the answer model.

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
    url = "https://www.threads.com/@chengweilai2/post/DZZImVsCWU-?xmt=AQG0MLHN7M4RJOdbdF1HzSG5Qm-9a1b2aOB5HN4ksjCrhQ"
    with downloader.parse(url=url) as parsed:
        console.print(parsed.chain)
        console.print(parsed.reply_branches)
