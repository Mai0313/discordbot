import re
import json
import types
from pathlib import Path
from datetime import UTC, datetime
from functools import cached_property
from urllib.parse import urlparse

from rich import get_console
from pydantic import Field, BaseModel, computed_field
import requests

console = get_console()


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


class ThreadsURL(BaseModel):
    """Parses and normalises a Threads post URL."""

    raw_url: str

    @computed_field
    @cached_property
    def clean_url(self) -> str:
        parsed = urlparse(self.raw_url)
        netloc = parsed.netloc
        if netloc in ("www.threads.com", "threads.com"):
            netloc = "www.threads.net"
        return f"{parsed.scheme}://{netloc}{parsed.path}"

    @computed_field
    @cached_property
    def post_code(self) -> str:
        parsed = urlparse(self.raw_url)
        path_parts = parsed.path.strip("/").split("/")
        return path_parts[-1] if path_parts else ""


# ---------------------------------------------------------------------------
# API data models
# ---------------------------------------------------------------------------


class User(BaseModel):
    id: str = Field(default="", description="Full user ID")
    pk: str = Field(default="", description="User primary key")
    username: str = Field(default="", description="Username handle")
    full_name: str = Field(default="", description="Display name")
    profile_pic_url: str = Field(default="", description="Profile picture URL")
    hd_profile_pic_versions: list[dict] | None = Field(
        default=None, description="HD profile picture versions"
    )
    is_verified: bool = Field(default=False, description="Blue-check verification status")
    transparency_label: str | None = Field(default=None, description="Account transparency label")
    transparency_product: str | None = Field(default=None, description="Transparency product type")
    transparency_product_enabled: bool = Field(
        default=False, description="Whether transparency product is enabled"
    )
    has_onboarded_to_text_post_app: bool = Field(
        default=False, description="Whether user has onboarded to Threads"
    )
    text_post_app_is_private: bool = Field(
        default=False, description="Whether the user's Threads account is private"
    )
    friendship_status: dict | None = Field(
        default=None, description="Friendship/follow status with viewer"
    )


class Caption(BaseModel):
    text: str = Field(default="", description="Caption text content")
    pk: str = Field(default="", description="Caption primary key")
    has_translation: bool | None = Field(
        default=None, description="Whether a translation is available"
    )


class VideoVersion(BaseModel):
    url: str = Field(default="", description="Video file URL")


class ImageCandidate(BaseModel):
    url: str = Field(default="", description="Image URL")
    height: int = Field(default=0, description="Image height in pixels")
    width: int = Field(default=0, description="Image width in pixels")


class ImageVersions2(BaseModel):
    candidates: list[ImageCandidate] = Field(
        default_factory=list, description="Available image resolutions"
    )


class CarouselMedia(BaseModel):
    pk: str | None = Field(default=None, description="Carousel item primary key")
    id: str | None = Field(default=None, description="Carousel item full ID")
    code: str | None = Field(default=None, description="Carousel item short code")
    video_versions: list[VideoVersion] | None = Field(
        default=None, description="Available video versions"
    )
    image_versions2: ImageVersions2 | None = Field(
        default=None, description="Available image versions"
    )
    original_height: int | None = Field(
        default=None, description="Original media height in pixels"
    )
    original_width: int | None = Field(default=None, description="Original media width in pixels")
    accessibility_caption: str | None = Field(
        default=None, description="Alt text for accessibility"
    )
    media_type: int | None = Field(default=None, description="Media type (1=image, 2=video)")
    has_audio: bool | None = Field(default=None, description="Whether the media has audio")
    usertags: dict | None = Field(default=None, description="Tagged users in this carousel item")


class LinkFragment(BaseModel):
    url: str = Field(default="", description="Link destination URL")


class MentionFragment(BaseModel):
    username: str = Field(default="", description="Mentioned username")


class Fragment(BaseModel):
    fragment_type: str = Field(
        default="plaintext", description="Type of fragment: plaintext, mention, link, etc."
    )
    plaintext: str = Field(default="", description="Plain text content of the fragment")
    link_fragment: LinkFragment | None = Field(
        default=None, description="Link data if fragment is a link"
    )
    mention_fragment: MentionFragment | None = Field(
        default=None, description="Mention data if fragment is a @mention"
    )
    inline_sticker_fragment: dict | None = Field(default=None, description="Inline sticker data")
    linkified_web_url: str | None = Field(default=None, description="Auto-linkified web URL")
    linkified_in_app_url: str | None = Field(default=None, description="Auto-linkified in-app URL")
    styling_info: dict | None = Field(default=None, description="Text styling/formatting info")


class TextFragments(BaseModel):
    fragments: list[Fragment] = Field(
        default_factory=list, description="Ordered list of text fragments"
    )


class ShareInfo(BaseModel):
    quoted_post: dict | None = Field(default=None, description="Quoted post data")
    reposted_post: dict | None = Field(default=None, description="Reposted post data")
    quoted_attachment_post: dict | None = Field(
        default=None, description="Quoted attachment post data"
    )
    quoted_attachment_post_unavailable: bool = Field(
        default=False, description="Whether the quoted attachment post is unavailable"
    )
    quoted_attachment_author_attribution_allowed: bool = Field(
        default=True, description="Whether author attribution is allowed for quoted attachment"
    )


class ReplyApprovalInfo(BaseModel):
    hidden_reply_reason: str | None = Field(default=None, description="Reason a reply was hidden")
    pending_reply_status: str | None = Field(
        default=None, description="Pending reply approval status"
    )
    pending_reply_count: int | None = Field(default=None, description="Number of pending replies")
    ignored_reply_count: int | None = Field(default=None, description="Number of ignored replies")


class PinnedPostInfo(BaseModel):
    is_pinned_to_parent_post: bool = Field(
        default=False, description="Whether pinned to parent post"
    )
    is_pinned_to_profile: bool = Field(default=False, description="Whether pinned to profile")


class TextPostAppInfo(BaseModel):
    id: str = Field(default="", description="TextPostAppInfo ID")

    # Engagement metrics
    direct_reply_count: int | None = Field(default=None, description="Number of direct replies")
    repost_count: int | None = Field(default=None, description="Number of reposts")
    quote_count: int | None = Field(default=None, description="Number of quote posts")
    reshare_count: int | None = Field(default=None, description="Total reshare count")
    self_thread_count: int | None = Field(
        default=None, description="Number of self-thread replies by the author"
    )

    # Post status flags
    is_reply: bool | None = Field(default=None, description="Whether this post is a reply")
    is_post_unavailable: bool | None = Field(
        default=None, description="Whether the post is unavailable"
    )
    is_ghost_post: bool | None = Field(
        default=None, description="Whether the post is a ghost/shadow post"
    )
    is_spoiler_media: bool | None = Field(
        default=None, description="Whether media is marked as spoiler"
    )
    is_markup: bool | None = Field(
        default=None, description="Whether the post uses markup formatting"
    )
    is_liked_by_root_author: bool | None = Field(
        default=None, description="Whether liked by the root thread author"
    )

    # Reply & permission controls
    reply_control: str = Field(
        default="everyone",
        description="Who can reply: everyone, accounts_you_follow, mentioned_only",
    )
    can_reply: bool | None = Field(default=None, description="Whether the viewer can reply")
    can_private_reply: bool | None = Field(
        default=None, description="Whether the viewer can send a private reply"
    )
    is_reply_approval_enabled: bool | None = Field(
        default=None, description="Whether reply approval is enabled"
    )
    reply_approval_info: ReplyApprovalInfo | None = Field(
        default=None, description="Reply approval details"
    )
    show_header_follow: bool | None = Field(
        default=None, description="Whether to show follow button in header"
    )

    # Content data
    text_fragments: TextFragments | None = Field(
        default=None, description="Structured text fragments with links/mentions"
    )
    share_info: ShareInfo | None = Field(default=None, description="Quoted/reposted post info")
    pinned_post_info: PinnedPostInfo | None = Field(default=None, description="Pin status")

    # Author context
    reply_to_author: User | None = Field(
        default=None, description="Author of the post being replied to"
    )
    root_post_author: User | None = Field(default=None, description="Original thread author")
    private_reply_partner: dict | None = Field(
        default=None, description="Private reply partner info"
    )
    author_notif_control: dict | None = Field(
        default=None, description="Author notification control settings"
    )

    # Attachments & previews
    link_preview_attachment: dict | None = Field(
        default=None, description="Link preview card attachment"
    )
    link_preview_response: dict | None = Field(
        default=None, description="Link preview response data"
    )
    linked_inline_media: dict | None = Field(default=None, description="Inline linked media")
    snippet_attachment_info: dict | None = Field(
        default=None, description="Snippet attachment info"
    )
    attachment_tombstone_info: dict | None = Field(
        default=None, description="Tombstone info for removed attachments"
    )

    # Ghost post details
    ghost_post_exp_time_ms: int | None = Field(
        default=None, description="Ghost post expiration time in ms"
    )
    ghost_post_approximate_like_count_str: str | None = Field(
        default=None, description="Approximate like count for ghost posts"
    )
    ghost_post_reply_type: str | None = Field(default=None, description="Ghost post reply type")
    ghost_post_approximate_reply_count_str: str | None = Field(
        default=None, description="Approximate reply count for ghost posts"
    )

    # Miscellaneous metadata
    hush_info: dict | None = Field(default=None, description="Hush/mute info")
    self_thread_info: dict | None = Field(default=None, description="Self-thread context info")
    tag_header: dict | None = Field(default=None, description="Tag header info")
    system_status_message: str | None = Field(default=None, description="System status message")
    post_unavailable_reason: str | None = Field(
        default=None, description="Reason the post is unavailable"
    )
    post_tombstone_info: dict | None = Field(
        default=None, description="Tombstone info for deleted posts"
    )
    related_trends_info: dict | None = Field(default=None, description="Related trending topics")
    custom_feed_preview_info: dict | None = Field(
        default=None, description="Custom feed preview data"
    )
    special_effects_enabled_str: str = Field(
        default="", description="Special effects enabled string"
    )
    algo_tweaks_info: dict | None = Field(default=None, description="Algorithm tweaks info")

    # Platform-specific
    platform_podcast_episode_info: dict | None = Field(
        default=None, description="Podcast episode info"
    )
    platform_podcast_info: dict | None = Field(default=None, description="Podcast info")
    game_score_share_info: dict | None = Field(default=None, description="Game score sharing info")
    public_view_count_card_attachment_info: dict | None = Field(
        default=None, description="Public view count card info"
    )


class Post(BaseModel):
    """Represents a single Threads post parsed from the API JSON."""

    # Identifiers
    pk: str = Field(default="", description="Post primary key")
    id: str = Field(default="", description="Full post ID (pk_userPk format)")
    code: str = Field(default="", description="Post short code used in URLs")

    # Content
    caption: Caption | None = Field(default=None, description="Post caption")
    user: User | None = Field(default=None, description="Post author")
    text_post_app_info: TextPostAppInfo | None = Field(
        default=None, description="Threads-specific post info and engagement metrics"
    )

    # Media
    media_type: int | None = Field(
        default=None, description="Media type (1=image, 2=video, 8=carousel)"
    )
    carousel_media: list[CarouselMedia] | None = Field(
        default=None, description="Carousel media items"
    )
    video_versions: list[VideoVersion] | None = Field(
        default=None, description="Available video versions"
    )
    image_versions2: ImageVersions2 | None = Field(
        default=None, description="Available image versions"
    )
    original_height: int | None = Field(
        default=None, description="Original media height in pixels"
    )
    original_width: int | None = Field(default=None, description="Original media width in pixels")
    has_audio: bool | None = Field(default=None, description="Whether the media has audio")
    accessibility_caption: str | None = Field(
        default=None, description="Alt text for accessibility"
    )

    # Engagement
    like_count: int | None = Field(default=None, description="Number of likes")
    like_and_view_counts_disabled: bool | None = Field(
        default=None, description="Whether like/view counts are hidden"
    )

    # Metadata
    taken_at: int | None = Field(default=None, description="Post creation timestamp (Unix epoch)")
    caption_is_edited: bool | None = Field(
        default=None, description="Whether the caption has been edited"
    )
    is_paid_partnership: bool | None = Field(
        default=None, description="Whether this is a paid partnership/sponsored post"
    )
    gen_ai_detection_method: dict | None = Field(
        default=None, description="AI-generated content detection method"
    )
    canonical_url: str | None = Field(default=None, description="Canonical URL of the post")
    organic_tracking_token: str | None = Field(
        default=None, description="Analytics tracking token"
    )

    # Additional media info
    audio: dict | None = Field(default=None, description="Audio attachment data")
    transcription_data: dict | None = Field(
        default=None, description="Audio/video transcription data"
    )
    usertags: dict | None = Field(default=None, description="Tagged users in media")
    giphy_media_info: dict | None = Field(default=None, description="GIPHY sticker/GIF info")
    media_overlay_info: dict | None = Field(default=None, description="Media overlay info")
    caption_add_on: dict | None = Field(default=None, description="Caption add-on data")

    # Platform flags
    is_fb_only: bool | None = Field(default=None, description="Whether post is Facebook-only")
    is_internal_only: bool | None = Field(
        default=None, description="Whether post is internal-only"
    )

    # -- derived properties ---------------------------------------------------

    @property
    def caption_text(self) -> str:
        if self.text_post_app_info and self.text_post_app_info.text_fragments:
            fragments_text = "".join(
                f.plaintext for f in self.text_post_app_info.text_fragments.fragments
            )
            if fragments_text:
                return fragments_text
        return self.caption.text if self.caption else ""

    @property
    def author_name(self) -> str:
        return self.user.username if self.user else ""

    @property
    def author_icon_url(self) -> str:
        return self.user.profile_pic_url if self.user else ""

    @property
    def reply_count(self) -> int:
        return (self.text_post_app_info.direct_reply_count or 0) if self.text_post_app_info else 0

    @property
    def repost_count(self) -> int:
        return (self.text_post_app_info.repost_count or 0) if self.text_post_app_info else 0

    @property
    def quote_count(self) -> int:
        return (self.text_post_app_info.quote_count or 0) if self.text_post_app_info else 0

    @property
    def reshare_count(self) -> int:
        return (self.text_post_app_info.reshare_count or 0) if self.text_post_app_info else 0

    @property
    def media_urls(self) -> list[str]:
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


# ---------------------------------------------------------------------------
# HTML → Post extraction models
# ---------------------------------------------------------------------------


class ThreadItem(BaseModel):
    post: Post | None = Field(default=None)


class ThreadData(BaseModel):
    thread_items: list[ThreadItem] = Field(default_factory=list)

    def find_post(self, post_code: str) -> Post | None:
        for item in self.thread_items:
            if item.post and item.post.code == post_code:
                return item.post
        return None


# ---------------------------------------------------------------------------
# Output model (public API — fields unchanged)
# ---------------------------------------------------------------------------


class ThreadsOutput(BaseModel):
    """Output model for Threads downloader."""

    text: str = Field(default="")
    url: str = Field(default="")
    image_urls: list[str] = Field(default=[])
    video_urls: list[str] = Field(default_factory=list)
    video_paths: list[Path] = Field(default=[])
    author_name: str = Field(default="")
    author_icon_url: str = Field(default="")
    like_count: int = Field(default=0)
    reply_count: int = Field(default=0)
    repost_count: int = Field(default=0)
    quote_count: int = Field(default=0)
    reshare_count: int = Field(default=0)
    taken_at: datetime | None = Field(default=None, description="Post creation time")

    def unlink(self) -> None:
        for path in self.video_paths:
            path.unlink(missing_ok=True)

    def __enter__(self):  # noqa: D105
        return self

    def __exit__(  # noqa: D105
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ):
        self.unlink()


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

_SJS_PATTERN = re.compile(
    r'<script type="application/json"[^>]*data-sjs>(.*?)</script>', re.DOTALL
)


class ThreadsDownloader(BaseModel):
    """A downloader for extracting text and media from Threads.net posts."""

    output_folder: str = Field(default="./data/threads")

    # -- HTTP -----------------------------------------------------------------

    def _fetch_html(self, url: str) -> str:
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/html"}
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch HTML from {url}: {e}") from e

    # -- HTML parsing ---------------------------------------------------------

    @staticmethod
    def _find_keys(obj: dict | list | str | float | None, key: str) -> list:
        results: list = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key and isinstance(v, list | dict):
                    results.append(v)
                else:
                    results.extend(ThreadsDownloader._find_keys(v, key))
        elif isinstance(obj, list):
            for item in obj:
                results.extend(ThreadsDownloader._find_keys(item, key))
        return results

    def _parse_post_from_html(self, html: str, post_code: str) -> Post | None:
        for match in _SJS_PATTERN.finditer(html):
            text = match.group(1)
            if "thread_items" not in text:
                continue

            try:
                data = json.loads(text)
                raw_lists = self._find_keys(data, "thread_items")
                for raw_items in raw_lists:
                    if not isinstance(raw_items, list):
                        continue
                    thread_data = ThreadData(thread_items=raw_items)
                    post = thread_data.find_post(post_code)
                    if post:
                        return post
            except (json.JSONDecodeError, ValueError):
                continue

        return None

    # -- Media download -------------------------------------------------------

    @staticmethod
    def _determine_extension(media_url: str) -> str:
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
        try:
            response = requests.get(url, stream=True, timeout=15)
            response.raise_for_status()

            filepath = Path(self.output_folder) / filename
            with Path.open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return filepath
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download media from {url}: {e}") from e

    # -- Public API -----------------------------------------------------------

    def extract_post_data(self, url: str) -> Post | None:
        threads_url = ThreadsURL(raw_url=url)
        html = self._fetch_html(threads_url.clean_url)
        return self._parse_post_from_html(html, threads_url.post_code)

    def parse(self, url: str) -> ThreadsOutput:
        post = self.extract_post_data(url)
        if not post:
            return ThreadsOutput()

        post_code = post.code or "unknown"
        image_urls: list[str] = []
        video_urls: list[str] = []
        video_paths: list[Path] = []

        for i, media_url in enumerate(post.media_urls):
            ext = self._determine_extension(media_url)
            if ext == "mp4":
                video_urls.append(media_url)
                filename = f"threads_{post_code}_{i}.{ext}"
                filepath = self.download_media(media_url, filename)
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
            like_count=post.like_count or 0,
            reply_count=post.reply_count,
            repost_count=post.repost_count,
            quote_count=post.quote_count,
            reshare_count=post.reshare_count,
            taken_at=taken_at,
        )


if __name__ == "__main__":
    test_url = "https://www.threads.com/@show4653/post/DWYp35uGh4l"
    # test_url = "https://www.threads.com/@cyj308/post/DVn6dqzjzQf?hl=zh-tw"
    downloader = ThreadsDownloader()
    with downloader.parse(test_url) as result:
        console.print(result)
