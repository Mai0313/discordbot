"""Shared base models and threading helpers for platform readers.

- `PlatformDownloader`: entry point declaring `parse_metadata(url=...)`. Subclasses that write to
  disk declare a required `output_folder` and add a `parse` (returns a conversation) or `download`
  (returns downloaded files) method.
- `PlatformOutput` / `PlatformConversation`: shared post and discussion shapes; single-media
  platforms return a `<Platform>Metadata` instead.
- `LinkableCommentOutput` / `LinkableComments`: specialization for platforms whose URLs can name
  a single comment.
- `thread_branches`: groups comments into `reply_branches` by parent id.

Unimplemented base methods raise `NotImplementedError` rather than using `abc.ABC`.
"""

from datetime import datetime
from functools import cached_property

from pydantic import Field, BaseModel, computed_field


class PlatformDownloader(BaseModel):
    """Reads one platform, given a URL.

    Subclasses that write files declare a required `output_folder` and a `parse` or `download`
    method. Read-only subclasses hold no state and can be reused across callers.
    """

    def parse_metadata(self, url: str) -> BaseModel:
        """Returns the platform's parsed model for `url`, writing nothing to disk.

        The return type narrows per platform to that platform's own `<Platform>Conversation` or
        `<Platform>Metadata`. Synchronous and blocking on every platform here; callers hand it to
        `asyncio.to_thread` themselves.

        Args:
            url (str): The post or video URL to read.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError


class PlatformOutput(BaseModel):
    """One post or comment in the shared cross-platform schema.

    Platform-specific fields live on subclasses rather than here.
    """

    text: str = Field(default="", description="The post or comment body")
    url: str = Field(default="", description="Where it can be read")
    author_name: str = Field(default="", description="The author's handle")
    author_icon_url: str = Field(default="", description="The author's profile picture")
    image_urls: list[str] = Field(default_factory=list, description="Still images it carries")
    video_urls: list[str] = Field(default_factory=list, description="Videos it carries")
    like_count: int = Field(
        default=0, description="Likes, by whatever name the platform gives them"
    )
    comment_count: int = Field(
        default=0,
        description="Direct replies, named for the concept the sources share rather than for any one platform's word for it",
    )
    taken_at: datetime | None = Field(default=None, description="When it was published")

    @computed_field
    @cached_property
    def is_readable(self) -> bool:
        """Returns whether enough came back to be worth showing (`text`, `image_urls`, or `video_urls`).

        A platform whose own payload carries an explicit unavailable flag answers that question
        separately on the model mirroring its schema; the two rules must not be collapsed.
        """
        return bool(self.text or self.image_urls or self.video_urls)


class PlatformConversation[OutputT: PlatformOutput](BaseModel):
    """A post, its ancestor chain, and the discussion under it.

    Generic in the platform's `PlatformOutput` subclass. Every platform exposes the same fields
    (`chain`, `reply_branches`, `selected_comment_id`), leaving inapplicable ones empty.
    """

    chain: list[OutputT] = Field(
        default_factory=list,
        description="The chain ending at the linked post, ordered `[root, ..., parent, target]`",
    )
    reply_branches: list[list[OutputT]] = Field(
        default_factory=list,
        description="One list per reply branch under the target, each ordered from the direct reply outward; only on Threads is an item's index its nesting depth, since `thread_branches` appends sibling replies to the same branch",
    )
    selected_comment_id: str = Field(
        default="",
        description="The comment the URL singled out on a platform whose links can name one, empty when it named none",
    )

    @computed_field
    @cached_property
    def target(self) -> OutputT | None:
        """Returns the linked post (`chain[-1]`), or `None` when empty.

        Cached; callers must treat the conversation as immutable after construction.
        """
        return self.chain[-1] if self.chain else None

    @computed_field
    @cached_property
    def selected_comment(self) -> OutputT | None:
        """Returns the comment the URL singled out, or `None` on this base class.

        Platforms whose URLs can address a single comment subclass `LinkableComments`.
        """
        return None

    @property
    def comments(self) -> list[OutputT]:
        """Returns every reply, flattened out of `reply_branches` in page order.

        Plain property rather than a `computed_field` to avoid duplicating replies in dumps.
        """
        return [reply for branch in self.reply_branches for reply in branch]

    @property
    def posts(self) -> list[OutputT]:
        """Returns every post and reply on the page: `chain` oldest first, then `comments` in page order."""
        return [*self.chain, *self.comments]


class LinkableCommentOutput(PlatformOutput):
    """One post or comment on a platform whose links can name a single comment."""

    comment_id: str = Field(
        default="", description="The comment's own numeric id; empty on the post itself"
    )


class LinkableComments[OutputT: LinkableCommentOutput](PlatformConversation[OutputT]):
    """A conversation whose URL can single out one of its comments by id."""

    @computed_field
    @cached_property
    def selected_comment(self) -> OutputT | None:
        """Returns the comment the URL singled out, or `None` when none was named or preloaded."""
        if not self.selected_comment_id:
            return None
        return next(
            (
                comment
                for comment in self.comments
                if comment.comment_id == self.selected_comment_id
            ),
            None,
        )


def thread_branches[OutputT: PlatformOutput](
    comments: dict[str, OutputT], parents: dict[str, str]
) -> list[list[OutputT]]:
    """Returns `comments` grouped into `reply_branches` lists, each reply behind the comment it answers.

    A comment whose parent has not appeared before it starts a new branch.

    Args:
        comments (dict[str, OutputT]): Every comment, keyed by its own id, in page order.
        parents (dict[str, str]): The id of the comment each reply answers, keyed by the reply's id.
    """
    branches: list[list[OutputT]] = []
    index: dict[str, list[OutputT]] = {}
    for comment_id, comment in comments.items():
        parent = parents.get(comment_id)
        branch = index.get(parent) if parent else None
        if branch is None:
            branch = [comment]
            branches.append(branch)
        else:
            branch.append(comment)
        index[comment_id] = branch
    return branches
