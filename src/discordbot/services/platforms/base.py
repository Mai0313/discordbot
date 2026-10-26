"""Shared base models and threading helpers for platform readers.

Three base classes define the common surface across platforms:

- `PlatformDownloader` is the entry point, and `parse_metadata(url=...)` is the whole of the
  required surface. A platform that writes a file adds a second method of its own; see below.
- `PlatformOutput` holds the common fields a renderer reads without knowing which platform served
  them. Each platform keeps its own extras on top; that is the platform's reality, not drift.
- `PlatformConversation` is a post plus the discussion around it, generic in the output model its
  `chain` holds, so a subclass names its own model once and gets the conversation accessors typed.

`thread_branches` builds that model's `reply_branches` for a platform whose comments name the one
they answer.

`LinkableCommentOutput` and `LinkableComments` specialise the last two for a platform whose links
can name a single comment: the comment's own id, and the lookup that resolves `selected_comment`
from it.

Unimplemented methods raise `NotImplementedError` rather than using `abc.ABC`.

A second download/parse method is deliberately not declared here because the return types and
options differ across platforms. Many platforms write nothing to disk, handing back image URLs
for whoever shows the post to fetch. Those that do write have incompatible signatures: a walk that
downloads as it goes returns its conversation with each file on the post that owns it, while a
plain fetch returns a model naming its files and takes platform-specific options (quality preset,
image cap, pre-parsed metadata, stop signal) alongside `url`. Accordingly, `parse` returns a
parsed conversation and `download` returns files.

A platform whose result is a single piece of media returns a `<Platform>Metadata` instead of a
`<Platform>Conversation` and subclasses neither conversation model.
"""

from datetime import datetime
from functools import cached_property

from pydantic import Field, BaseModel, computed_field


class PlatformDownloader(BaseModel):
    """Reads one platform, given a URL.

    A subclass that writes a file carries a required `output_folder` and a second method of its
    own, which the module docstring has the reason for. One that only reads a page holds no state
    at all, so a single instance serves every caller.
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
    """One post or comment, in the vocabulary every platform here shares.

    What a caller may read without asking which source it holds. A platform carrying more puts it
    on its own subclass. A field only some platforms publish does not belong here at all: it
    lives on the ones that have it, so a caller reading this model is never handed a zero that
    means "this platform does not have the concept".
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
    """A post, whatever leads to it, and whatever hangs off it.

    Generic in the output model it holds so a platform declares its own model once and the
    accessors below come back typed.

    A field a platform cannot fill is still carried rather than dropped, so the accessors mean
    the same thing everywhere: `chain` is a list even where a platform serves no ancestors, and
    `selected_comment_id` is empty on a platform whose replies have URLs of their own.
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
        """Returns the linked post itself (`chain[-1]`), or `None` when it could not be read.

        Cached, which makes "a conversation is built once and never mutated" load-bearing:
        pydantic invalidates a `cached_property` on neither an in-place `chain.append(...)` nor a
        `chain = [...]` assignment, so a mutated conversation keeps answering with its old target.
        Every platform builds both lists as locals and hands them to the constructor finished;
        media downloads run before the object exists, not after.
        """
        return self.chain[-1] if self.chain else None

    @computed_field
    @cached_property
    def selected_comment(self) -> OutputT | None:
        """Returns the comment the URL singled out, or `None` unless the platform can name one.

        Always `None` on the base class because `selected_comment_id` is matched against a
        comment id not carried on `PlatformOutput`; platforms whose links can address a single
        comment subclass `LinkableComments`, which performs the lookup.
        """
        return None

    @property
    def comments(self) -> list[OutputT]:
        """Returns every reply, flattened out of `reply_branches` in page order.

        Exposed as a plain property rather than a computed field because it re-slices data
        `reply_branches` already carries, so serializing it would duplicate every reply in a dump.
        `target` and `selected_comment` instead resolve a pointer that a dump cannot derive on its
        own.
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
        """Returns the comment the URL singled out, or `None` when none was named or preloaded.

        A named comment missing from the page is an ordinary miss rather than an error: a page
        that preloads only the first handful resolves a link to an older comment to `None`, and
        the caller shows the post alone.
        """
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

    A comment whose parent has not appeared before it opens a branch of its own rather than being
    dropped, so a reply the page served without its parent still reaches the caller.

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
