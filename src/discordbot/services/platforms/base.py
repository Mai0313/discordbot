"""What every platform in this package has in common, and nothing else.

Three bases, each answering a question that used to be answered once per platform:

- `PlatformDownloader` is the entry point, and `parse_metadata(*, url=...)` is the whole of the
  required surface. A platform that writes a file adds a second method of its own; see below.
- `PlatformOutput` is the nine fields a renderer reads without knowing which platform served
  them. Each platform keeps its own extras on top; that is the platform's reality, not drift.
- `PlatformConversation` is a post plus the discussion around it, generic in the Output its
  `chain` holds, so a subclass names its own model once and gets the four accessors typed.

`raise NotImplementedError` rather than `abc.ABC`, matching `TemporaryDownload.unlink` one module
over: this repo's answer to an unimplemented member is a raise, and an ABC would catch only the
forgotten override while saying nothing about the signature — which is the half that actually
drifted. `tests/test_platform_shape.py` is what holds the signature.

**A second method is deliberately NOT declared here, and the reason is the return type.** Many
platforms write nothing to disk at all, handing image URLs to Discord to fetch itself. The ones
that do write have nothing a base could hold them to: a walk that downloads as it goes yields
its conversation from a context manager, because what it has to clean up hangs off that
conversation, while a plain fetch returns a `TemporaryDownload` and takes its own options (a
quality preset, an image cap, an already-parsed post, a stop signal) alongside the url. A shared
declaration would have to be the union of all that or the intersection, and the intersection is
empty. So the name differs too, on purpose: `parse` hands back a parsed conversation, `download`
hands back files.

A platform whose result is a single piece of media returns a `<Platform>Metadata` instead of a
`<Platform>Conversation` and takes neither of the two models below. There is no base for that
half, and inventing one would mean a `chain` of length one and a `reply_branches` nothing can
ever fill — the same reason a field no platform can populate does not belong in the shared
output at all.
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

    def parse_metadata(self, *, url: str) -> BaseModel:
        """Parses what is at `url`, writing nothing to disk.

        The return type narrows per platform to that platform's own `<Platform>Conversation` or
        `<Platform>Metadata`. Synchronous and blocking on every platform here — callers hand it to
        `asyncio.to_thread` themselves rather than this layer choosing a concurrency model for
        them.

        Args:
            url: The post or video URL to read.

        Returns:
            The platform's own parsed model.

        Raises:
            NotImplementedError: Always; a platform module overrides this.
        """
        raise NotImplementedError


class PlatformOutput(BaseModel):
    """One post or comment, in the vocabulary every platform here shares.

    What a caller may read without asking which source it holds. A platform carrying more puts it
    on its own subclass. A field only some platforms publish does not belong here at all: it
    lives on the ones that have it, so a caller reading this model is never handed a zero that
    means "this platform does not have the concept".

    Attributes:
        text: The post or comment body.
        url: Where it can be read.
        author_name: The author's handle.
        author_icon_url: The author's profile picture.
        image_urls: Still images it carries.
        video_urls: Videos it carries.
        like_count: Likes, by whatever name the platform gives them.
        comment_count: Direct replies, named for the concept the sources share rather than for
            any one platform's word for it.
        taken_at: When it was published.
    """

    text: str = Field(default="", description="The post or comment body")
    url: str = Field(default="", description="Where it can be read")
    author_name: str = Field(default="", description="The author's handle")
    author_icon_url: str = Field(default="", description="The author's profile picture")
    image_urls: list[str] = Field(default_factory=list, description="Still images it carries")
    video_urls: list[str] = Field(default_factory=list, description="Videos it carries")
    like_count: int = Field(default=0, description="Likes, by whatever name the platform gives")
    comment_count: int = Field(default=0, description="Number of direct replies")
    taken_at: datetime | None = Field(default=None, description="When it was published")

    @computed_field
    @cached_property
    def is_readable(self) -> bool:
        """Whether enough came back to be worth showing.

        Text or media. A platform whose own payload carries an explicit unavailable flag answers
        that question separately and further up, on the model mirroring its schema; the two rules
        are not the same and must not be collapsed.
        """
        return bool(self.text or self.image_urls or self.video_urls)


class PlatformConversation[OutputT: PlatformOutput](BaseModel):
    """A post, whatever leads to it, and whatever hangs off it.

    Generic in the Output it holds so a platform declares its own model once and the four
    accessors below come back typed. The surface is identical on every platform on purpose: a
    caller written against one reads the others without learning a second set of rules, which is
    what lets one function build AI input from any of them.

    A field a platform cannot fill is still carried rather than dropped, so the accessors mean
    the same thing everywhere — `chain` is a list even where a platform serves no ancestors, and
    `selected_comment_id` is empty on a platform whose replies have URLs of their own.

    Attributes:
        chain: The chain ending at the linked post, ordered `[root, ..., parent, target]`.
        reply_branches: One list per reply branch under the target, each ordered from the direct
            reply outward, so an item's index in its branch is its nesting depth.
        selected_comment_id: The comment the URL singled out, for a platform whose links can.
    """

    chain: list[OutputT] = Field(
        default_factory=list, description="The chain ending at the linked post, root first"
    )
    reply_branches: list[list[OutputT]] = Field(
        default_factory=list,
        description="One list per reply branch under the target, direct reply first",
    )
    selected_comment_id: str = Field(
        default="", description="The comment the URL singled out, empty when it named none"
    )

    @computed_field
    @cached_property
    def target(self) -> OutputT | None:
        """The linked post itself, or None when it could not be read.

        The chain's LAST entry, which is the whole reason the field is a list.

        Cached, which makes "a conversation is built once and never mutated" load-bearing rather
        than merely true of the code today: pydantic invalidates a `cached_property` on neither an
        in-place `chain.append(...)` nor a `chain = [...]` assignment, so a mutated conversation
        keeps answering with its old target. Every platform builds both lists as locals and hands
        them to the constructor finished; media downloads run before the object exists, not after.
        """
        return self.chain[-1] if self.chain else None

    @computed_field
    @cached_property
    def selected_comment(self) -> OutputT | None:
        """The comment the URL singled out, None unless the platform can name one.

        None here rather than a lookup, because `selected_comment_id` is matched against an id
        the comment carries, and that id is not one of the nine — a platform whose links can
        address a single comment overrides this and reads its own field.
        """
        return None

    @property
    def comments(self) -> list[OutputT]:
        """Every reply, flattened out of the branches in page order.

        A plain property rather than a computed field: it re-slices data `reply_branches` already
        carries, so serializing it would put every reply in a dump twice. The two computed fields
        above resolve a POINTER instead, which a dump cannot derive on its own and which is what a
        hand test wants to see.
        """
        return [reply for branch in self.reply_branches for reply in branch]

    @property
    def posts(self) -> list[OutputT]:
        """Everything the page yielded: the chain oldest first, then the replies in page order."""
        return [*self.chain, *self.comments]
