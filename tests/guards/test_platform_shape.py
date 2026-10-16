"""Pins the shape every platform under `discordbot.services.platforms` shares.

Two surfaces: the conversation MODELS Threads, Facebook, Instagram and Twitter parse into, and the
entry point every DOWNLOADER answers on. The sources converged on purpose: a caller written
against one reads the others without learning a second set of rules. Nothing else in the suite
would notice them drifting apart again — every other test exercises one platform, and a new
source added by copying whichever file was opened first passes all of them. Downloaders answering
the same question can drift on how it is asked, too — a positional `url` on one and keyword-only
on another — with no other test seeing it.

So the classes here are DISCOVERED rather than listed: every `*Conversation` and every
`*Downloader` under the package is swept up, and every guard below covers a new platform without
being told about it. The two `test_the_sweep_finds_*` tests are the deliberate exceptions — they
name them all, so a new one fails until somebody writes the name down, which is what stops a
platform arriving without anyone having read this file.

A sweep driven by a name also collects the bases the platforms are built FROM.
`PlatformConversation` ends in `Conversation`, is a `BaseModel` and is defined in this package, so
it satisfies every part of the filter — and then fails the guards for a reason that says nothing
about any platform: its `chain` is annotated with the bare TypeVar, so `_output_model` hands back
`OutputT` rather than a model.

It is excluded on the one property that actually separates the two: the base still carries an
UNBOUND type parameter, and a platform that named its Output has none left. Not
`__pydantic_generic_metadata__["args"]`, which records what a specialised ALIAS was handed and is
therefore empty on the base and on every platform alike — a filter reading it cannot tell them
apart at all, and collects either all of them or none.

Which is exactly the hole the exclusion itself could hide, so `test_the_sweep_finds_every_source`
asserts what was excluded as well as what was kept: a platform accidentally written as a generic
would otherwise vanish from the sweep and pass every guard below by covering nothing.

`PlatformDownloader` is excluded BY IDENTITY rather than by the module it lives in. Sweeping it
would check the base against itself — its `parse_metadata` is the `NotImplementedError` stub every
assertion below exists to say a platform replaced, and its declared return is the bare `BaseModel`
a platform must narrow. Skipping `base.py` wholesale would do the same job today and stop doing it
the moment that file gains a second class, so the exclusion names the one thing it means, and
`test_the_sweep_finds_every_downloader` asserts what was excluded as well as what was kept.
"""

from typing import Any, Protocol, cast
from inspect import Parameter, signature
from pkgutil import walk_packages
from importlib import import_module

import pytest
from pydantic import BaseModel

# Namespace import: the package object itself is the input, for its __path__.
import discordbot.services.platforms
from discordbot.services.platforms.base import PlatformDownloader


class _Conversation(Protocol):
    """The surface this module exists to hold every source to.

    Spelled out as a Protocol because the classes under test are DISCOVERED, so the checker sees
    only `type[BaseModel]` and every read below would be an `unresolved-attribute`. It buys the
    reads their types and nothing more: every value reaches it through a `cast`, which is an
    unchecked assertion, so a member vanishing from one source is caught by the assertions below
    and never by the checker.
    """

    @property
    def target(self) -> BaseModel | None: ...
    @property
    def selected_comment(self) -> BaseModel | None: ...
    @property
    def comments(self) -> list[BaseModel]: ...
    @property
    def posts(self) -> list[BaseModel]: ...


# The fields, computed fields and properties every conversation carries. `comments` and `posts`
# are plain properties on purpose: they re-slice what `reply_branches` already holds, so a
# computed field would put every comment in a dump twice, while `target` and `selected_comment`
# resolve a pointer a dump cannot derive on its own.
_CONVERSATION_FIELDS = frozenset({"chain", "reply_branches", "selected_comment_id"})
_CONVERSATION_COMPUTED = frozenset({"target", "selected_comment"})
_CONVERSATION_PROPERTIES = frozenset({"comments", "posts"})

# What a caller may read off any post or comment from any source. Each carries more on top —
# Threads its `video_paths`, `quoted` and repost/quote counters, Facebook its `group_name`,
# Instagram its `author_full_name`, Twitter its `video_poster_urls` and `is_truncated` — and those
# are the platform's own reality rather than drift. `share_count` is deliberately NOT here: Threads
# and Facebook publish one and Instagram does not, and a field nothing populates is worse than an
# absent one. `retweet_count` is absent for the same reason: Twitter reports one for an embedded
# post and never for the post asked for.
_OUTPUT_FIELDS = frozenset({
    "text",
    "url",
    "author_name",
    "author_icon_url",
    "image_urls",
    "video_urls",
    "like_count",
    "comment_count",
    "taken_at",
})
_OUTPUT_COMPUTED = frozenset({"is_readable"})


def _platform_classes(suffix: str) -> dict[str, type[BaseModel]]:
    """Every model under `discordbot.services.platforms` named `*<suffix>`, bases included.

    `walk_packages` rather than `iter_modules`, so a platform written as a subpackage is still
    found. With the top level alone its classes reach neither this sweep nor the tripwire, and a
    name nothing collected cannot make a set-equality assertion fail — it passes by matching an
    expectation that also does not list it.
    """
    found: dict[str, type[BaseModel]] = {}
    for module in walk_packages(
        path=discordbot.services.platforms.__path__, prefix="discordbot.services.platforms."
    ):
        imported = import_module(name=module.name)
        for name, value in vars(imported).items():
            if not name.endswith(suffix) or not isinstance(value, type):
                continue
            if issubclass(value, BaseModel) and value.__module__ == imported.__name__:
                found[name] = value
    return found


def _conversation_classes() -> dict[str, type[BaseModel]]:
    """The platform conversations: every `*Conversation` swept up that bound its Output."""
    return {
        name: value
        for name, value in _platform_classes(suffix="Conversation").items()
        if not value.__pydantic_generic_metadata__["parameters"]
    }


def _generic_conversation_classes() -> set[str]:
    """What the filter excluded — the bases, and anything that forgot to name its Output."""
    return {
        name
        for name, value in _platform_classes(suffix="Conversation").items()
        if value.__pydantic_generic_metadata__["parameters"]
    }


def _build(
    cls: type[BaseModel], chain_length: int = 1
) -> tuple[_Conversation, list[BaseModel], BaseModel]:
    """A conversation of the discovered class, plus the chain and the reply that built it."""
    output = cast("Any", _output_model(cls=cls))
    chain = [output(text=f"chain post {index}", url="u") for index in range(chain_length)]
    reply = output(text="a comment", url="u")
    conversation = cast("Any", cls)(chain=chain, reply_branches=[[reply]])
    return cast("_Conversation", conversation), chain, reply


def _output_model(cls: type[BaseModel]) -> type[BaseModel]:
    """The `<Platform>Output` a conversation's `chain` holds."""
    return cast("Any", cls.model_fields["chain"].annotation).__args__[0]


def _properties(cls: type) -> set[str]:
    """The public plain properties a caller can read off the class, inherited ones included.

    Walks the MRO rather than the class itself because `comments` and `posts` now come from
    `PlatformConversation`, which is the point of having it. Nothing is lost by looking wider: the
    assertion that matters is that these two are NOT computed fields, and the exact-equality check
    on `model_computed_fields` is what proves that.
    """
    return {
        name
        for klass in cls.__mro__
        for name, value in vars(klass).items()
        if isinstance(value, property) and not name.startswith("_")
    }


def test_the_sweep_finds_every_source() -> None:
    """A guard over a set it failed to collect would pass by finding nothing.

    Both halves are named, because the sweep can lose a source two ways: by not reaching its
    module, and by excluding it as a generic. The second is the exclusion this file relies on, so
    what it removed is asserted too — `PlatformConversation` and nothing else.
    """
    assert set(_conversation_classes()) == {
        "ThreadsConversation",
        "FacebookConversation",
        "InstagramConversation",
        "TwitterConversation",
    }
    assert _generic_conversation_classes() == {"PlatformConversation"}


@pytest.mark.parametrize("name", sorted(_conversation_classes()))
def test_every_conversation_carries_the_same_surface(name: str) -> None:
    """One shape across every source, so a caller never has to ask which platform it holds."""
    cls = _conversation_classes()[name]

    assert set(cls.model_fields) == _CONVERSATION_FIELDS
    assert set(cls.model_computed_fields) == _CONVERSATION_COMPUTED
    assert _properties(cls=cls) >= _CONVERSATION_PROPERTIES


@pytest.mark.parametrize("name", sorted(_conversation_classes()))
def test_every_output_carries_the_common_core(name: str) -> None:
    """The fields a renderer reads without knowing which source it is rendering."""
    cls = _conversation_classes()[name]
    output = _output_model(cls=cls)

    assert set(output.model_fields) >= _OUTPUT_FIELDS
    assert set(output.model_computed_fields) >= _OUTPUT_COMPUTED


@pytest.mark.parametrize("name", sorted(_conversation_classes()))
def test_a_dump_carries_each_comment_once(name: str) -> None:
    """`comments` re-slices `reply_branches`, so serializing it doubles every comment."""
    cls = _conversation_classes()[name]
    conversation, _chain, _reply = _build(cls=cls)

    dumped = cls.model_dump(cast("Any", conversation))

    assert "comments" not in dumped
    assert "posts" not in dumped
    assert str(dumped).count("a comment") == 1


@pytest.mark.parametrize("name", sorted(_conversation_classes()))
def test_target_is_the_last_chain_entry_and_posts_is_everything(name: str) -> None:
    """`target` is the chain's LAST entry, which is the whole reason the field is a list.

    A one-element chain would pin nothing here and leave "the only entry" passing for "the last
    one". Every source is handed two, including Facebook and Instagram, whose pages never serve an
    ancestor: the accessor has to mean the same thing everywhere, whatever a page actually serves.
    """
    cls = _conversation_classes()[name]
    conversation, chain, reply = _build(cls=cls, chain_length=2)

    assert conversation.target is chain[-1]
    assert conversation.comments == [reply]
    assert conversation.posts == [*chain, reply]


@pytest.mark.parametrize("name", sorted(_conversation_classes()))
def test_an_empty_conversation_reads_as_unreadable(name: str) -> None:
    """The degraded outcome every source shares: a login wall, a deletion, a private account."""
    empty = cast("_Conversation", _conversation_classes()[name]())

    assert empty.target is None
    assert empty.selected_comment is None
    assert empty.comments == []
    assert empty.posts == []


# Every platform reading a link today. A new one has to be written in here, which is the step that
# makes someone read the rules above.
_EXPECTED_DOWNLOADERS = frozenset({
    "ThreadsDownloader",
    "FacebookDownloader",
    "InstagramDownloader",
    "TwitterDownloader",
    "DouyinDownloader",
    "VideoDownloader",
})


def _downloader_classes() -> dict[str, type[PlatformDownloader]]:
    """The platform downloaders: every `*Downloader` swept up except the contract itself."""
    return {
        name: cast("type[PlatformDownloader]", value)
        for name, value in _platform_classes(suffix="Downloader").items()
        if value is not PlatformDownloader
    }


def test_the_sweep_finds_every_downloader() -> None:
    """A guard over a set it failed to collect would pass by finding nothing.

    Both halves are named, because the sweep can lose a platform two ways: by not reaching its
    module, and by being excluded alongside the contract. What the exclusion removed is therefore
    asserted too — `PlatformDownloader` and nothing else.
    """
    assert set(_downloader_classes()) == set(_EXPECTED_DOWNLOADERS)
    assert set(_platform_classes(suffix="Downloader")) - set(_EXPECTED_DOWNLOADERS) == {
        "PlatformDownloader"
    }


@pytest.mark.parametrize("name", sorted(_downloader_classes()))
def test_every_downloader_answers_the_shared_contract(name: str) -> None:
    """Inheriting the base is what makes `parse_metadata` a contract rather than a coincidence."""
    assert issubclass(_downloader_classes()[name], PlatformDownloader)


@pytest.mark.parametrize("name", sorted(_downloader_classes()))
def test_every_downloader_declares_its_own_parse_metadata(name: str) -> None:
    """A platform that inherited the stub would raise on its first real call instead of at import."""
    cls = _downloader_classes()[name]

    assert "parse_metadata" in vars(cls)


@pytest.mark.parametrize("name", sorted(_downloader_classes()))
def test_parse_metadata_takes_only_a_url(name: str) -> None:
    """One spelling at every call site, so a platform swap is not also a signature change."""
    # Read off the class, so the signature is the unbound function's and carries `self`.
    parameters = signature(_downloader_classes()[name].parse_metadata).parameters
    arguments = [parameter for parameter in parameters if parameter != "self"]

    assert arguments == ["url"], f"{name}.parse_metadata takes more than a url"
    assert parameters["url"].kind is Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["url"].annotation is str


@pytest.mark.parametrize("name", sorted(_downloader_classes()))
def test_parse_metadata_returns_a_model_of_its_own_module(name: str) -> None:
    """The return narrows to this platform's `<Platform>Conversation` or `<Platform>Metadata`.

    Declared in the same module, which is what stops a platform answering with another's model: the
    shape of a Douyin post is Douyin's to describe, and a shared return type would have to be the
    union of everything or the intersection of nothing.
    """
    cls = _downloader_classes()[name]
    returns = signature(cls.parse_metadata).return_annotation

    assert isinstance(returns, type), f"{name}.parse_metadata returns {returns!r}, not a class"
    assert issubclass(returns, BaseModel), f"{name}.parse_metadata does not return a model"
    assert returns is not BaseModel, f"{name}.parse_metadata did not narrow the base's return"
    assert returns.__module__ == cls.__module__, (
        f"{name}.parse_metadata returns {returns.__name__}, declared in another module"
    )


@pytest.mark.parametrize("name", sorted(_downloader_classes()))
def test_a_download_folder_is_required_when_it_exists(name: str) -> None:
    """A default would put scratch files somewhere nobody chose, and `data/` is one typo away.

    Only a platform that writes a file has the field at all: Facebook and Instagram hand image
    URLs to Discord and keep no state, so one instance serves every caller there.
    """
    field = _downloader_classes()[name].model_fields.get("output_folder")
    if field is None:
        return

    assert field.is_required(), f"{name}.output_folder has a default"


@pytest.mark.parametrize("name", sorted(_downloader_classes()))
def test_a_parse_half_takes_the_same_url(name: str) -> None:
    """`parse` is optional, but a platform that has one spells its url like `parse_metadata` does.

    Only Threads has it. Three platforms write files, but their second methods have nothing
    a base could hold them to: Threads returns a conversation with each file on the post that
    owns it, while Douyin and yt-dlp spell theirs `download` and return a model naming their
    files, taking their own options alongside the url. That is why the base declares no second
    method at all — see its module docstring.
    """
    parse = getattr(_downloader_classes()[name], "parse", None)
    if parse is None:
        return

    url = signature(parse).parameters["url"]

    assert url.kind is Parameter.POSITIONAL_OR_KEYWORD
    assert url.annotation is str
