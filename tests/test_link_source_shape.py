"""Pins the one shape Threads, Facebook and Instagram parse into.

The three sources converged on purpose: a caller written against one reads the others without
learning a second set of rules, which is what lets the link-source builders and the expansion
cogs share a vocabulary instead of three. Nothing else in the suite would notice them drifting
apart again — every other test exercises one platform, and a fourth source added by copying
whichever file was opened first passes all of them.

So the classes here are DISCOVERED rather than listed: every `*Conversation` under
`discordbot.services.platforms` is swept up, and every guard below covers a new source without
being told about it. `test_the_sweep_finds_every_source` is the one deliberate exception — it
names the three, so a fourth fails it until somebody writes the name down, which is what stops a
source arriving without anyone having read this file.

A sweep driven by a name also collects the base the platforms are built FROM.
`PlatformConversation` ends in `Conversation`, is a `BaseModel` and is defined in this package, so
it satisfies every part of the filter — and then fails the guards for a reason that says nothing
about any platform: its `chain` is annotated with the bare TypeVar, so `_output_model` hands back
`OutputT` rather than a model.

It is excluded on the one property that actually separates the two: the base still carries an
UNBOUND type parameter, and a platform that named its Output has none left. Not
`__pydantic_generic_metadata__["args"]`, which records what a specialised ALIAS was handed and is
therefore empty on the base and on all three platforms alike — a filter reading it cannot tell them
apart at all, and collects either all four or none.

Which is exactly the hole the exclusion itself could hide, so `test_the_sweep_finds_every_source`
asserts what was excluded as well as what was kept: a platform accidentally written as a generic
would otherwise vanish from the sweep and pass every guard below by covering nothing.
"""

from typing import Any, Protocol, cast
from pkgutil import walk_packages
from importlib import import_module

import pytest
from pydantic import BaseModel

# Namespace import: the package object itself is the input, for its __path__.
import discordbot.services.platforms


class _Conversation(Protocol):
    """The surface this module exists to hold the three sources to.

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

# What a caller may read off any post or comment from any of the three. Each source carries more
# on top — Threads its `video_paths`, `quoted` and repost/quote counters, Facebook its
# `group_name`, Instagram its `author_full_name` — and those are the platform's own reality
# rather than drift. `share_count` is deliberately NOT here: Threads and Facebook publish one and
# Instagram does not, and a field nothing populates is worse than an absent one.
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


def _all_conversation_classes() -> dict[str, type[BaseModel]]:
    """Every `*Conversation` model under `discordbot.services.platforms`, base included.

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
            if not name.endswith("Conversation") or not isinstance(value, type):
                continue
            if issubclass(value, BaseModel) and value.__module__ == imported.__name__:
                found[name] = value
    return found


def _conversation_classes() -> dict[str, type[BaseModel]]:
    """The platform conversations: everything above that bound its Output."""
    return {
        name: value
        for name, value in _all_conversation_classes().items()
        if not value.__pydantic_generic_metadata__["parameters"]
    }


def _generic_conversation_classes() -> set[str]:
    """What the filter excluded — the bases, and anything that forgot to name its Output."""
    return {
        name
        for name, value in _all_conversation_classes().items()
        if value.__pydantic_generic_metadata__["parameters"]
    }


def _build(
    *, cls: type[BaseModel], chain_length: int = 1
) -> tuple[_Conversation, list[BaseModel], BaseModel]:
    """A conversation of the discovered class, plus the chain and the reply that built it."""
    output = cast("Any", _output_model(cls=cls))
    chain = [output(text=f"chain post {index}", url="u") for index in range(chain_length)]
    reply = output(text="a comment", url="u")
    conversation = cast("Any", cls)(chain=chain, reply_branches=[[reply]])
    return cast("_Conversation", conversation), chain, reply


def _output_model(*, cls: type[BaseModel]) -> type[BaseModel]:
    """The `<Platform>Output` a conversation's `chain` holds."""
    return cast("Any", cls.model_fields["chain"].annotation).__args__[0]


def _properties(*, cls: type) -> set[str]:
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
    }
    assert _generic_conversation_classes() == {"PlatformConversation"}


@pytest.mark.parametrize("name", sorted(_conversation_classes()))
def test_every_conversation_carries_the_same_surface(name: str) -> None:
    """One shape across the three, so a caller never has to ask which platform it holds."""
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

    Only Threads ever serves ancestors, so a one-element chain would pin nothing here and leave
    "the only entry" passing for "the last one". Facebook and Instagram are handed two anyway:
    the accessor has to mean the same thing everywhere, whatever their pages actually serve.
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
