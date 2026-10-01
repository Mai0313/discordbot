"""What every auto-expansion cog owes, checked against all of them at once.

A pasted link is the same feature on every platform, and the whole point of it is that a reader
learns it once: the same reply slot under the link, the same reactions meaning the same things,
the same silence in the channel when it does not work out. Cogs drifting apart is what that
promise fails as, and it fails quietly — every cog passes its own tests while the set of them
stops agreeing.

Most of the shell now lives in `utils/expansion_cog.py`, so most of this file asserts that a cog
did NOT take a piece of it back: not its own status mark, not its own failure classification, not
its own listener. The written-down list of cogs is `test_every_expansion_cog_is_accounted_for`,
which fails until a new source is named — that is what stops a source arriving with nobody
having read this file.

The shell's behaviour is run here too, once per cog, through the cog's own read with only its
downloader staged: when the listener stays quiet, what order the slot and the marks go on in, and
what a failure leaves behind. A cog's card is tested outside this file, and nothing here is
tested again there.
Staging is the other thing written down per cog, in `_STAGES`, so a new cog also fails those tests
until it says how its downloader is stood in for.

What deliberately is NOT here: how a post is rendered. A Threads chain, a Facebook comment
preload and a Douyin clip are different things and their cards should differ.
"""

from types import SimpleNamespace
from typing import Any, Literal, cast
import inspect
from pathlib import Path
import importlib
import contextlib
from collections.abc import Callable, Iterator

import pytest
from nextcord import Message, Forbidden

from discordbot.utils import expansion_cog as expansion_module
from discordbot.typings.emojis import LINK_SOURCE_EMOJIS
from discordbot.utils.link_errors import LinkRetryableError, LinkUnavailableError
from discordbot.utils.expansion_cog import (
    ExpansionCog,
    ConversationExpansionCog,
    expansion_failure_emoji,
    report_expansion_read_failure,
    report_expansion_delivery_failure,
)
from discordbot.services.platforms.base import PlatformConversation
from discordbot.services.platforms.threads import ThreadsOutput, ThreadsConversation
from discordbot.services.platforms.twitter import TwitterConversation
from discordbot.services.platforms.facebook import FacebookConversation
from discordbot.utils.expansion_placeholder import (
    EXPANSION_DONE_EMOJI,
    EXPANSION_FAILED_EMOJI,
    EXPANSION_WORKING_EMOJI,
    EXPANSION_UNREADABLE_EMOJI,
    EXPANSION_RETRY_LATER_EMOJI,
    ExpansionPlaceholder,
)
from discordbot.services.platforms.instagram import InstagramConversation

from tests.helpers.casting import as_message, make_forbidden, make_server_error
from tests.helpers.source_tree import PACKAGE
from tests.helpers.link_sources import (
    BOT_USER_ID,
    TWITTER_URL,
    FACEBOOK_URL,
    INSTAGRAM_URL,
    StubDouyinDownloader,
    stub_bot,
    twitter_post,
    facebook_post,
    instagram_post,
    hosting_off_planner,
    stub_conversation_cog,
)
from tests.helpers.discord_mocks import (
    FakeUser,
    FakeGuild,
    FakeDiscordMessage,
    placeholder_withdrawn,
)

_COGS_DIR = PACKAGE / "cogs"

# Every status mark an expansion may answer with lives in one module, so a literal left in a cog
# is the drift this file exists to catch: it is what lets one platform quietly answer ⚠️ where the
# others answer ⏱️. The platform markers in `typings/emojis.py` are not status marks and stay
# where they are.
_STATUS_LITERALS = (
    EXPANSION_WORKING_EMOJI,
    EXPANSION_DONE_EMOJI,
    EXPANSION_RETRY_LATER_EMOJI,
    EXPANSION_UNREADABLE_EMOJI,
    EXPANSION_FAILED_EMOJI,
)

# Calls that decide an outcome for every platform at once. A cog making one of them is deciding
# for itself again, which is how the marks and the log levels stopped agreeing before the shell
# existed.
_SHARED_DECISIONS = (
    "expansion_failure_emoji(",
    "report_expansion_read_failure(",
    "report_expansion_delivery_failure(",
    "send_expansion_placeholder(",
)


def _expansion_cog_classes() -> list[type[ExpansionCog[Any]]]:
    """Every cog class built on the shared expansion shell.

    Discovered by the base class rather than by a string in the source, so a cog that stops
    importing one helper cannot drop out of this file's coverage without anyone noticing.
    """
    found: list[type[ExpansionCog[Any]]] = []
    for entry in sorted(_COGS_DIR.iterdir()):
        source = entry / "cog.py"
        if entry.name.startswith("_") or not source.is_file():
            continue
        module = importlib.import_module(f"discordbot.cogs.{entry.name}.cog")
        for value in vars(module).values():
            if (
                inspect.isclass(value)
                and issubclass(value, ExpansionCog)
                and value is not ExpansionCog
                and value.__module__ == module.__name__
            ):
                found.append(value)
    return found


_COGS = _expansion_cog_classes()


def _cog_id(cog: type) -> str:
    """Names a parametrized case after the cog package it came from."""
    return cog.__module__.split(".")[-2]


def _cog_source(cog: type) -> str:
    """Reads a cog module's own source, for the checks that are about what it does not do."""
    return Path(inspect.getsourcefile(cog) or "").read_text(encoding="utf-8")


# What a staged read answers with: a post the cog can show, one it reads but cannot show, or an
# error its downloader raises.
type _Outcome = Literal["readable", "unreadable"] | Exception


class _Staged:
    """One cog whose downloader answers a staged outcome, and a guild message carrying its link."""

    def __init__(self, *, cog: ExpansionCog[Any], message: FakeDiscordMessage) -> None:
        """Holds the pair; nothing is served until `serve` installs a downloader factory."""
        self.cog = cog
        self.message = message
        # One entry per read the cog started: how many replies were already posted at that moment.
        self.reads: list[int] = []

    def serve(self, *, factory: Callable[..., object]) -> None:
        """Installs `factory` as the cog's per-read downloader seam, recording every read."""

        def recording(**kwargs: object) -> object:
            """Notes the read starting, then builds the staged downloader."""
            self.reads.append(len(self.message.replies))
            return factory(**kwargs)

        self.cog.__dict__["downloader_factory"] = recording


def _stage_conversation(
    *,
    cog: type[ExpansionCog[Any]],
    outcome: _Outcome,
    url: str,
    readable: PlatformConversation[Any],
    unreadable: PlatformConversation[Any],
) -> _Staged:
    """Stages a Facebook, Instagram or Twitter cog, whose unreadable post is an empty one."""
    if isinstance(outcome, Exception):
        instance, stub = stub_conversation_cog(cog_type=cog, outcome=outcome)
    else:
        instance, stub = stub_conversation_cog(
            cog_type=cog, outcome=readable if outcome == "readable" else unreadable
        )
    staged = _Staged(cog=instance, message=FakeDiscordMessage(content=url, guild=FakeGuild()))
    staged.serve(factory=lambda: stub)
    return staged


def _stage_threads(*, cog: type[ExpansionCog[Any]], outcome: _Outcome) -> _Staged:
    """Stages the Threads cog, whose unreadable post is a walk that found no chain."""
    url = "https://www.threads.com/@alice/post/ABC123"
    instance = cog(bot=stub_bot())
    instance.__dict__["media_delivery"] = hosting_off_planner()
    staged = _Staged(cog=instance, message=FakeDiscordMessage(content=url, guild=FakeGuild()))
    readable = ThreadsConversation(chain=[ThreadsOutput(text="post body", url=url)])

    @contextlib.contextmanager
    def walk(*, url: str) -> Iterator[ThreadsConversation]:
        """Enters the way `ThreadsDownloader.parse` does, so the failure lands in the walk."""
        del url
        if isinstance(outcome, Exception):
            raise outcome
        yield readable if outcome == "readable" else ThreadsConversation()

    staged.serve(factory=lambda output_folder: SimpleNamespace(parse=walk))
    return staged


def _stage_douyin(*, cog: type[ExpansionCog[Any]], outcome: _Outcome) -> _Staged:
    """Stages the Douyin cog.

    Douyin has no empty post: what it reads and then refuses is media nothing can carry, staged
    here as a clip past a four-byte upload ceiling with hosting off. A failure is raised by the
    download, the read's last step, so it crosses everything `read` does before reaching the shell.
    """
    instance = cog(bot=stub_bot())
    instance.__dict__["media_delivery"] = hosting_off_planner()
    guild = FakeGuild(filesize_limit=4) if outcome == "unreadable" else FakeGuild()
    staged = _Staged(
        cog=instance,
        message=FakeDiscordMessage(content="https://v.douyin.com/abc123", guild=guild),
    )
    error = outcome if isinstance(outcome, Exception) else None
    staged.serve(
        factory=lambda output_folder: StubDouyinDownloader(
            output_folder=output_folder, download_error=error
        )
    )
    return staged


_STAGES: dict[str, Callable[..., _Staged]] = {
    "parse_douyin": _stage_douyin,
    "parse_threads": _stage_threads,
    "parse_facebook": lambda *, cog, outcome: _stage_conversation(
        cog=cog,
        outcome=outcome,
        url=FACEBOOK_URL,
        readable=facebook_post(),
        unreadable=FacebookConversation(),
    ),
    "parse_instagram": lambda *, cog, outcome: _stage_conversation(
        cog=cog,
        outcome=outcome,
        url=INSTAGRAM_URL,
        readable=instagram_post(),
        unreadable=InstagramConversation(),
    ),
    "parse_twitter": lambda *, cog, outcome: _stage_conversation(
        cog=cog,
        outcome=outcome,
        url=TWITTER_URL,
        readable=twitter_post(),
        unreadable=TwitterConversation(),
    ),
}


def _stage(*, cog: type[ExpansionCog[Any]], outcome: _Outcome) -> _Staged:
    """Stages `cog` through its entry in `_STAGES`, which a new cog needs one of."""
    return _STAGES[_cog_id(cog=cog)](cog=cog, outcome=outcome)


def test_every_expansion_cog_is_accounted_for() -> None:
    """The written-down list of cogs, so a new source cannot arrive unread.

    Everything else here is discovered but its staging. This is the tripwire: a new expansion cog
    fails here, and the fix is to read this file, add its name, and give it an entry in `_STAGES`.
    """
    assert {_cog_id(cog=cog) for cog in _COGS} == {
        "parse_douyin",
        "parse_facebook",
        "parse_instagram",
        "parse_threads",
        "parse_twitter",
    }


def test_the_discovery_finds_something() -> None:
    """An empty parameter set skips every test below it and reports success.

    That is how this file went quietly blank once already: the probe was a string in the source,
    the string moved into the shared shell, and eight parametrized tests turned into skips.
    """
    assert _COGS


@pytest.mark.parametrize(argnames="cog", argvalues=_COGS, ids=_cog_id)
def test_an_expansion_cog_names_its_platform_the_shared_way(cog: type[ExpansionCog[Any]]) -> None:
    """`SOURCE` keys the pending-expansion rows, the marker lookup and the resume sweep.

    A key nothing else uses would look fine until a restart: the sweep reads its own rows by that
    string, so a private spelling resumes nothing and reports nothing either. It is also what the
    shell subscripts for the platform marker, so a stray spelling raises mid-expansion.
    """
    assert cog.SOURCE in LINK_SOURCE_EMOJIS


def test_no_two_expansion_cogs_share_a_source_key() -> None:
    """Two cogs on one key would each resume the other's interrupted expansions."""
    keys = [cog.SOURCE for cog in _COGS]

    assert len(set(keys)) == len(keys)


@pytest.mark.parametrize(argnames="cog", argvalues=_COGS, ids=_cog_id)
def test_an_expansion_cog_declares_what_the_shell_asks_it_for(
    cog: type[ExpansionCog[Any]],
) -> None:
    """A cog that leaves a hook unfilled does nothing at all, and says nothing about it.

    A conversation cog inherits its read and its card, so what it owes is the reader, the bound
    on it and the parts of the card only its platform knows. Its reader is checked here rather
    than by the tests below, which install a stand-in over it.
    """
    assert cog.PLATFORM
    assert cog.PLACEHOLDER_TEXT
    assert cog.URL_PATTERN.pattern
    assert cog.read is not ExpansionCog.read
    assert cog.build_delivery is not ExpansionCog.build_delivery
    if issubclass(cog, ConversationExpansionCog):
        assert cog.READ_TIMEOUT_SECONDS > 0
        assert cog.EMBED_COLOR
        assert callable(getattr(cog, "downloader_factory", None))
        assert cog._footer_text is not ConversationExpansionCog._footer_text


@pytest.mark.parametrize(argnames="cog", argvalues=_COGS, ids=_cog_id)
def test_an_expansion_cog_keeps_the_shared_listener(cog: type[ExpansionCog[Any]]) -> None:
    """The listener, the restart sweep and the expansion body are the shell's, not a cog's.

    Overriding any of them is how the reply slot, the reaction order and the resume contract
    stopped agreeing when each cog held its own copy. `_expand` in particular is what
    `resume_expansion_placeholders` calls with the listener's own four arguments, so a cog
    redefining it can break a restart and nothing else.
    """
    assert cog.on_message is ExpansionCog.on_message
    assert cog.on_ready is ExpansionCog.on_ready
    assert cog._expand is ExpansionCog._expand
    assert cog._mark_failed is ExpansionCog._mark_failed


@pytest.mark.parametrize(argnames="cog", argvalues=_COGS, ids=_cog_id)
def test_an_expansion_cog_spells_no_status_mark_of_its_own(cog: type[ExpansionCog[Any]]) -> None:
    """One symbol, one meaning, whichever platform was linked.

    The reaction is the entire report — an expansion that produced nothing says nothing in the
    channel — so a cog inventing its own mark, or reusing a shared one for a different outcome,
    is the whole feature's vocabulary coming apart.
    """
    body = "\n".join(
        line for line in _cog_source(cog=cog).split("\n") if not line.lstrip().startswith("#")
    )

    for literal in _STATUS_LITERALS:
        assert f'"{literal}"' not in body, f"{cog.__module__} spells {literal} itself"


@pytest.mark.parametrize(argnames="cog", argvalues=_COGS, ids=_cog_id)
def test_an_expansion_cog_decides_no_shared_outcome_of_its_own(
    cog: type[ExpansionCog[Any]],
) -> None:
    """A cog reading the error's type itself is how the platforms stop agreeing.

    Every one of these calls answers the same question for every platform, so the shell makes
    them and a cog that makes one again has taken the decision back. `parse_douyin` used to own
    the logging split alone and was right; the others logged a deleted post at `warn` with a
    traceback, which is what makes a real regression unfindable.
    """
    source = _cog_source(cog=cog)

    for call in _SHARED_DECISIONS:
        assert call not in source, f"{cog.__module__} makes the shared decision {call}"
    assert "isinstance(error, TimeoutError)" not in source


@pytest.mark.parametrize(argnames="cog", argvalues=_COGS, ids=_cog_id)
async def test_a_failure_with_nothing_on_the_message_still_names_the_platform(
    cog: type[ExpansionCog[Any]],
) -> None:
    """A cross on its own cannot say which link died, and a message can carry two.

    `current_emoji` is None exactly when claiming the reply slot failed, which is both the refused
    channel and the Discord 5xx that raises straight past it. Behavioural rather than a source
    scan, because what matters is that the marker lands whichever call site got there.
    """
    instance = cog(bot=stub_bot())
    message = FakeDiscordMessage()

    await instance._mark_failed(message=as_message(fake=message), current_emoji=None)

    assert message.reactions == [LINK_SOURCE_EMOJIS[cog.SOURCE], EXPANSION_FAILED_EMOJI]


@pytest.mark.parametrize(argnames="cog", argvalues=_COGS, ids=_cog_id)
async def test_a_link_the_reply_pipeline_will_answer_is_left_alone(
    cog: type[ExpansionCog[Any]],
) -> None:
    """A mention or a DM hands the link to `gen_reply`, and expanding it as well reads it twice."""
    mentioned = _stage(cog=cog, outcome="readable")
    mentioned.message.content = f"<@{BOT_USER_ID}> what is this {mentioned.message.content}"
    direct = _stage(cog=cog, outcome="readable")
    direct.message.guild = None

    for staged in (mentioned, direct):
        await staged.cog.on_message(message=as_message(fake=staged.message))

        assert staged.reads == []
        assert staged.message.reactions == []
        assert staged.message.replies == []


@pytest.mark.parametrize(argnames="cog", argvalues=_COGS, ids=_cog_id)
async def test_a_bot_author_is_ignored(cog: type[ExpansionCog[Any]]) -> None:
    """Otherwise the bot's own posts, and other bots' link cards, would be expanded again."""
    staged = _stage(cog=cog, outcome="readable")
    staged.message.author = FakeUser(bot=True)

    await staged.cog.on_message(message=as_message(fake=staged.message))

    assert staged.reads == []
    assert staged.message.reactions == []


@pytest.mark.parametrize(argnames="cog", argvalues=_COGS, ids=_cog_id)
async def test_the_reply_slot_is_claimed_before_the_reactions_and_the_read(
    cog: type[ExpansionCog[Any]],
) -> None:
    """The card is what the reader is waiting for, so nothing queues in front of its slot.

    Both reactions share one per-channel rate-limit bucket that a message send does not, so
    reacting first only delays the placeholder. Claiming it before the read is what keeps the card
    directly under the link rather than wherever the channel has got to once the post is read.
    """
    staged = _stage(cog=cog, outcome="readable")
    reactions_when_claimed: list[str] = []
    claim = staged.message.reply

    async def recording_reply(**kwargs: object) -> object:
        """Snapshots the reaction row at the moment the slot is claimed."""
        reactions_when_claimed.extend(staged.message.reactions)
        return await claim(**cast("Any", kwargs))

    staged.message.reply = recording_reply  # ty: ignore[invalid-assignment]

    await staged.cog.on_message(message=as_message(fake=staged.message))

    assert reactions_when_claimed == []
    assert staged.reads == [1]
    assert staged.message.reactions[-1] == EXPANSION_DONE_EMOJI


@pytest.mark.parametrize(argnames="cog", argvalues=_COGS, ids=_cog_id)
async def test_the_platform_marker_rides_beside_the_status_chain(
    cog: type[ExpansionCog[Any]],
) -> None:
    """The marker says WHICH link was read, which the status mark alone cannot.

    The chain only ever removes its own reaction, so the marker outlives every step of it.
    """
    staged = _stage(cog=cog, outcome="readable")
    marker = LINK_SOURCE_EMOJIS[cog.SOURCE]

    await staged.cog.on_message(message=as_message(fake=staged.message))

    assert staged.message.reactions == [marker, EXPANSION_WORKING_EMOJI, EXPANSION_DONE_EMOJI]
    assert all(emoji != marker for emoji, _ in staged.message.removed)


@pytest.mark.parametrize(argnames="cog", argvalues=_COGS, ids=_cog_id)
async def test_a_refused_slot_reads_nothing_and_still_names_the_platform(
    cog: type[ExpansionCog[Any]],
) -> None:
    """A channel that will not take the placeholder will not take the card either.

    Finding that out before the read costs no request, which matters most on a platform that bans
    on request volume. The marker still goes on, because a channel granting Add Reactions but not
    Send Messages is exactly where someone has to work out which of two links died.
    """
    staged = _stage(cog=cog, outcome="readable")

    async def refuse(**kwargs: object) -> object:
        """Answers the way a channel the bot cannot post in does."""
        del kwargs
        raise make_forbidden()

    staged.message.reply = refuse  # ty: ignore[invalid-assignment]

    await staged.cog.on_message(message=as_message(fake=staged.message))

    assert staged.reads == []
    assert staged.message.reactions == [LINK_SOURCE_EMOJIS[cog.SOURCE], EXPANSION_FAILED_EMOJI]


@pytest.mark.parametrize(argnames="cog", argvalues=_COGS, ids=_cog_id)
@pytest.mark.parametrize(
    argnames=("outcome", "expected"),
    argvalues=[
        ("unreadable", EXPANSION_UNREADABLE_EMOJI),
        (LinkRetryableError("429"), EXPANSION_RETRY_LATER_EMOJI),
        (RuntimeError("the parser blew up"), EXPANSION_FAILED_EMOJI),
    ],
    ids=["unshowable", "refused", "broke"],
)
async def test_an_expansion_that_delivers_nothing_leaves_only_its_mark(
    cog: type[ExpansionCog[Any]], outcome: _Outcome, expected: str
) -> None:
    """The reaction is the whole report: the placeholder is taken back and nothing is said.

    Which mark is the point. A deleted or private post is the post's own state and a platform
    under load is a link that works in a minute, so neither may earn the cross, which says the bot
    broke: telling someone a working link is dead is the worst outcome this feature has.
    """
    staged = _stage(cog=cog, outcome=outcome)

    await staged.cog.on_message(message=as_message(fake=staged.message))

    assert staged.message.reactions == [
        LINK_SOURCE_EMOJIS[cog.SOURCE],
        EXPANSION_WORKING_EMOJI,
        expected,
    ]
    assert placeholder_withdrawn(message=staged.message)
    assert not staged.message.suppressed  # nothing was delivered, so the link keeps its preview


@pytest.mark.parametrize(argnames="cog", argvalues=_COGS, ids=_cog_id)
async def test_a_failure_outside_the_read_still_marks_the_message(
    cog: type[ExpansionCog[Any]],
) -> None:
    """The listener's outer handler is the last line of defence, so nothing escapes unmarked."""
    staged = _stage(cog=cog, outcome="readable")

    async def explode(
        *, message: Message, url: str, current_emoji: str, placeholder: ExpansionPlaceholder
    ) -> None:
        """Fails the way a Discord API error outside the read and the send does."""
        del message, url, current_emoji, placeholder
        raise RuntimeError("discord exploded")

    staged.cog.__dict__["_expand"] = explode

    await staged.cog.on_message(message=as_message(fake=staged.message))

    assert staged.message.reactions == [
        LINK_SOURCE_EMOJIS[cog.SOURCE],
        EXPANSION_WORKING_EMOJI,
        EXPANSION_FAILED_EMOJI,
    ]
    assert placeholder_withdrawn(message=staged.message)


@pytest.mark.parametrize(
    argnames=("error", "expected"),
    argvalues=[
        (LinkRetryableError("429"), EXPANSION_RETRY_LATER_EMOJI),
        (TimeoutError(), EXPANSION_RETRY_LATER_EMOJI),
        (LinkUnavailableError("410"), EXPANSION_UNREADABLE_EMOJI),
        (RuntimeError("the parser blew up"), EXPANSION_FAILED_EMOJI),
    ],
    ids=["refused", "stalled", "gone", "broke"],
)
def test_one_failure_earns_the_same_mark_on_every_platform(
    error: Exception, expected: str
) -> None:
    """The vocabulary is only worth anything if the same failure reads the same everywhere.

    This pins the mapping itself; `test_an_expansion_cog_decides_no_shared_outcome_of_its_own` is
    what says every cog actually goes through it. Parametrizing this one over the cogs too would
    have looked like every platform was checked while testing one function repeatedly.
    """
    assert expansion_failure_emoji(error=error) == expected


@pytest.mark.parametrize(
    argnames=("error", "expected"),
    argvalues=[
        (LinkUnavailableError("410"), "info"),
        (LinkRetryableError("429"), "warn"),
        (TimeoutError(), "warn"),
        (RuntimeError("the parser blew up"), "error"),
    ],
    ids=["gone", "refused", "stalled", "broke"],
)
def test_a_read_failure_is_logged_at_the_severity_its_outcome_earns(
    error: Exception, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ladder is keyed on how tolerable the failure is, not on how deep it happened.

    A deleted post logged at `warn` with a traceback is what makes a real regression unfindable,
    and it is `.github/CONTRIBUTING.md#logging`'s own example of `info`.
    """
    levels: list[str] = []
    for level in ("info", "warn", "error"):
        monkeypatch.setattr(
            target=expansion_module.logfire,
            name=level,
            value=lambda _message, level=level, **fields: levels.append(level),
        )

    report_expansion_read_failure(
        error=error, platform="Threads", url="https://example.test/p/1", message_id=7
    )

    assert levels == [expected]


def test_a_routine_remote_outcome_carries_its_reason_and_no_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A traceback for a deleted post is noise; the platform's own words are not.

    Douyin's filter reason exists in no other line, which is why the `info` branch keeps it while
    dropping the exception the ladder says that level usually does not carry.
    """
    recorded: dict[str, object] = {}
    monkeypatch.setattr(
        target=expansion_module.logfire,
        name="info",
        value=lambda _message, **fields: recorded.update(fields),
    )

    report_expansion_read_failure(
        error=LinkUnavailableError("Douyin will not serve 123: filtered"),
        platform="Douyin",
        url="https://example.test/p/1",
        message_id=7,
    )

    assert "filtered" in str(recorded["reason"])
    assert "_exc_info" not in recorded
    assert recorded["message_id"] == 7


@pytest.mark.parametrize(argnames="cog", argvalues=_COGS, ids=_cog_id)
async def test_a_guild_that_refuses_the_preview_suppress_still_gets_the_card(
    cog: type[ExpansionCog[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hiding Discord's own preview takes Manage Messages, which many guilds never grant.

    That refusal repeats on every link pasted there with an identical stack, so it is reported
    with the ids alone (`.github/CONTRIBUTING.md#logging`), and the card lands regardless.
    """
    staged = _stage(cog=cog, outcome="readable")
    staged.message.edit_failure = make_forbidden(message="Missing Permissions")
    warns: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        target=expansion_module.logfire,
        name="warn",
        value=lambda message, **fields: warns.append((message, fields)),
    )

    await staged.cog.on_message(message=as_message(fake=staged.message))

    assert staged.message.reactions[-1] == EXPANSION_DONE_EMOJI
    assert warns == [
        (
            "Could not suppress the source message embed",
            {"message_id": 1, "guild_id": 100, "error_type": "Forbidden"},
        )
    ]


@pytest.mark.parametrize(
    argnames=("error", "level", "traceback"),
    argvalues=[
        (
            Forbidden(
                response=make_forbidden().response,
                message={
                    "code": 400001,
                    "message": "Access to file uploads has been limited for this guild",
                },
            ),
            "warn",
            False,
        ),
        (make_server_error(), "error", True),
    ],
    ids=["refused", "broke"],
)
def test_a_refused_delivery_names_its_code_instead_of_a_traceback(
    error: Exception, level: str, traceback: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal's stack is the same every time, while its code says which refusal it was.

    Not every `Forbidden` here is a missing permission (#719), so the code is what survives of
    the traceback; a 5xx keeps its traceback.
    """
    reports: list[tuple[str, dict[str, object]]] = []
    for name in ("info", "warn", "error"):
        monkeypatch.setattr(
            target=expansion_module.logfire,
            name=name,
            value=lambda _message, name=name, **fields: reports.append((name, fields)),
        )

    report_expansion_delivery_failure(
        error=error, platform="Threads", url="https://example.test/p/1", message_id=7, channel_id=8
    )

    assert [(name, "_exc_info" in fields) for name, fields in reports] == [(level, traceback)]
    if not traceback:
        assert reports[0][1]["code"] == 400001
