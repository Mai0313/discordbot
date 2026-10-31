"""Tests for one reply turn: routing, dispatch and linked-post context."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal
import asyncio
from collections import Counter

import pytest
from pydantic import Field, BaseModel

from discordbot.typings.emojis import (
    DOUYIN_EMOJI,
    THREADS_EMOJI,
    TWITTER_EMOJI,
    BILIBILI_EMOJI,
    FACEBOOK_EMOJI,
    INSTAGRAM_EMOJI,
)
from discordbot.typings.models import RouteClassification
from discordbot.cogs.gen_reply.answer import AnswerTurn
from discordbot.cogs.gen_reply.context import ReplyContext, ReplyContextBuilder
from discordbot.cogs.gen_reply.routing import RouteClassifier
from discordbot.cogs.gen_reply.pipeline import UNROUTED_REPLY
from discordbot.typings.context_budgets import HISTORY_MESSAGE_LIMIT
from discordbot.cogs.gen_reply.media_reply import MediaReplyRoutes
from discordbot.cogs.gen_reply.speculation import (
    discard_link_tasks,
    run_until_deadline,
    await_deadline_bound_task,
)
from discordbot.cogs.gen_reply.link_sources import link_context_blocks
from discordbot.cogs.gen_reply.status_marks import DONE_EMOJI

from tests.helpers.casting import as_message
from tests.helpers.llm_input import (
    LINK_SOURCE_BLOCKS,
    block_index,
    request_input,
    has_timeout_notice,
    has_link_context_block,
    extract_link_context_block,
)
from tests.helpers.usage_log import usage_records
from tests.helpers.reply_fakes import (
    _LINK_SOURCES,
    FakeAuthor,
    FakeMessage,
    FakeReference,
    _cog,
    _recorded,
    _build_stub,
    _config_stub,
    _run_pipeline,
    _classify_stub,
    _install_streamer,
)
from tests.helpers.link_sources import SAMPLE_POST_URLS
from tests.helpers.logfire_capture import capture_logs

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable, Awaitable

    from openai.types.responses.response_input_param import EasyInputMessageParam

    from discordbot.typings.llm import LLMConfig
    from discordbot.cogs.gen_reply.cog import ReplyGeneratorCogs


@pytest.fixture
def quiet_turn(monkeypatch: pytest.MonkeyPatch, no_memory_review: None) -> None:
    """A whole turn with a canned answer, no memory review and no status reactions."""

    async def silent_reaction(**kwargs: object) -> object:
        """Reports the status reaction as applied without touching the message."""
        return kwargs["emoji"]

    _install_streamer(monkeypatch=monkeypatch)
    monkeypatch.setattr("discordbot.utils.reactions.update_reaction", silent_reaction)


def _failing_build(
    after: Callable[[], Awaitable[object]],
) -> Callable[..., Awaitable[ReplyContext]]:
    """A `ReplyContextBuilder.build` that waits for the route's picks, then `after`, then fails."""

    async def build(
        self: ReplyContextBuilder, recall_picks: asyncio.Future[list[str]], **kwargs: object
    ) -> ReplyContext:
        """Fails once the route has resolved and `after` has returned."""
        del self, kwargs
        await recall_picks
        await after()
        raise RuntimeError("prep exploded")

    return build


def _delayed_build(seconds: float) -> Callable[..., Awaitable[ReplyContext]]:
    """The real `ReplyContextBuilder.build`, started `seconds` after the route's picks land."""
    real_build = ReplyContextBuilder.build

    async def build(
        self: ReplyContextBuilder,
        **kwargs: Any,  # noqa: ANN401 -- forwarded untouched to the real build
    ) -> ReplyContext:
        """Holds the build back past the picks, then runs it unchanged."""
        await kwargs["recall_picks"]
        await asyncio.sleep(seconds)
        return await real_build(self, **kwargs)

    return build


class _CleanupBoundBuilder:
    """A link builder that, once cancelled, holds its cleanup open until `release` is set."""

    def __init__(self) -> None:
        """Starts with no cancellation seen and the cleanup unreleased."""
        self.cleanup_started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancellations = 0

    async def __call__(self, **kwargs: object) -> list[EasyInputMessageParam]:
        """Sleeps until cancelled, then counts every cancellation its cleanup receives."""
        del kwargs
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            self.cancellations += 1
            self.cleanup_started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancellations += 1
                raise
            raise
        return []


@pytest.mark.parametrize(
    argnames=("route", "expected_call"),
    argvalues=[("IMAGE", "handle_image"), ("VIDEO", "handle_video"), ("QA", "stream_answer")],
)
async def test_gen_reply_on_message_dispatches_routes(  # noqa: PLR0915 -- orchestrates per-route stubs
    monkeypatch: pytest.MonkeyPatch, route: Literal["IMAGE", "VIDEO", "QA"], expected_call: str
) -> None:
    """Verifies on_message dispatches each route to the expected handler."""
    cog = _cog()
    # A deployment where every route can run: VIDEO needs a Gemini key, while the QA video
    # marker's switch is off to show it gates only the marker, never the route.
    cog.config.gemini_api_key = "test-key"
    cog.config.inline_video_enabled = False
    # Distinctive non-fallback grade so the effort reaching the answer model is checked to
    # be the graded value, not the "high" default a failed parse would also produce.
    _recorded(cog).responses.output_parsed = RouteClassification(decision=route, effort="low")
    calls: list[str] = []
    prompts: list[str] = []
    prep_requests: list[int] = []
    prepared_context = ReplyContext()

    async def fake_prepare(
        self: object, history_limit: int, parts_task: object, recall: object, recall_picks: object
    ) -> ReplyContext:
        """Records context requests while staying off the memory and history paths."""
        del self, parts_task, recall, recall_picks
        prep_requests.append(history_limit)
        return prepared_context

    async def fake_reaction(
        message: FakeMessage, bot_user: object, emoji: str, previous: str | None = None
    ) -> str:
        """Records reaction state transitions."""
        calls.append(f"reaction:{emoji}")
        return emoji

    async def fake_image_handler(
        self: object, user_prompt: str, context_task: asyncio.Task[ReplyContext]
    ) -> None:
        """Records image handler dispatch and drains the handed-over context task."""
        del self
        contexts.append(await context_task)
        prompts.append(user_prompt)
        calls.append("handle_image")

    async def fake_video_handler(
        self: object, user_prompt: str, context_task: asyncio.Task[ReplyContext]
    ) -> None:
        """Records video handler dispatch and drains the handed-over context task."""
        del self
        contexts.append(await context_task)
        prompts.append(user_prompt)
        calls.append("handle_video")

    effort_flags: list[str] = []
    contexts: list[ReplyContext] = []

    async def fake_message_handler(  # noqa: PLR0913 -- stub mirrors AnswerTurn.stream_answer's signature
        self: object,
        system_prompt: str,
        context: ReplyContext,
        effort: str = "high",
        allow_research: bool = False,
        yt_url: str | None = None,
    ) -> None:
        """Records slow message handler dispatch."""
        del yt_url, allow_research
        calls.append("stream_answer")
        effort_flags.append(effort)
        contexts.append(context)

    monkeypatch.setattr(ReplyContextBuilder, "build", fake_prepare)
    monkeypatch.setattr("discordbot.utils.reactions.update_reaction", fake_reaction)
    monkeypatch.setattr(MediaReplyRoutes, "handle_image", fake_image_handler)
    monkeypatch.setattr(MediaReplyRoutes, "handle_video", fake_video_handler)
    monkeypatch.setattr(AnswerTurn, "stream_answer", fake_message_handler)

    message = FakeMessage(content="<@!999> hello", author=FakeAuthor(user_id=1))
    await cog.on_message(message=as_message(fake=message))
    assert expected_call in calls
    # Route, effort and recall are one triage call on every route, IMAGE and VIDEO included.
    assert len(_recorded(cog).responses.parse_models) == 1
    assert calls[-1] == f"reaction:{DONE_EMOJI}"
    # Every route consumes the one speculative context, IMAGE/VIDEO once their media is on
    # screen, so each issues exactly one prep request and receives that very context instance.
    assert Counter(prep_requests) == Counter([HISTORY_MESSAGE_LIMIT])
    assert len(contexts) == 1
    assert contexts[0] is prepared_context
    if route in {"IMAGE", "VIDEO"}:
        assert prompts == ["hello"]
        assert effort_flags == []
    else:
        # The route's grade flows end-to-end into the QA answer model.
        assert effort_flags == ["low"]


async def test_a_reply_records_the_route_it_took(
    monkeypatch: pytest.MonkeyPatch, usage_log_isolated_dir: Path
) -> None:
    """One reply turn is one usage record, named after the route that served it."""
    cog = _cog()

    async def fake_message_handler(self: object, **kwargs: object) -> None:
        """Stands in for the answer so the turn completes without an LLM call."""
        del kwargs

    monkeypatch.setattr(
        RouteClassifier, "classify", _classify_stub(route=RouteClassification(decision="QA"))
    )
    monkeypatch.setattr(ReplyContextBuilder, "build", _build_stub(context=ReplyContext()))
    monkeypatch.setattr(AnswerTurn, "stream_answer", fake_message_handler)

    message = FakeMessage(content="<@999> recap", author=FakeAuthor(user_id=7))
    await cog.on_message(message=as_message(fake=message))

    (record,) = usage_records(directory=usage_log_isolated_dir)
    assert (record["kind"], record["name"]) == ("reply", "QA")
    assert record["user_id"] == 7
    assert message.guild is not None
    assert record["guild_id"] == message.guild.id

    # The empty-prompt `?` reply runs no model and takes no route, so it is a misfire
    # rather than a conversation and stays out of the records.
    empty = FakeMessage(content="<@999>", author=FakeAuthor(user_id=7))
    empty.guild = None
    await cog.on_message(message=as_message(fake=empty))

    assert len(usage_records(directory=usage_log_isolated_dir)) == 1


async def test_a_failed_reply_records_that_it_never_routed(
    monkeypatch: pytest.MonkeyPatch, usage_log_isolated_dir: Path
) -> None:
    """Someone still talked to the bot, so a failure before the router is still recorded."""
    cog = _cog()

    monkeypatch.setattr(RouteClassifier, "classify", _classify_stub(route=RuntimeError("boom")))
    monkeypatch.setattr(ReplyContextBuilder, "build", _build_stub(context=ReplyContext()))

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=7))
    await cog.on_message(message=as_message(fake=message))

    (record,) = usage_records(directory=usage_log_isolated_dir)
    assert (record["kind"], record["name"]) == ("reply", UNROUTED_REPLY)


async def test_a_failed_route_cancels_the_build_waiting_on_its_picks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The picks are never resolved when the route raises, so the turn must cancel the build."""
    cog = _cog()
    build_cancelled = asyncio.Event()

    async def waiting_prepare(
        self: object, recall_picks: asyncio.Future[list[str]], **kwargs: object
    ) -> ReplyContext:
        """Waits on the picks the way the real build does, and notes being cancelled."""
        del self, kwargs
        try:
            await recall_picks
        except asyncio.CancelledError:
            build_cancelled.set()
            raise
        return ReplyContext()

    monkeypatch.setattr(RouteClassifier, "classify", _classify_stub(route=RuntimeError("boom")))
    monkeypatch.setattr(ReplyContextBuilder, "build", waiting_prepare)

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=7))
    # Bounded so a turn that leaves the build waiting fails here instead of hanging the suite.
    await asyncio.wait_for(fut=cog.on_message(message=as_message(fake=message)), timeout=5)

    assert build_cancelled.is_set()


@pytest.mark.usefixtures("quiet_turn")
async def test_on_message_answers_a_keyless_video_route_as_qa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a Gemini key a VIDEO route is answered as QA, never handed to the video handler.

    That handler pays for the director and then renders direct to Google, so on a keyless
    deployment it could only fail the whole turn.
    """
    cog = _cog()
    video_prompts: list[str] = []

    async def fake_video_handler(
        self: object, user_prompt: str, context_task: asyncio.Task[ReplyContext]
    ) -> None:
        """Records the dispatch and drains the handed-over context task."""
        del self
        await context_task
        video_prompts.append(user_prompt)

    monkeypatch.setattr(
        RouteClassifier, "classify", _classify_stub(route=RouteClassification(decision="VIDEO"))
    )
    monkeypatch.setattr(ReplyContextBuilder, "build", _build_stub(context=ReplyContext()))
    monkeypatch.setattr(MediaReplyRoutes, "handle_video", fake_video_handler)
    streams = _install_streamer(monkeypatch=monkeypatch)

    message = FakeMessage(content="<@!999> make a video of a cat", author=FakeAuthor(user_id=1))
    await cog.on_message(message=as_message(fake=message))

    assert video_prompts == []
    assert len(streams) == 1


def _link_config(gemini_api_key: str) -> LLMConfig:
    """The config fields a QA reply carrying a linked post actually reads."""
    return _config_stub(
        douyin_video_enabled=True,
        bilibili_video_enabled=True,
        file_api_enabled=True,
        gemini_api_key=gemini_api_key,
    )


class _LinkCase(BaseModel):
    """How the pipeline tests drive one registered link source."""

    builder: str = Field(..., description="The `registry` global its builder is patched onto.")
    non_post_url: str = Field(
        ..., description="A link on the same site that names no post, so nothing is read."
    )
    emoji: str = Field(..., description="The marker a read of it leaves on the message.")
    reads_replied_to: bool = Field(
        ..., description="Whether a link in the replied-to message is read as well."
    )
    media_switch: str = Field(..., description="The config field that turns its media ingest off.")


# Keyed by registry name. Every family below is parametrized over the registry itself, so a source
# added there without a row here fails at collection instead of going untested.
_LINK_CASES: dict[str, _LinkCase] = {
    "threads": _LinkCase(
        builder="build_threads_context_messages",
        non_post_url="https://www.threads.com/@user",
        emoji=THREADS_EMOJI,
        reads_replied_to=True,
        media_switch="file_api_enabled",
    ),
    "facebook": _LinkCase(
        builder="build_facebook_context_messages",
        non_post_url="https://www.facebook.com/groups/123/",
        emoji=FACEBOOK_EMOJI,
        reads_replied_to=True,
        media_switch="file_api_enabled",
    ),
    "instagram": _LinkCase(
        builder="build_instagram_context_messages",
        non_post_url="https://www.instagram.com/instagram/",
        emoji=INSTAGRAM_EMOJI,
        reads_replied_to=True,
        media_switch="file_api_enabled",
    ),
    "twitter": _LinkCase(
        builder="build_twitter_context_messages",
        non_post_url="https://x.com/Dbacks",
        emoji=TWITTER_EMOJI,
        reads_replied_to=True,
        media_switch="file_api_enabled",
    ),
    "douyin": _LinkCase(
        builder="build_douyin_context_messages",
        non_post_url="https://www.douyin.com/user/MS4wLjABAAAAxyz",
        emoji=DOUYIN_EMOJI,
        reads_replied_to=False,
        media_switch="douyin_video_enabled",
    ),
    "bilibili": _LinkCase(
        builder="build_bilibili_context_messages",
        non_post_url="https://live.bilibili.com/12345",
        emoji=BILIBILI_EMOJI,
        reads_replied_to=False,
        media_switch="bilibili_video_enabled",
    ),
}


_LINK_POST_BODY = "MOCK POST BODY"


class _FakeLinkBuilder:
    """Stands in for one source's builder, returning the block a readable post produces."""

    def __init__(self, source: str, delay: float) -> None:
        """Answers as `source`, `delay` seconds after each call."""
        self.source = source
        self.delay = delay
        self.calls: list[dict[str, object]] = []
        self.cancellations = 0

    async def __call__(self, **kwargs: object) -> list[EasyInputMessageParam]:
        """Records the call's kwargs, and every cancellation that lands while it waits."""
        self.calls.append(kwargs)
        if self.delay:
            try:
                await asyncio.sleep(self.delay)
            except asyncio.CancelledError:
                self.cancellations += 1
                raise
        return link_context_blocks(
            separator=LINK_SOURCE_BLOCKS[self.source].separators[0], text=_LINK_POST_BODY
        )


def _patch_link_builder(
    monkeypatch: pytest.MonkeyPatch, source: str, delay: float = 0
) -> _FakeLinkBuilder:
    """Puts a `_FakeLinkBuilder` where the registry looks `source`'s builder up."""
    builder = _FakeLinkBuilder(source=source, delay=delay)
    monkeypatch.setattr(
        f"discordbot.cogs.gen_reply.link_sources.registry.{_LINK_CASES[source].builder}", builder
    )
    return builder


def _link_cog(
    sources: list[str], decision: str = "QA", gemini_api_key: str = "key"
) -> ReplyGeneratorCogs:
    """A cog under `_link_config` whose route call picks `decision` and selects `sources`.

    The config and the toolkit carry the same Gemini key, as they do when the cog builds its
    toolkit from its own config.
    """
    cog = _cog()
    _recorded(cog).responses.output_parsed = RouteClassification.model_validate({
        "decision": decision,
        "link_context_sources": sources,
    })
    cog.config = _link_config(gemini_api_key=gemini_api_key)
    cog.toolkit.gemini_api_key = gemini_api_key
    return cog


def _link_message(text: str) -> FakeMessage:
    """A message addressed to the bot, so the whole turn runs on it."""
    return FakeMessage(content=f"<@999> {text}", author=FakeAuthor(user_id=1))


@pytest.mark.parametrize("name", _LINK_SOURCES)
@pytest.mark.usefixtures("quiet_turn")
async def test_on_message_does_not_start_incidental_link_context(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """An incidental registered link starts no source work and injects no source claim."""
    cog = _link_cog(sources=[])
    builder = _patch_link_builder(monkeypatch=monkeypatch, source=name)

    await cog.on_message(
        message=as_message(fake=_link_message(text=f"unrelated question {SAMPLE_POST_URLS[name]}"))
    )

    assert builder.calls == []
    assert not has_link_context_block(
        request=request_input(responses=_recorded(cog).responses), source=name
    )


@pytest.mark.parametrize("name", _LINK_SOURCES)
@pytest.mark.usefixtures("quiet_turn")
async def test_on_message_injects_a_selected_link_source_before_current(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """The post the router selected reaches the answer input, ahead of the current message.

    The message also gets the source's persistent marker, the same one its expansion cog adds; a
    source with no expansion cog is marked by this path alone.
    """
    case = _LINK_CASES[name]
    cog = _link_cog(sources=[name])
    builder = _patch_link_builder(monkeypatch=monkeypatch, source=name)
    message = _link_message(text=f"這在講什麼 {SAMPLE_POST_URLS[name]}")

    await cog.on_message(message=as_message(fake=message))

    (call,) = builder.calls
    assert call["url"] == SAMPLE_POST_URLS[name]
    assert call["gemini_client"] is cog.toolkit.gemini_client
    assert call["allow_media_ingest"] is True
    answer = request_input(responses=_recorded(cog).responses)
    assert extract_link_context_block(request=answer, source=name) == _LINK_POST_BODY
    assert block_index(request=answer, kind=name) < block_index(request=answer, kind="current")
    assert case.emoji in message.added_reactions


@pytest.mark.parametrize("name", _LINK_SOURCES)
@pytest.mark.usefixtures("quiet_turn")
async def test_on_message_hands_each_link_build_the_deadline_it_is_held_to(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """A builder is told the very deadline its build is cancelled at.

    Its media step stops short of that deadline so the post's text still comes back; told any
    other one, a slow read would again leave the step running into the cancellation.
    """
    cog = _link_cog(sources=[name])
    builder = _patch_link_builder(monkeypatch=monkeypatch, source=name)
    enforced: list[float] = []

    async def recording_run_until_deadline(
        awaitable: Awaitable[list[EasyInputMessageParam]], deadline: float
    ) -> list[EasyInputMessageParam]:
        """Notes the deadline the pipeline holds the build to, then enforces it as usual."""
        enforced.append(deadline)
        return await run_until_deadline(awaitable=awaitable, deadline=deadline)

    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.pipeline.run_until_deadline", recording_run_until_deadline
    )

    await cog.on_message(
        message=as_message(fake=_link_message(text=f"這在講什麼 {SAMPLE_POST_URLS[name]}"))
    )

    (call,) = builder.calls
    assert enforced == [call["deadline"]]


@pytest.mark.parametrize("name", _LINK_SOURCES)
@pytest.mark.usefixtures("quiet_turn")
async def test_on_message_reads_a_linked_post_without_a_gemini_key(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """A keyless deployment still gets the linked post's text, not a generic failure.

    The direct client raises on an empty key, so touching it while assembling the builder call
    would fail the whole reply before the builder's own text-only degradation could run. The
    ingest flag needs the key as well, or a builder would be told it may upload while holding no
    client to upload with.
    """
    cog = _link_cog(sources=[name], gemini_api_key="")
    builder = _patch_link_builder(monkeypatch=monkeypatch, source=name)

    await cog.on_message(
        message=as_message(fake=_link_message(text=f"這在講什麼 {SAMPLE_POST_URLS[name]}"))
    )

    (call,) = builder.calls
    assert call["gemini_client"] is None
    assert call["allow_media_ingest"] is False
    assert has_link_context_block(
        request=request_input(responses=_recorded(cog).responses), source=name
    )


@pytest.mark.parametrize("name", _LINK_SOURCES)
@pytest.mark.usefixtures("quiet_turn")
async def test_on_message_link_media_ingest_kill_switch(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """With the source's switch off the builder still runs, but is told not to fetch the media."""
    case = _LINK_CASES[name]
    cog = _link_cog(sources=[name])
    monkeypatch.setattr(cog.config, case.media_switch, False)
    builder = _patch_link_builder(monkeypatch=monkeypatch, source=name)

    await cog.on_message(
        message=as_message(fake=_link_message(text=f"這在講什麼 {SAMPLE_POST_URLS[name]}"))
    )

    assert [call["allow_media_ingest"] for call in builder.calls] == [False]


@pytest.mark.parametrize("name", _LINK_SOURCES)
@pytest.mark.usefixtures("quiet_turn")
async def test_on_message_skips_a_link_that_names_no_post(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """A profile, group page or live room is no post, so reading it would only waste a request."""
    cog = _link_cog(sources=[name])
    builder = _patch_link_builder(monkeypatch=monkeypatch, source=name)

    await cog.on_message(
        message=as_message(fake=_link_message(text=f"這是誰 {_LINK_CASES[name].non_post_url}"))
    )

    assert builder.calls == []
    assert not has_link_context_block(
        request=request_input(responses=_recorded(cog).responses), source=name
    )


@pytest.mark.parametrize("name", _LINK_SOURCES)
@pytest.mark.usefixtures("quiet_turn")
async def test_on_message_does_not_start_link_context_on_image_route(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """A non-QA route never starts link work even if the router selects that source."""
    cog = _link_cog(sources=[name], decision="IMAGE")
    builder = _patch_link_builder(monkeypatch=monkeypatch, source=name)

    async def drain_context(
        self: MediaReplyRoutes, context_task: asyncio.Task[ReplyContext], **kwargs: object
    ) -> None:
        """Accepts the dispatched image request."""
        del self, kwargs
        await context_task

    monkeypatch.setattr(MediaReplyRoutes, "handle_image", drain_context)

    await cog.on_message(
        message=as_message(fake=_link_message(text=f"畫這個 {SAMPLE_POST_URLS[name]}"))
    )

    assert builder.calls == []


@pytest.mark.parametrize("name", _LINK_SOURCES)
@pytest.mark.usefixtures("quiet_turn")
async def test_on_message_link_grace_timeout_injects_notice(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """A build slower than the post-route grace injects the source's timeout notice instead.

    The notice keeps the model from claiming it cannot open the link, and the answer still
    streams.
    """
    cog = _link_cog(sources=[name])
    monkeypatch.setattr("discordbot.cogs.gen_reply.pipeline.LINK_CONTEXT_GRACE_SECONDS", 0.01)
    _patch_link_builder(monkeypatch=monkeypatch, source=name, delay=5)

    await cog.on_message(
        message=as_message(fake=_link_message(text=f"這在講什麼 {SAMPLE_POST_URLS[name]}"))
    )

    assert has_timeout_notice(
        request=request_input(responses=_recorded(cog).responses), source=name
    )


@pytest.mark.parametrize("name", _LINK_SOURCES)
@pytest.mark.usefixtures("quiet_turn")
async def test_on_message_reads_a_replied_to_link_only_for_a_discussion_source(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """Mentioning the bot in a reply to someone else's link reads it only where that adds news.

    A discussion source reads the comments its expansion never shows, so a reply asking about
    them has nothing else to answer from, the way a replied-to YouTube link is watched. A clip
    would only be read a second time, and stays on the current message; Douyin and Bilibili are
    rate-limit sensitive besides, so a passing mention one hop away is not worth their fetch.
    """
    case = _LINK_CASES[name]
    cog = _link_cog(sources=[name])
    builder = _patch_link_builder(monkeypatch=monkeypatch, source=name)
    parent = FakeMessage(
        content=f"看看這篇 {SAMPLE_POST_URLS[name]}", author=FakeAuthor(user_id=4)
    )
    message = _link_message(text="這篇底下在吵什麼")
    message.reference = FakeReference(resolved=parent)

    await cog.on_message(message=as_message(fake=message))

    answer = request_input(responses=_recorded(cog).responses)
    if case.reads_replied_to:
        assert [call["url"] for call in builder.calls] == [SAMPLE_POST_URLS[name]]
        assert extract_link_context_block(request=answer, source=name) == _LINK_POST_BODY
    else:
        assert builder.calls == []
        assert not has_link_context_block(request=answer, source=name)


@pytest.mark.usefixtures("quiet_turn")
async def test_on_message_link_context_grace_starts_when_route_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A builder that finishes after the deadline cannot win while preparation is still running."""
    cog = _link_cog(sources=["douyin"])
    monkeypatch.setattr("discordbot.cogs.gen_reply.pipeline.LINK_CONTEXT_GRACE_SECONDS", 0.12)
    # Finishes after the shared grace but before the delayed resolver observes it.
    builder = _patch_link_builder(monkeypatch=monkeypatch, source="douyin", delay=0.14)
    monkeypatch.setattr(ReplyContextBuilder, "build", _delayed_build(seconds=0.18))

    await cog.on_message(
        message=as_message(fake=_link_message(text=f"這在講什麼 {SAMPLE_POST_URLS['douyin']}"))
    )

    answer = request_input(responses=_recorded(cog).responses)
    assert has_timeout_notice(request=answer, source="douyin")
    assert builder.cancellations == 1


@pytest.mark.usefixtures("quiet_turn")
async def test_on_message_keeps_link_context_finished_before_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A builder completed before the deadline remains usable after delayed preparation."""
    cog = _link_cog(sources=["douyin"])
    monkeypatch.setattr("discordbot.cogs.gen_reply.pipeline.LINK_CONTEXT_GRACE_SECONDS", 0.12)
    _patch_link_builder(monkeypatch=monkeypatch, source="douyin")
    monkeypatch.setattr(ReplyContextBuilder, "build", _delayed_build(seconds=0.18))

    await cog.on_message(
        message=as_message(fake=_link_message(text=f"這在講什麼 {SAMPLE_POST_URLS['douyin']}"))
    )

    answer = request_input(responses=_recorded(cog).responses)
    assert has_link_context_block(request=answer, source="douyin")
    assert not has_timeout_notice(request=answer, source="douyin")


def _build_after(after: Callable[[], Awaitable[object]]) -> Callable[..., Awaitable[ReplyContext]]:
    """A `ReplyContextBuilder.build` that finishes once the route's picks and `after` resolve."""

    async def build(
        self: ReplyContextBuilder, recall_picks: asyncio.Future[list[str]], **kwargs: object
    ) -> ReplyContext:
        """Waits on the picks and `after`, then returns an empty context."""
        del self, kwargs
        await recall_picks
        await after()
        return ReplyContext()

    return build


@pytest.mark.usefixtures("quiet_turn")
async def test_on_message_waits_for_deadline_cancelled_link_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolver lets a deadline-cancelled builder finish cleanup before injecting its notice."""
    cog = _link_cog(sources=["douyin"])
    monkeypatch.setattr("discordbot.cogs.gen_reply.pipeline.LINK_CONTEXT_GRACE_SECONDS", 0.01)
    builder = _CleanupBoundBuilder()
    resolving = asyncio.Event()

    async def signalling_await(**kwargs: Any) -> list[EasyInputMessageParam]:  # noqa: ANN401 -- forwarded untouched to the real resolver
        """Marks the turn as waiting on the deadline-cancelled build."""
        resolving.set()
        return await await_deadline_bound_task(**kwargs)

    monkeypatch.setattr(
        ReplyContextBuilder, "build", _build_after(after=builder.cleanup_started.wait)
    )
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.link_sources.registry.build_douyin_context_messages", builder
    )
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.pipeline.await_deadline_bound_task", signalling_await
    )

    message = _link_message(text=f"這在講什麼 {SAMPLE_POST_URLS['douyin']}")
    message_task = asyncio.create_task(coro=cog.on_message(message=as_message(fake=message)))
    try:
        await asyncio.wait_for(fut=resolving.wait(), timeout=1)
        await asyncio.sleep(0)
        assert builder.cancellations == 1
        assert not message_task.done()
    finally:
        builder.release.set()
        await message_task

    answer = request_input(responses=_recorded(cog).responses)
    assert has_timeout_notice(request=answer, source="douyin")


@pytest.mark.usefixtures("quiet_turn")
async def test_on_message_cancellation_waits_for_deadline_cancelled_link_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outer cancellation waits for a deadline-owned builder cleanup before propagating."""
    cog = _link_cog(sources=["douyin"])
    monkeypatch.setattr("discordbot.cogs.gen_reply.pipeline.LINK_CONTEXT_GRACE_SECONDS", 0.01)
    builder = _CleanupBoundBuilder()
    resolving = asyncio.Event()

    async def signalling_await(**kwargs: Any) -> list[EasyInputMessageParam]:  # noqa: ANN401 -- forwarded untouched to the real resolver
        """Marks the turn as waiting on the deadline-cancelled build."""
        resolving.set()
        return await await_deadline_bound_task(**kwargs)

    monkeypatch.setattr(
        ReplyContextBuilder, "build", _build_after(after=builder.cleanup_started.wait)
    )
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.link_sources.registry.build_douyin_context_messages", builder
    )
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.pipeline.await_deadline_bound_task", signalling_await
    )

    message = _link_message(text=f"這在講什麼 {SAMPLE_POST_URLS['douyin']}")
    message_task = asyncio.create_task(coro=cog.on_message(message=as_message(fake=message)))
    await asyncio.wait_for(fut=resolving.wait(), timeout=1)
    await asyncio.sleep(0)
    message_task.cancel()
    try:
        await asyncio.sleep(0)
        assert builder.cancellations == 1
        assert not message_task.done()
    finally:
        builder.release.set()

    with pytest.raises(asyncio.CancelledError):
        await message_task
    assert builder.cancellations == 1


async def test_deadline_bound_task_outer_cancel_before_deadline_cancels_builder() -> None:
    """An outer cancellation before the deadline owns and drains the still-running builder."""
    builder_started = asyncio.Event()
    builder_cancelled = asyncio.Event()

    async def pending_builder() -> None:
        """Runs until the resolver cancellation owns it."""
        builder_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            builder_cancelled.set()
            raise

    event_loop = asyncio.get_running_loop()
    deadline = event_loop.time() + 5
    builder_task = asyncio.create_task(
        coro=run_until_deadline(awaitable=pending_builder(), deadline=deadline)
    )
    resolver_task = asyncio.create_task(
        coro=await_deadline_bound_task(
            task=builder_task, deadline=deadline, label="test", message_id=1
        )
    )
    await asyncio.wait_for(fut=builder_started.wait(), timeout=1)
    resolver_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await resolver_task
    assert builder_cancelled.is_set()
    assert builder_task.done()


@pytest.mark.usefixtures("quiet_turn")
async def test_on_message_cancelled_link_wait_logs_builder_failure_under_its_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A builder failing while a cancelled turn drains it is logged with that turn's message id."""
    cog = _link_cog(sources=["douyin"])
    builder_started = asyncio.Event()
    resolving = asyncio.Event()

    async def failing_builder(**kwargs: object) -> list[EasyInputMessageParam]:
        """Runs until cancelled, then fails instead of cancelling."""
        del kwargs
        builder_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise RuntimeError("builder cleanup failed") from None
        return []

    async def signalling_await(**kwargs: Any) -> list[EasyInputMessageParam]:  # noqa: ANN401 -- forwarded untouched to the real resolver
        """Marks the turn as waiting on the build, then waits exactly as the pipeline does."""
        resolving.set()
        return await await_deadline_bound_task(**kwargs)

    warned = capture_logs(monkeypatch=monkeypatch, level="warn")
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.link_sources.registry.build_douyin_context_messages",
        failing_builder,
    )
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.pipeline.await_deadline_bound_task", signalling_await
    )

    message = _link_message(text=f"這在講什麼 {SAMPLE_POST_URLS['douyin']}")
    message_task = asyncio.create_task(coro=cog.on_message(message=as_message(fake=message)))
    await asyncio.wait_for(fut=builder_started.wait(), timeout=1)
    await asyncio.wait_for(fut=resolving.wait(), timeout=1)
    message_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await message_task
    assert [
        fields["message_id"] for _, fields in warned if fields.get("task_label") == "douyin"
    ] == [message.id]


async def test_run_until_deadline_keeps_result_completed_before_delayed_resume() -> None:
    """A completed builder wins even if a briefly blocked loop resumes its waiter after deadline."""
    event_loop = asyncio.get_running_loop()
    result_future = event_loop.create_future()
    result_future.add_done_callback(lambda _: time.sleep(0.05))
    event_loop.call_soon(result_future.set_result, "ready")

    result = await run_until_deadline(awaitable=result_future, deadline=event_loop.time() + 0.02)

    assert result == "ready"


@pytest.mark.usefixtures("quiet_turn")
async def test_on_message_selected_link_contexts_share_one_post_route_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sequential resolution cannot grant every selected builder a fresh timeout."""
    cog = _link_cog(sources=["threads", "douyin"])
    monkeypatch.setattr("discordbot.cogs.gen_reply.pipeline.LINK_CONTEXT_GRACE_SECONDS", 0.12)
    # Threads uses most of the shared budget before the first registry entry resolves; Douyin
    # would finish under a second fresh timeout, but not under the same shared deadline.
    _patch_link_builder(monkeypatch=monkeypatch, source="threads", delay=0.14)
    _patch_link_builder(monkeypatch=monkeypatch, source="douyin", delay=0.22)

    urls = f"{SAMPLE_POST_URLS['threads']} {SAMPLE_POST_URLS['douyin']}"
    await cog.on_message(message=as_message(fake=_link_message(text=f"這兩個在講什麼 {urls}")))

    answer = request_input(responses=_recorded(cog).responses)
    assert has_timeout_notice(request=answer, source="threads")
    assert has_timeout_notice(request=answer, source="douyin")


@pytest.mark.usefixtures("quiet_turn")
async def test_on_message_finally_backstop_cancels_link_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure after QA routing still cancels its selected in-flight link build."""
    cog = _link_cog(sources=["bilibili"])
    builder = _patch_link_builder(monkeypatch=monkeypatch, source="bilibili", delay=30)
    # Yields once after the picks, so the selected builder is in flight when the build fails.
    monkeypatch.setattr(
        ReplyContextBuilder, "build", _failing_build(after=lambda: asyncio.sleep(0))
    )

    await cog.on_message(
        message=as_message(fake=_link_message(text=f"這在講什麼 {SAMPLE_POST_URLS['bilibili']}"))
    )

    assert builder.cancellations == 1


@pytest.mark.usefixtures("quiet_turn")
async def test_on_message_finally_waits_for_deadline_owned_link_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prep failure drains a deadline-owned builder cleanup without cancelling it twice."""
    cog = _link_cog(sources=["bilibili"])
    monkeypatch.setattr("discordbot.cogs.gen_reply.pipeline.LINK_CONTEXT_GRACE_SECONDS", 0.01)
    builder = _CleanupBoundBuilder()
    draining = asyncio.Event()

    async def signalling_discard(**kwargs: Any) -> None:  # noqa: ANN401 -- forwarded untouched to the real drain
        """Marks the finally block as draining the link tasks."""
        draining.set()
        await discard_link_tasks(**kwargs)

    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.link_sources.registry.build_bilibili_context_messages", builder
    )
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.pipeline.discard_link_tasks", signalling_discard
    )
    # Fails while the selected builder still owns its deadline cancellation cleanup.
    monkeypatch.setattr(
        ReplyContextBuilder, "build", _failing_build(after=builder.cleanup_started.wait)
    )

    message = _link_message(text=f"這在講什麼 {SAMPLE_POST_URLS['bilibili']}")
    message_task = asyncio.create_task(coro=cog.on_message(message=as_message(fake=message)))
    try:
        await asyncio.wait_for(fut=draining.wait(), timeout=1)
        await asyncio.sleep(0)
        assert builder.cancellations == 1
        assert not message_task.done()
    finally:
        builder.release.set()
        await message_task

    assert builder.cancellations == 1


@pytest.mark.parametrize(
    ("selected_sources", "expected_order"),
    [
        (["threads", "douyin", "bilibili"], ["threads", "douyin", "bilibili"]),
        (["bilibili", "threads"], ["threads", "bilibili"]),
    ],
)
@pytest.mark.usefixtures("quiet_turn")
async def test_on_message_orders_selected_link_blocks_in_registry_order(
    monkeypatch: pytest.MonkeyPatch, selected_sources: list[str], expected_order: list[str]
) -> None:
    """Selected sources are injected in registry order, not URL or router-return order.

    The URLs are pasted in reverse registry order on purpose: the splice must follow
    `LINK_CONTEXT_SOURCES` order (threads, douyin, bilibili), not text order, so the answer
    input stays deterministic however the user arranged the links.
    """
    cog = _link_cog(sources=selected_sources)
    patched = ("threads", "douyin", "bilibili")
    for name in patched:
        _patch_link_builder(monkeypatch=monkeypatch, source=name)

    urls = " ".join(SAMPLE_POST_URLS[name] for name in reversed(patched))
    await cog.on_message(message=as_message(fake=_link_message(text=f"這幾個在講什麼 {urls}")))

    answer = request_input(responses=_recorded(cog).responses)
    positions = [block_index(request=answer, kind=name) for name in expected_order]
    assert positions == sorted(positions)
    assert positions[-1] < block_index(request=answer, kind="current")
    for name in set(patched) - set(expected_order):
        assert not has_link_context_block(request=answer, source=name)


async def test_handle_message_reply_uses_route_effort() -> None:
    """The answer request's reasoning effort follows the route decision."""
    cog = _cog()
    _recorded(cog).responses.output_parsed = RouteClassification(decision="QA", effort="low")
    message = FakeMessage(content="<@999> why", author=FakeAuthor(user_id=1))

    await _run_pipeline(cog=cog, message=message)

    assert _recorded(cog).responses.create_reasonings[-1]["effort"] == "low"
