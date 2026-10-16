"""Tests for the triage call that picks a reply's route."""

from __future__ import annotations

# openai 3.x builds its exceptions on httpx2, so a request or response handed to one has to
# come from there. google-genai is still on httpx 0.x, and both live in the environment.
import httpx2
from openai import APIError, APITimeoutError, BadRequestError
import pytest
from pydantic import ValidationError

from discordbot.typings.models import RouteClassification, RecallRouteClassification
from discordbot.cogs.gen_reply.recall import (
    RecallCandidate,
    render_server_memory_block,
    render_callable_users_block,
)
from discordbot.services.memory.store import user_scope
from discordbot.cogs.gen_reply.prompts import ROUTE_RECALL_SECTION, route_prompt

from tests.helpers.casting import as_message
from tests.helpers.llm_input import (
    request_input,
    extract_callable_user_ids,
    extract_user_memory_blocks,
)
from tests.helpers.reply_fakes import (
    _ROUTE_CURRENT,
    _ROUTE_REFERENCE,
    FakeAuthor,
    FakeMessage,
    FakeAttachment,
    _cog,
    _route,
    _recorded,
    _seed_fact,
    _classifier,
    _seed_alias,
    _text_event,
    _run_pipeline,
    _completed_event,
)
from tests.helpers.logfire_capture import capture_logs


@pytest.mark.usefixtures("no_memory_review")
async def test_an_unparseable_route_keeps_the_author_memory_and_drops_only_the_picks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A triage answer outside its schema loses the optional pick, never the turn."""
    cog = _cog()
    _seed_fact(scope=user_scope(user_id=1), text="甲")
    _seed_fact(scope=user_scope(user_id=42), text="不該注入的第三人")
    _seed_alias(subject_id=42, text="Boss(社群暱稱:李董)")

    with pytest.raises(ValidationError) as invalid:
        RecallRouteClassification.model_validate(obj={"decision": "QA", "recall_user_ids": 42})

    async def reject(**kwargs: object) -> object:
        """Fails the way `responses.parse` does on a reply outside the schema."""
        del kwargs
        raise invalid.value

    monkeypatch.setattr(_recorded(cog).responses, "parse", reject)

    _recorded(cog).responses.stream_queue = [
        [_text_event(delta="照常回答"), _completed_event(input_tokens=5, output_tokens=6)]
    ]

    message = FakeMessage(content="<@999> 李董在嗎", author=FakeAuthor(user_id=1))
    await _run_pipeline(cog=cog, message=message)

    # The answer request still ran with the deterministic memory only.
    assert (message.replies[0].content or "").startswith("照常回答")
    answer = request_input(responses=_recorded(cog).responses)
    assert "甲" in (extract_user_memory_blocks(request=answer).get(1) or "")
    assert 42 not in extract_user_memory_blocks(request=answer)


async def test_route_classify_carries_decision_and_defaults_qa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The route classifies the reply mode and grades effort; unparsed output falls back to QA."""
    cog = _cog()
    _recorded(cog).responses.output_parsed = RouteClassification(
        decision="IMAGE", link_context_sources=["threads", "bilibili"], effort="low"
    )
    message = FakeMessage(content="draw a cat", author=FakeAuthor(user_id=1))
    routed = await _route(cog=cog, message=message)
    assert routed.decision == "IMAGE"
    assert routed.link_context_sources == ["threads", "bilibili"]
    assert routed.effort == "low"

    warned = capture_logs(monkeypatch=monkeypatch, level="warn")
    _recorded(cog).responses.output_parsed = None
    fallback = await _route(cog=cog, message=message)
    assert fallback.decision == "QA"
    assert fallback.link_context_sources == []
    assert fallback.effort == "high"
    # Nothing raised, so this record is the only trace that the route was never read.
    assert [text for text, _ in warned] == [
        "RouteClassification returned no parsed output; defaulting to QA"
    ]


async def test_route_grades_effort_even_on_what_it_cannot_read() -> None:
    """An attachment or a URL is graded by the model, not settled in code."""
    cog = _cog()
    _recorded(cog).responses.output_parsed = RouteClassification(decision="QA", effort="low")

    with_attachment = FakeMessage(content="how do I fix this", author=FakeAuthor(user_id=1))
    with_attachment.attachments = [FakeAttachment(filename="shot.png", content_type="image/png")]
    assert (await _route(cog=cog, message=with_attachment)).effort == "low"

    with_url = FakeMessage(content="這篇 https://example.test/post", author=FakeAuthor(user_id=1))
    assert (await _route(cog=cog, message=with_url)).effort == "low"

    # Both reached the model: a code-decided "high" for these would grade a sticker-only
    # reaction as if it hid something to read, and buys nothing the prompt does not already
    # deliver on its own.
    assert len(_recorded(cog).responses.parse_models) == 2


_ROUTE_SERVER_MEMORY = render_server_memory_block(
    memory="## 成員稱呼\n* Boss(社群暱稱:李董)[id: 42]"
)


async def test_the_route_reads_neither_memory_block_without_candidates() -> None:
    """With nothing to pick from, the route asks the plain question over the messages alone.

    The server memory is handed in on every turn that has one, and must still stay out: only a
    candidate block gives the route a reason to read it.
    """
    cog = _cog()
    # Staged with picks, which the plain schema has no field for.
    _recorded(cog).responses.output_parsed = RecallRouteClassification(
        decision="QA", effort="low", recall_user_ids=["42"]
    )

    route = await _classifier(cog=cog, message=as_message(fake=FakeMessage())).classify(
        reference_messages=_ROUTE_REFERENCE,
        current_message=_ROUTE_CURRENT,
        recall_candidates={},
        server_memory_block=_ROUTE_SERVER_MEMORY,
    )

    responses = _recorded(cog).responses
    assert responses.parse_text_formats == [RouteClassification]
    assert responses.parse_instructions == [route_prompt(inline_image_enabled=True)]
    assert responses.parse_inputs == [[*_ROUTE_REFERENCE, *_ROUTE_CURRENT]]
    assert type(route) is RouteClassification
    assert route.effort == "low"


async def test_the_route_reads_the_server_memory_first_and_the_candidates_last() -> None:
    """Offered candidates, the route reads the table first and decides against the block last."""
    cog = _cog()
    _recorded(cog).responses.output_parsed = RecallRouteClassification(
        decision="QA", effort="low", recall_user_ids=["42"]
    )
    candidates = {
        42: RecallCandidate(prompt_label="Boss(社群暱稱:李董)"),
        43: RecallCandidate(prompt_label="Bob(社群暱稱:阿伯)"),
    }
    classifier = _classifier(cog=cog, message=as_message(fake=FakeMessage()))

    route = await classifier.classify(
        reference_messages=_ROUTE_REFERENCE,
        current_message=_ROUTE_CURRENT,
        recall_candidates=candidates,
        server_memory_block=_ROUTE_SERVER_MEMORY,
    )
    # A guild with a candidate but no server memory of its own has no block to lead with.
    await classifier.classify(
        reference_messages=_ROUTE_REFERENCE,
        current_message=_ROUTE_CURRENT,
        recall_candidates=candidates,
        server_memory_block=None,
    )

    responses = _recorded(cog).responses
    assert responses.parse_text_formats == [RecallRouteClassification] * 2
    assert (
        responses.parse_instructions
        == [route_prompt(inline_image_enabled=True) + ROUTE_RECALL_SECTION] * 2
    )
    candidate_block = render_callable_users_block(allowed=candidates)
    assert responses.parse_inputs == [
        [_ROUTE_SERVER_MEMORY, *_ROUTE_REFERENCE, *_ROUTE_CURRENT, candidate_block],
        [*_ROUTE_REFERENCE, *_ROUTE_CURRENT, candidate_block],
    ]
    assert extract_callable_user_ids(request=[candidate_block]) == {42, 43}
    assert isinstance(route, RecallRouteClassification)
    assert route.recall_user_ids == ["42"]
    assert route.effort == "low"


async def test_an_unparseable_route_falls_back_to_a_plain_high_effort_qa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A triage answer that fails its schema loses the picks along with everything else.

    The fallback is the plain class, so the pipeline resolves no picks from it, and `high`,
    the grade the route gives whatever it is unsure about.
    """
    cog = _cog()
    with pytest.raises(ValidationError) as invalid:
        RecallRouteClassification.model_validate(obj={"decision": "SING"})

    async def reject(**kwargs: object) -> object:
        """Fails the way `responses.parse` does on a reply outside the schema."""
        del kwargs
        raise invalid.value

    monkeypatch.setattr(_recorded(cog).responses, "parse", reject)

    route = await _classifier(cog=cog, message=as_message(fake=FakeMessage())).classify(
        reference_messages=_ROUTE_REFERENCE,
        current_message=_ROUTE_CURRENT,
        recall_candidates={42: RecallCandidate(prompt_label="Boss(社群暱稱:李董)")},
        server_memory_block=_ROUTE_SERVER_MEMORY,
    )

    assert type(route) is RouteClassification
    assert route == RouteClassification(decision="QA", effort="high")


async def test_a_transient_route_failure_falls_back_but_a_refusal_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider outage on the triage call costs the route, not the reply.

    The same plain high-effort QA as an answer outside the schema. A refusal keeps raising, so a
    defect in the request is not buried under a turn that looks routed.
    """
    cog = _cog()
    request = httpx2.Request(method="POST", url="http://proxy/v1/responses")
    failures: list[Exception] = [
        APIError(message="high demand", request=request, body={"code": "503"}),
        APITimeoutError(request=request),
        BadRequestError(
            "no", response=httpx2.Response(status_code=400, request=request, json={}), body=None
        ),
    ]

    async def fail(**kwargs: object) -> object:
        """Fails the way `responses.parse` does when the provider does."""
        del kwargs
        raise failures.pop(0)

    monkeypatch.setattr(_recorded(cog).responses, "parse", fail)
    classifier = _classifier(cog=cog, message=as_message(fake=FakeMessage()))

    async def classify() -> RouteClassification:
        """One route call on the candidate shape, which the fallback must also drop."""
        return await classifier.classify(
            reference_messages=_ROUTE_REFERENCE,
            current_message=_ROUTE_CURRENT,
            recall_candidates={42: RecallCandidate(prompt_label="Boss(社群暱稱:李董)")},
            server_memory_block=_ROUTE_SERVER_MEMORY,
        )

    for _ in range(2):
        route = await classify()
        assert type(route) is RouteClassification
        assert route == RouteClassification(decision="QA", effort="high")
    with pytest.raises(BadRequestError):
        await classify()


async def test_route_input_excludes_attachment_payloads() -> None:
    """The route request sees an attachment marker instead of the file payload."""
    cog = _cog()
    message = FakeMessage(content="<@999> see", author=FakeAuthor(user_id=1))
    message.attachments = [FakeAttachment(filename="note.txt", content_type="text/plain")]

    await _route(cog=cog, message=message)

    rendered = str(_recorded(cog).responses.parse_inputs[-1])
    assert "input_file" not in rendered
    assert "[attachment: file]" in rendered
