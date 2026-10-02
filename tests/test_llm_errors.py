"""Tests for reading a provider's message and status out of an LLM failure."""

import json

# openai 3.x builds its exceptions on httpx2, so a request or response handed to one has to
# come from there. google-genai is still on httpx 0.x, and both live in the environment.
import httpx2
from openai import APIError, APITimeoutError, BadRequestError, APIConnectionError
from google.genai.errors import ClientError

from discordbot.utils.llm_errors import (
    llm_status_code,
    extract_friendly_error,
    is_retryable_llm_error,
)

from tests.helpers.casting import make_invalid_form_body


def test_extract_friendly_error_prefers_nested_provider_message() -> None:
    """Verifies nested provider errors are preferred over wrapper text."""
    raw = """wrapper b'{"error": {"message": "quota exceeded"}}'"""
    assert extract_friendly_error(exc=RuntimeError(raw)) == "quota exceeded"
    assert extract_friendly_error(exc=RuntimeError("plain failure")) == "plain failure"
    assert extract_friendly_error(exc=RuntimeError("bad b'not json'")) == "bad b'not json'"


def test_extract_friendly_error_reads_a_decoded_400_body() -> None:
    """A plain provider 400 is read off the exception, not out of its dict-repr string."""
    refusal = "Input blocked: Sorry, we can't create videos with real people's names."
    body = {"error": {"message": refusal, "code": "invalid_request"}}

    # `_make_status_error` unwraps the `error` object into `.body` before raising, and renders
    # the whole document into the message as a Python dict repr.
    request = httpx2.Request(method="POST", url="http://proxy/v1/images/generations")
    response = httpx2.Response(status_code=400, request=request, json=body)
    proxied = BadRequestError(f"Error code: 400 - {body}", response=response, body=body["error"])
    assert extract_friendly_error(exc=proxied) == refusal

    # The direct-to-Google path keeps the whole document on `.details` instead.
    assert extract_friendly_error(exc=ClientError(400, body, None)) == refusal

    # A non-streaming LiteLLM 400 needs both steps: the dict repr escapes the wrapped chain's
    # quotes to `b\'...\'`, so the bytes literal is only reachable once `.body` has replaced the
    # text being scanned.
    chain = """litellm.BadRequestError: VertexAIException - b'{"error": {"message": "quota"}}'"""
    wrapped_body = {"error": {"message": chain, "code": "400"}}
    wrapped = BadRequestError(
        f"Error code: 400 - {wrapped_body}", response=response, body=wrapped_body["error"]
    )
    assert extract_friendly_error(exc=wrapped) == "quota"


def test_extract_friendly_error_peels_a_flattened_litellm_wrapper_chain() -> None:
    """The provider's own sentence reaches the embed whichever way LiteLLM rendered it.

    Which of the two shapes arrives records WHEN the request broke, not what broke, so the same
    provider message shows up in both and neither may bring the wrapper chain along with it.
    """
    request = httpx2.Request(method="POST", url="http://proxy/v1/responses")
    high_demand = (
        "This model is currently experiencing high demand. Spikes in demand are usually "
        "temporary. Please try again later."
    )
    chain = "litellm.MidStreamFallbackError: litellm.ServiceUnavailableError: Vertex_ai_beta"

    def _mid_stream(message: str) -> APIError:
        frame = {"message": message, "type": "None", "param": "None", "code": "503"}
        return APIError(message=message, request=request, body=frame)

    # The stream never opened, so the handler read the body as bytes and `str()` embedded their
    # repr: the provider document rides along unparsed.
    document = {"error": {"code": 503, "message": high_demand, "status": "UNAVAILABLE"}}
    unopened = f"{chain}Exception - {json.dumps(obj=document).encode()!r}\n"
    assert extract_friendly_error(exc=_mid_stream(message=unopened)) == high_demand

    # The stream opened and a chunk carried the error object, so LiteLLM parsed it and re-rendered
    # it as `<canonical status> - <message>`, leaving no bytes literal to find.
    flattened = f"{chain}Exception - UNAVAILABLE - {high_demand}"
    assert extract_friendly_error(exc=_mid_stream(message=flattened)) == high_demand

    # The shapes are not per-message: a second message came through the same path in the same
    # window, and its status is a different canonical one.
    deadline = "Deadline expired before operation could complete."
    expired = f"{chain}Exception - DEADLINE_EXCEEDED - {deadline}"
    assert extract_friendly_error(exc=_mid_stream(message=expired)) == deadline

    # A segment is a run of words rather than one token: the vertex mapping puts a second label
    # inside the marker on some branches, and the openai one puts one in front of it.
    blocked = "The response was blocked by the safety filter."
    labelled = f"litellm.BadRequestError: Vertex_ai_betaException BadRequestError - {blocked}"
    assert extract_friendly_error(exc=_mid_stream(message=labelled)) == blocked

    # The run is anchored, so a class named after neither `Error` nor `Exception` would cost not
    # its own segment but every segment behind it. `Timeout` is the only one.
    timed_out = f"litellm.Timeout: Timeout Error: VertexAIException - {deadline}"
    assert extract_friendly_error(exc=_mid_stream(message=timed_out)) == deadline

    # Peeling that leaves nothing behind is discarded, since a chain still says more than an
    # empty embed would.
    bare = "litellm.APIConnectionError: Vertex_ai_betaException - "
    assert extract_friendly_error(exc=_mid_stream(message=bare)) == bare


def test_is_retryable_llm_error_reads_the_status_out_of_every_wrapper_shape() -> None:
    """A transient upstream failure is retried; a refusal and an unreadable one are not."""
    request = httpx2.Request(method="POST", url="http://proxy/v1/responses")

    # The shape this exists for. LiteLLM reports a mid-stream provider failure as an SSE error
    # frame holding `ProxyException.to_dict()`, whose `code` is a decimal STRING, and openai's
    # streaming layer re-raises it as a bare APIError carrying that frame as its body. Nothing
    # here is typed, so a check reading `.status_code` alone sees no status at all.
    frame = {
        "message": "litellm.MidStreamFallbackError: litellm.ServiceUnavailableError: ...",
        "type": "None",
        "param": "None",
        "code": "503",
    }
    mid_stream = APIError(message=str(frame["message"]), request=request, body=frame)
    assert llm_status_code(exc=mid_stream) == 503
    assert is_retryable_llm_error(exc=mid_stream) is True

    # Same wrapper, a refusal underneath: re-sending it only makes the user wait for the same
    # answer three times.
    refusal = APIError(message="blocked", request=request, body={**frame, "code": "400"})
    assert is_retryable_llm_error(exc=refusal) is False

    # A status the SDK typed wins over the body, and an unreadable failure is not retried:
    # a status that cannot be read is as likely to be a refusal as an outage.
    response = httpx2.Response(status_code=400, request=request, json={})
    assert is_retryable_llm_error(exc=BadRequestError("no", response=response, body=None)) is False
    assert is_retryable_llm_error(exc=APIError(message="?", request=request, body=None)) is False
    assert is_retryable_llm_error(exc=RuntimeError("boom")) is False

    # A Discord write failure escaping the streamer must never re-run the answer. It carries a
    # plain int `code` of its own -- 50035 is what an oversized final write raises -- which the
    # status read would otherwise take for a 5xx.
    assert llm_status_code(exc=make_invalid_form_body()) == 50035
    assert is_retryable_llm_error(exc=make_invalid_form_body()) is False

    # Transport failures carry no status of any kind; `APITimeoutError` rides in as a subclass.
    assert is_retryable_llm_error(exc=APIConnectionError(request=request)) is True
    assert is_retryable_llm_error(exc=APITimeoutError(request=request)) is True

    # A `google.genai.errors` failure keeps the status as an int on `.code`.
    assert llm_status_code(exc=ClientError(429, {"error": {"message": "slow down"}}, None)) == 429
    assert (
        is_retryable_llm_error(exc=ClientError(429, {"error": {"message": "slow"}}, None)) is True
    )
    assert (
        is_retryable_llm_error(exc=ClientError(400, {"error": {"message": "bad"}}, None)) is False
    )
