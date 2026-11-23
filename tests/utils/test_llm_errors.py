"""Tests for reading a provider's message and status out of an LLM failure."""

import json

import httpx2
from openai import (
    APIError,
    NotFoundError,
    APIStatusError,
    RateLimitError,
    APITimeoutError,
    BadRequestError,
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    PermissionDeniedError,
    UnprocessableEntityError,
)
from google.genai.errors import APIError as GenAIAPIError
from google.genai.errors import ClientError, ServerError
from google.genai._gaos.lib.compat_errors import APIError as InteractionsAPIError
from google.genai._gaos.lib.compat_errors import APITimeoutError as InteractionsTimeoutError
from google.genai._gaos.lib.compat_errors import APIConnectionError as InteractionsConnectionError

from discordbot.utils.llm_errors import (
    HIGH_DEMAND_ERROR_NOTICE,
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

    Which of the two shapes arrives records WHEN the request broke, not what broke, so non-masked
    statuses surface their unwrapped sentence while 503 overload collapses to the generic notice.
    """
    request = httpx2.Request(method="POST", url="http://proxy/v1/responses")
    high_demand = HIGH_DEMAND_ERROR_NOTICE
    chain = "litellm.MidStreamFallbackError: litellm.ServiceUnavailableError: Vertex_ai_beta"

    def _mid_stream(message: str, code: str = "503") -> APIError:
        frame = {"message": message, "type": "None", "param": "None", "code": code}
        return APIError(message=message, request=request, body=frame)

    # The stream never opened, so the handler read the body as bytes and `str()` embedded their
    # repr: the provider document rides along unparsed.
    document = {"error": {"code": 503, "message": high_demand, "status": "UNAVAILABLE"}}
    unopened = f"{chain}Exception - {json.dumps(obj=document).encode()!r}\n"
    assert extract_friendly_error(exc=_mid_stream(message=unopened)) == HIGH_DEMAND_ERROR_NOTICE

    # The stream opened and a chunk carried the error object, so LiteLLM parsed it and re-rendered
    # it as `<canonical status> - <message>`, leaving no bytes literal to find.
    flattened = f"{chain}Exception - UNAVAILABLE - {high_demand}"
    assert extract_friendly_error(exc=_mid_stream(message=flattened)) == HIGH_DEMAND_ERROR_NOTICE

    # The shapes are not per-message: a second message came through the same path in the same
    # window, and its status is a different canonical one.
    deadline = "Deadline expired before operation could complete."
    expired = f"{chain}Exception - DEADLINE_EXCEEDED - {deadline}"
    assert extract_friendly_error(exc=_mid_stream(message=expired, code="504")) == (
        HIGH_DEMAND_ERROR_NOTICE
    )

    # A segment is a run of words rather than one token: the vertex mapping puts a second label
    # inside the marker on some branches, and the openai one puts one in front of it.
    blocked = "The response was blocked by the safety filter."
    labelled = f"litellm.BadRequestError: Vertex_ai_betaException BadRequestError - {blocked}"
    assert extract_friendly_error(exc=_mid_stream(message=labelled, code="400")) == blocked

    # The run is anchored, so a class named after neither `Error` nor `Exception` would cost not
    # its own segment but every segment behind it. `Timeout` is the only one.
    timed_out = f"litellm.Timeout: Timeout Error: VertexAIException - {deadline}"
    assert extract_friendly_error(exc=_mid_stream(message=timed_out, code="408")) == deadline

    # Peeling that leaves nothing behind is discarded, since a chain still says more than an
    # empty embed would.
    bare = "litellm.APIConnectionError: Vertex_ai_betaException - "
    assert extract_friendly_error(exc=_mid_stream(message=bare, code="None")) == bare


def test_extract_friendly_error_masks_overload_and_quota_errors() -> None:
    """Operator/backend failures (`401`/`402`/`403`/`404`/`429`/`5xx`, non-timeout connect) collapse to the notice."""
    request = httpx2.Request(method="POST", url="http://proxy/v1/responses")
    quota_msg = (
        "You exceeded your current quota, please check your plan and billing details. "
        "For more information on this error, head to: https://ai.dev/rate-limit."
    )

    def _status_err(cls: type[APIStatusError], status: int, msg: str) -> APIStatusError:
        return cls(
            message=f"Error code: {status} - {msg}",
            response=httpx2.Response(status_code=status, request=request, json={}),
            body={"error": {"message": msg, "code": status}},
        )

    for masked_exc in (
        _status_err(cls=AuthenticationError, status=401, msg="Invalid API key"),
        _status_err(cls=APIStatusError, status=402, msg=quota_msg),
        _status_err(cls=PermissionDeniedError, status=403, msg="Permission denied"),
        _status_err(cls=NotFoundError, status=404, msg="Model not found"),
        _status_err(cls=RateLimitError, status=429, msg=quota_msg),
        _status_err(cls=InternalServerError, status=500, msg="Internal error"),
        _status_err(cls=APIStatusError, status=502, msg="Bad gateway"),
        _status_err(cls=InternalServerError, status=503, msg="upstream overloaded"),
        _status_err(cls=APIStatusError, status=504, msg="Gateway timeout"),
        APIConnectionError(request=request),
        InteractionsConnectionError(message="Connection error.", request=request),
    ):
        assert extract_friendly_error(exc=masked_exc) == HIGH_DEMAND_ERROR_NOTICE

    # google.genai.errors (APIError, ClientError, ServerError)
    genai_429 = GenAIAPIError(429, {"error": {"code": 429, "message": quota_msg}}, None)
    genai_503 = GenAIAPIError(503, {"error": {"code": 503, "message": "busy"}}, None)
    assert extract_friendly_error(exc=genai_429) == HIGH_DEMAND_ERROR_NOTICE
    assert extract_friendly_error(exc=genai_503) == HIGH_DEMAND_ERROR_NOTICE
    assert (
        extract_friendly_error(exc=ClientError(429, {"error": {"message": quota_msg}}, None))
        == HIGH_DEMAND_ERROR_NOTICE
    )
    assert (
        extract_friendly_error(exc=ServerError(503, {"error": {"message": "busy"}}, None))
        == HIGH_DEMAND_ERROR_NOTICE
    )

    # Interactions APIError preserves 400 and masks 429/503.
    ix_429 = InteractionsAPIError.generate(
        status_code=429,
        body={"error": {"message": quota_msg}},
        message=quota_msg,
        response=httpx2.Response(status_code=429, request=request, json={}),
    )
    ix_503 = InteractionsAPIError.generate(
        status_code=503,
        body={"error": {"message": "busy"}},
        message="busy",
        response=httpx2.Response(status_code=503, request=request, json={}),
    )
    ix_400 = InteractionsAPIError.generate(
        status_code=400,
        body={"error": {"message": "bad input"}},
        message="bad input",
        response=httpx2.Response(status_code=400, request=request, json={}),
    )
    assert extract_friendly_error(exc=ix_429) == HIGH_DEMAND_ERROR_NOTICE
    assert extract_friendly_error(exc=ix_503) == HIGH_DEMAND_ERROR_NOTICE
    assert extract_friendly_error(exc=ix_400) == "bad input"

    # Outer 400 carrying masked string error codes in body, or RuntimeError wrapping embedded JSON.
    for code in (
        "authentication_error",
        "permission_error",
        "not_found_error",
        "rate_limit_exceeded",
        "insufficient_quota",
        "server_error",
    ):
        bad_with_code = BadRequestError(
            "Error code: 400",
            response=httpx2.Response(status_code=400, request=request, json={}),
            body={"message": quota_msg, "code": code},
        )
        assert extract_friendly_error(exc=bad_with_code) == HIGH_DEMAND_ERROR_NOTICE

    wrapped_429 = RuntimeError(
        f"litellm.RateLimitError: {json.dumps(obj={'error': {'code': 429, 'message': quota_msg}}).encode()!r}"
    )
    assert extract_friendly_error(exc=wrapped_429) == HIGH_DEMAND_ERROR_NOTICE
    wrapped_503_no_msg = RuntimeError(
        f"wrapper {json.dumps(obj={'error': {'code': 503, 'status': 'UNAVAILABLE'}}).encode()!r}"
    )
    assert extract_friendly_error(exc=wrapped_503_no_msg) == HIGH_DEMAND_ERROR_NOTICE
    wrapped_500 = RuntimeError(
        f"litellm.APIError: {json.dumps(obj={'error': {'code': 500, 'message': 'Internal model fault'}}).encode()!r}"
    )
    assert extract_friendly_error(exc=wrapped_500) == HIGH_DEMAND_ERROR_NOTICE

    # User-actionable 4xx (400/408/413/422), real timeouts, and plain non-masked JSON stay readable.
    assert (
        extract_friendly_error(exc=_status_err(cls=BadRequestError, status=400, msg="Bad input"))
        == "Bad input"
    )
    assert (
        extract_friendly_error(exc=_status_err(cls=APIStatusError, status=408, msg="Timed out"))
        == "Timed out"
    )
    assert (
        extract_friendly_error(exc=_status_err(cls=APIStatusError, status=413, msg="Too large"))
        == "Too large"
    )
    assert (
        extract_friendly_error(
            exc=_status_err(cls=UnprocessableEntityError, status=422, msg="Unprocessable")
        )
        == "Unprocessable"
    )
    wrapped_400 = RuntimeError(
        f"litellm.BadRequestError: {json.dumps(obj={'error': {'code': 400, 'message': 'Invalid argument'}}).encode()!r}"
    )
    assert extract_friendly_error(exc=wrapped_400) == "Invalid argument"
    assert extract_friendly_error(exc=TimeoutError("request timed out")) == "request timed out"
    assert extract_friendly_error(exc=APITimeoutError(request=request)) == "Request timed out."
    ix_timeout = InteractionsTimeoutError(request=request)
    assert extract_friendly_error(exc=ix_timeout) == str(ix_timeout)


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

    # Unicode superscripts pass `.isdigit()` but not `int()`, so they read as no status.
    superscript = APIError(message="odd", request=request, body={**frame, "code": "²"})
    assert llm_status_code(exc=superscript) is None
    assert is_retryable_llm_error(exc=superscript) is False

    # A status the SDK typed wins over the body, and an unreadable failure is not retried:
    # a status that cannot be read is as likely to be a refusal as an outage.
    response = httpx2.Response(status_code=400, request=request, json={})
    assert is_retryable_llm_error(exc=BadRequestError("no", response=response, body=None)) is False
    assert is_retryable_llm_error(exc=APIError(message="?", request=request, body=None)) is False
    assert is_retryable_llm_error(exc=RuntimeError("boom")) is False

    # A Discord write failure escaping the streamer must never re-run the answer. It carries a
    # plain int `code` of its own -- 50035 is what an oversized write raises -- which the
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

    # Nested `body["error"]["code"]` (int or ASCII decimal str) is read cleanly; bool/non-ASCII ignored.
    nested_429 = APIError(message="slow", request=request, body={"error": {"code": 429}})
    nested_503 = APIError(message="busy", request=request, body={"error": {"code": "503"}})
    non_ascii = APIError(message="odd", request=request, body={"error": {"code": "٢"}})
    bool_code = APIError(message="odd", request=request, body={"error": {"code": True}})
    assert llm_status_code(exc=nested_429) == 429
    assert is_retryable_llm_error(exc=nested_429) is True
    assert llm_status_code(exc=nested_503) == 503
    assert is_retryable_llm_error(exc=nested_503) is True
    assert llm_status_code(exc=non_ascii) is None
    assert llm_status_code(exc=bool_code) is None
