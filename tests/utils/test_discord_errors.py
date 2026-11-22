"""Tests for Discord error classifiers and structured failure reporting."""

import pytest
from nextcord import HTTPException

from discordbot.utils.discord_errors import is_reply_target_gone, report_discord_failure

from tests.helpers.casting import (
    make_forbidden,
    make_not_found,
    make_server_error,
    make_invalid_form_body,
)
from tests.helpers.logfire_capture import capture_levels


@pytest.mark.parametrize(
    argnames=("error", "expected"),
    argvalues=[
        (make_not_found(code=10008), True),
        (make_invalid_form_body(), True),
        (make_forbidden(code=50001), False),
        (make_server_error(), False),
    ],
    ids=["not_found", "code_50035", "forbidden", "server_error"],
)
def test_is_reply_target_gone_matches_not_found_and_50035(
    error: HTTPException, expected: bool
) -> None:
    """Only `NotFound` and `50035` count as a possibly deleted reply target."""
    assert is_reply_target_gone(error=error) is expected


def test_report_discord_failure_logs_forbidden_at_warn_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Forbidden` logs at `warn` with `code`, `error_type`, and subject fields, no traceback."""
    reports = capture_levels(monkeypatch=monkeypatch, levels=("debug", "info", "warn", "error"))
    error = make_forbidden(message="Missing Access", code=50001)

    report_discord_failure(
        error=error,
        not_found_message="gone",
        forbidden_message="refused",
        other_message="failed",
        channel_id=10,
        message_id=20,
    )

    assert reports == [
        (
            "warn",
            "refused",
            {"channel_id": 10, "message_id": 20, "code": 50001, "error_type": "Forbidden"},
        )
    ]


def test_report_discord_failure_handles_not_found_defaults_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`NotFound` defaults to `info` without traceback, can be silenced, or kept as a trace."""
    reports = capture_levels(monkeypatch=monkeypatch, levels=("debug", "info", "warn", "error"))
    error = make_not_found(message="Unknown Message", code=10008)

    report_discord_failure(
        error=error,
        not_found_message="gone",
        forbidden_message="refused",
        other_message="failed",
        message_id=20,
    )
    report_discord_failure(
        error=error,
        not_found_message=None,
        forbidden_message="refused",
        other_message="failed",
        message_id=20,
    )
    report_discord_failure(
        error=error,
        not_found_message="debug 404",
        not_found_level="debug",
        forbidden_message="refused",
        other_message="failed",
        message_id=20,
    )
    report_discord_failure(
        error=error,
        not_found_message="unexpected 404",
        not_found_level="warn",
        not_found_exc_info=True,
        forbidden_message="refused",
        other_message="failed",
        message_id=20,
    )

    assert reports == [
        ("info", "gone", {"message_id": 20, "code": 10008, "error_type": "NotFound"}),
        ("debug", "debug 404", {"message_id": 20, "code": 10008, "error_type": "NotFound"}),
        (
            "warn",
            "unexpected 404",
            {"message_id": 20, "code": 10008, "error_type": "NotFound", "_exc_info": error},
        ),
    ]


def test_report_discord_failure_keeps_traceback_on_5xx_and_non_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """5xx errors carry `code` + traceback; non-HTTP errors carry `error_type` + traceback."""
    reports = capture_levels(monkeypatch=monkeypatch, levels=("debug", "info", "warn", "error"))
    server_error = make_server_error(message="upstream", code=50000)
    timeout_error = TimeoutError("edit hung")

    report_discord_failure(
        error=server_error,
        not_found_message="gone",
        forbidden_message="refused",
        other_message="http failed",
        channel_id=10,
    )
    report_discord_failure(
        error=timeout_error,
        not_found_message="gone",
        forbidden_message="refused",
        other_message="timed out",
        other_level="error",
        channel_id=10,
    )

    assert reports == [
        (
            "warn",
            "http failed",
            {
                "channel_id": 10,
                "code": 50000,
                "error_type": "HTTPException",
                "_exc_info": server_error,
            },
        ),
        (
            "error",
            "timed out",
            {"channel_id": 10, "error_type": "TimeoutError", "_exc_info": timeout_error},
        ),
    ]
