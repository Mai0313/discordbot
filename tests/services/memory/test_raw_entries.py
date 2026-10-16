"""Tests for rendering observations into raw entries."""

from discordbot.services.memory.raw_entries import render_memory_observations

from tests.helpers.user_memory import _observation


def test_render_memory_observations_stamps_source_and_sharing() -> None:
    rendered = render_memory_observations(
        observations=(_observation(summary="喜歡簡短", sharing="source_only"),), source="guild 123"
    )
    lines = rendered.splitlines()
    assert "- source: guild 123" in lines
    assert "- sharing: source_only" in lines
    # The code-stamped fields sit between ttl_days and the observation text.
    assert lines.index("- ttl_days: null") < lines.index("- source: guild 123")
    assert lines.index("- source: guild 123") < lines.index("- sharing: source_only")
    assert lines.index("- sharing: source_only") < lines.index("- summary_zh: 喜歡簡短")


def test_render_memory_observations_without_source_omits_source_and_sharing() -> None:
    # The server flavor renders neither field.
    rendered = render_memory_observations(
        observations=(_observation(summary="喜歡簡短"),), source=None
    )
    assert "- source:" not in rendered
    assert "- sharing:" not in rendered
