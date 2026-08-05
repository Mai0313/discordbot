"""Turning a report into issue text, and the background write-up that improves it.

The issue is opened from the reporter's raw words first and rewritten afterwards. That
ordering is what makes the write-up genuinely best-effort: there is no fallback text to
invent when the model fails, because the issue is already sitting there in the words the
reporter used.

The reporter's own wording is always attached verbatim, fenced, no matter which version
of the body is above it. Fencing is not cosmetic: inside a code fence GitHub renders
`@name` and `#123` as text instead of notifying a stranger or linking someone else's
issue, and the reporter controls every character in there.
"""

import re
from typing import Final, Literal

from openai import AsyncOpenAI
from pydantic import Field, BaseModel

from discordbot.utils.llm import parse_responses_or_none
from discordbot.typings.models import ModelSettings
from discordbot.cogs.feedback.prompts import REPORT_WRITE_UP_PROMPT
from discordbot.cogs.feedback.database import FeedbackTicket

# No deadline of our own: by the time this runs the reporter already has their issue
# number and nobody is waiting. This is the `AsyncOpenAI` client's own ceiling, restated
# so a connection that hangs forever cannot leave the report without a title forever.
WRITE_UP_TIMEOUT_SECONDS: Final[float] = 600.0

# GitHub caps an issue body at 65536 characters; the drafts stay well under it so the
# attached original always fits underneath.
_MAX_BODY_CHARS: Final[int] = 30000
_MAX_TITLE_CHARS: Final[int] = 120

_BACKTICK_RUN = re.compile(r"`+")


class ReportWriteUp(BaseModel):
    """The structured write-up of one report."""

    label: str = Field(
        ..., description="One short Traditional Chinese line naming the problem for the reporter."
    )
    category: Literal["bug", "feature", "question"] = Field(
        ..., description="What kind of report this is."
    )
    title: str = Field(..., description="English issue title in conventional-commit style.")
    body: str = Field(..., description="English issue body in GitHub-flavoured markdown.")


def label_for_category(*, category: str) -> list[str]:
    """The GitHub labels an issue of this category carries.

    `user-report` is what makes these findable as a group later, and it is the handle
    that would rebuild the local store if it were ever lost.
    """
    kind = {"bug": "bug", "feature": "feature", "question": "question"}.get(category, "")
    return ["user-report", kind] if kind else ["user-report"]


def initial_issue_title(*, ticket: FeedbackTicket) -> str:
    """The title an issue is opened with, before any write-up exists.

    The reporter's own first line, because it is the only text that exists at that point
    and it is already the most accurate summary available.
    """
    first_line = next((line.strip() for line in ticket.raw_text.splitlines() if line.strip()), "")
    title = first_line or f"user report #{ticket.ticket_id}"
    return title[:_MAX_TITLE_CHARS]


def _fenced(*, text: str) -> str:
    """Wraps reporter-authored text in a fence longer than any backtick run inside it."""
    longest = max((len(match.group()) for match in _BACKTICK_RUN.finditer(text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def _reporter_line(*, ticket: FeedbackTicket) -> str:
    """The line naming who filed the report and from where."""
    origin = f"in {ticket.guild_name}" if ticket.guild_name else "in a direct message"
    filed = ticket.created_at.strftime("%Y-%m-%d")
    return (
        f"Reported through Discord by **{ticket.display_name}** "
        f"(`{ticket.user_name}`, id `{ticket.user_id}`) {origin} on {filed}."
    )


def render_issue_body(*, ticket: FeedbackTicket, write_up: ReportWriteUp | None) -> str:
    """Builds the issue body: the write-up when there is one, then the original wording."""
    lead = write_up.body.strip()[:_MAX_BODY_CHARS] if write_up is not None else ""
    original = _fenced(text=ticket.raw_text.strip() or "(empty)")
    sections = [
        lead,
        "---",
        _reporter_line(ticket=ticket),
        f"<details>\n<summary>Original wording</summary>\n\n{original}\n\n</details>",
    ]
    return "\n\n".join(section for section in sections if section)


async def write_up_report(
    *, client: AsyncOpenAI, model: ModelSettings, ticket: FeedbackTicket
) -> ReportWriteUp | None:
    """Runs the background write-up, returning None when it does not work out.

    None is an ordinary outcome, not an error: the issue keeps the reporter's own words
    and the panel keeps showing their first line, which is what it showed all along.
    """
    return await parse_responses_or_none(
        client=client,
        model=model,
        instructions=REPORT_WRITE_UP_PROMPT,
        user_text=f"<report>\n{ticket.raw_text}\n</report>",
        end_user_id=str(ticket.user_id),
        text_format=ReportWriteUp,
        timeout_seconds=WRITE_UP_TIMEOUT_SECONDS,
    )
