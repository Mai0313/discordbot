"""The direct message that tells a reporter their report was closed.

Its own module rather than part of `views.py`, which is the `/feedback` panel: everything
there is ephemeral and operated by whoever opened it, while this is an unsolicited message
arriving in a DM days later. The two share nothing but the embed type.

The maintainer answers on the issue, in English, because that is where the work happens.
Nobody who filed a report through Discord is obliged to read English, so the closing
comment is translated before it goes out. That call is best-effort in an unusual way: a
failure sends nothing at all, rather than sending the English. The report is not marked as
finished with, so the next sweep tries the whole thing again, which costs one more model
call and gets the reporter a message they can read instead of one they cannot.
"""

from datetime import UTC, datetime, timedelta

from openai import AsyncOpenAI
from nextcord import Embed
from pydantic import Field, BaseModel

from discordbot.utils.llm import parse_responses_or_none
from discordbot.typings.colors import NEUTRAL_GREY, DISCORD_GREEN
from discordbot.typings.models import ModelSettings
from discordbot.cogs.feedback.github import CloseOutcome, IssueComment
from discordbot.cogs.feedback.prompts import CLOSE_NOTICE_TRANSLATION_PROMPT
from discordbot.cogs.feedback.database import FeedbackTicket
from discordbot.typings.context_budgets import CLOSE_NOTICE_LANGUAGE_SAMPLE_CHARS

# Discord allows 1024 per field value and 4096 per description. Both caps here sit under
# those with room, and a cut is marked rather than silent: a maintainer's answer that just
# stops reads as the bot having lost the rest of it.
_MAX_COMMENT_CHARS = 900
_MAX_QUOTED_CHARS = 300
_TRUNCATED_SUFFIX = "…"

# How long before a close a comment can have been written and still be read as the reason
# for it. A day is generous on purpose: "fixed, shipping with the next release" is written
# when the work lands and the issue is closed whenever the maintainer next tidies up. What
# it excludes is the far more damaging case, an unanswered question from weeks earlier
# being handed to the reporter as the verdict on their report.
_CLOSING_WINDOW = timedelta(days=1)

# How long after a close it is still worth telling anyone. This exists for one moment: the
# first sweep after this feature ships, which finds every report ever closed sitting there
# with no notice row and would otherwise mail all of them at once, announcing outcomes from
# months ago as news. It keeps working afterwards for a bot that was down for a fortnight.
BACKFILL_CUTOFF = timedelta(days=7)

_TITLES: dict[CloseOutcome, str] = {"completed": "✅ 已完成", "not_planned": "⚪ 不列入計劃"}
_COLORS: dict[CloseOutcome, int] = {"completed": DISCORD_GREEN, "not_planned": NEUTRAL_GREY}
_NO_COMMENT: dict[CloseOutcome, str] = {
    "completed": "這張單標成已完成，開發者沒有另外留話。",
    "not_planned": "這張單標成不列入計劃，開發者沒有另外留話。",
}


class TranslatedComment(BaseModel):
    """The maintainer's closing words in the language the reporter writes in."""

    text: str = Field(
        ..., description="The maintainer's message, translated, with nothing added or removed."
    )


def _clipped(*, text: str, limit: int) -> str:
    """Returns `text` within `limit`, marked when something was cut off."""
    if len(text) <= limit:
        return text
    return text[: limit - len(_TRUNCATED_SUFFIX)].rstrip() + _TRUNCATED_SUFFIX


def _moment(*, stamp: str | None) -> datetime | None:
    """Parses one GitHub ISO-8601 timestamp into an aware UTC datetime.

    Never raises. A stamp this cannot read costs the notice its comment, not the notice
    itself, and every caller reads None as "cannot tell" rather than as "no comment".

    Always aware, even though GitHub's own stamps all carry `Z`. Everything these are
    compared against is aware, and subtracting a naive one raises `TypeError` inside a
    sweep whose broad `except` would swallow it: the failure mode is not a wrong answer but
    a sweep that logs every ten minutes and silently never delivers anything.
    """
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def closed_too_long_ago(*, closed_at: str | None) -> bool:
    """Whether a close is old enough that announcing it would read as a mistake.

    A close this cannot date is treated as recent. The alternative silently swallows a
    notice on the strength of a timestamp nobody could read, and the far more common reason
    for an unreadable one is a stub or an API change rather than genuine age.
    """
    closed = _moment(stamp=closed_at)
    if closed is None:
        return False
    return datetime.now(tz=UTC) - closed > BACKFILL_CUTOFF


def closing_comment(*, comments: list[IssueComment], closed_at: str | None) -> IssueComment | None:
    """The maintainer's explanation for closing the report, or None when there is none.

    `read_conversation` has already dropped passers-by and bots and marked the lines this
    bot relayed for the reporter, so anything left that is not theirs is the maintainer's.
    But being the maintainer's newest line does not make it the closing explanation, and
    handing over the wrong one is worse than handing over nothing: the reporter reads it as
    the answer to a report that it may predate by weeks.

    Two things have to hold. **Nobody spoke after it**, which is what separates an
    explanation from a question the reporter has since answered through the panel, the
    common shape of a thread that ends in a close with nothing further said. And it was
    **written around the close** rather than at some point in the issue's past, since
    closing and commenting are separate actions on GitHub and the explanation lands on
    either side of the close. `_CLOSING_WINDOW` is how far before it still counts; there is
    no bound on the other side, because the delivery pass is already waiting a fixed time
    and cannot see past it anyway.

    An unreadable or absent `closed_at` keeps the ordering rule and drops the window one.
    Losing the timing evidence is not a reason to also forget who spoke last.
    """
    if not comments or comments[-1].from_reporter:
        return None
    candidate = comments[-1]
    closed = _moment(stamp=closed_at)
    written = _moment(stamp=candidate.created_at)
    if closed is None or written is None:
        return candidate
    return candidate if written >= closed - _CLOSING_WINDOW else None


async def translate_comment(
    *, client: AsyncOpenAI, model: ModelSettings, ticket: FeedbackTicket, comment: IssueComment
) -> str | None:
    """Translates one closing comment into the language the reporter wrote in.

    Returns None when the call fails, and the caller must treat that as "not yet" rather
    than falling back to the English: the whole reason this runs is that the reporter may
    not read it.

    The report's own text is what says which language that is. `locale` rides along, but
    it cannot be trusted alone: it is the Discord client's UI language, read off
    `interaction.locale` when the report was filed, and somebody running an English client
    while writing Chinese is exactly the person this feature exists for. It is also allowed
    to be empty, which would leave nothing to aim at at all.
    """
    body = comment.body.strip()
    sample = ticket.raw_text.strip()[:CLOSE_NOTICE_LANGUAGE_SAMPLE_CHARS]
    translated = await parse_responses_or_none(
        client=client,
        model=model,
        instructions=CLOSE_NOTICE_TRANSLATION_PROMPT,
        user_text=(
            f"<reporter_wording>\n{sample}\n</reporter_wording>\n"
            f"<reporter_client_locale>{ticket.locale or 'unknown'}</reporter_client_locale>\n"
            f"<maintainer_message>\n{body}\n</maintainer_message>"
        ),
        end_user_id=str(ticket.user_id),
        text_format=TranslatedComment,
    )
    if translated is None:
        return None
    return translated.text.strip() or body


def build_close_notice_embed(
    *, ticket: FeedbackTicket, outcome: CloseOutcome, comment: IssueComment | None, text: str
) -> Embed:
    """Builds the message a reporter gets when their report is closed.

    `duplicate` never reaches here: that report is tracked on another issue and its own
    outcome is not settled, so there is nothing to tell anyone yet.

    The footer points at `/feedback` rather than at the issue. The issue is written in
    English and carries the reporter's own Discord name and id, so sending them to it would
    be handing someone a page about themselves that they may not be able to read.
    """
    number = f"#{ticket.issue_number}" if ticket.issue_number else ""
    embed = Embed(title=f"{_TITLES[outcome]} {number}".strip(), color=_COLORS[outcome])
    quoted = _clipped(text=ticket.summary_line, limit=_MAX_QUOTED_CHARS)
    embed.description = f"> {quoted}"
    if comment is None:
        embed.description = f"{embed.description}\n\n-# {_NO_COMMENT[outcome]}"
    else:
        embed.add_field(
            name=f"開發者 · {comment.created_at[:10]}",
            value=_clipped(text=text, limit=_MAX_COMMENT_CHARS) or "（空白）",
            inline=False,
        )
    embed.set_footer(text="用 /feedback 可以看這張單的完整對話")
    return embed
