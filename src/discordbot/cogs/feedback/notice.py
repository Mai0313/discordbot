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

from openai import AsyncOpenAI
from nextcord import Embed
from pydantic import Field, BaseModel

from discordbot.utils.llm import parse_responses_or_none
from discordbot.typings.colors import NEUTRAL_GREY, DISCORD_GREEN
from discordbot.typings.models import ModelSettings
from discordbot.cogs.feedback.github import CloseOutcome, IssueComment
from discordbot.cogs.feedback.prompts import CLOSE_NOTICE_TRANSLATION_PROMPT
from discordbot.cogs.feedback.database import FeedbackTicket

# Discord allows 1024 per field value and 4096 per description. Both caps here sit under
# those with room, and a cut is marked rather than silent: a maintainer's answer that just
# stops reads as the bot having lost the rest of it.
_MAX_COMMENT_CHARS = 900
_MAX_QUOTED_CHARS = 300
_TRUNCATED_SUFFIX = "…"

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


def closing_comment(*, comments: list[IssueComment]) -> IssueComment | None:
    """The maintainer's last word on the report, or None when they wrote nothing.

    `read_conversation` has already dropped passers-by and bots, and marks the lines this
    bot relayed for the reporter, so what is left of somebody else's is the answer itself.
    The last one rather than the one nearest the close: closing and commenting are separate
    actions, so the explanation is as often written just after the close as just before it,
    and in either order it is the newest thing the maintainer said.
    """
    theirs = [comment for comment in comments if not comment.from_reporter]
    return theirs[-1] if theirs else None


async def translate_comment(
    *, client: AsyncOpenAI, model: ModelSettings, ticket: FeedbackTicket, comment: IssueComment
) -> str | None:
    """Translates one closing comment into the reporter's language.

    Returns None when the call fails, and the caller must treat that as "not yet" rather
    than falling back to the English: the whole reason this runs is that the reporter may
    not read it.

    A report with no locale skips the model entirely and keeps the original. There is
    nothing to translate *into*, and guessing from the bot's own default language would be
    wrong for exactly the people this exists for.
    """
    body = comment.body.strip()
    if not ticket.locale:
        return body
    translated = await parse_responses_or_none(
        client=client,
        model=model,
        instructions=CLOSE_NOTICE_TRANSLATION_PROMPT,
        user_text=(
            f"<target_locale>{ticket.locale}</target_locale>\n"
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
