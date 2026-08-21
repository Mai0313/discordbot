"""The GitHub issues surface behind `/feedback`.

REST over `httpx`, which is already a dependency. Not the `gh` CLI (absent from the
runtime image, and it would need its own credential setup) and not an MCP server (an
agent-side tool, not a runtime library). Only six calls are needed, so the client is a
thin object rather than a dependency.

Every call raises `GitHubIssuesError` on a non-2xx answer. Nothing is degraded here: the
submit path decides what a failure means (the report is already stored locally by then),
and the panel decides what an unreadable issue looks like on screen.

A fresh `AsyncClient` per call is deliberate. The volume is a handful of requests per
panel open, and a client held on the cog would outlive the event loop it was built on.
"""

import re
from typing import Any, Final, Literal

import httpx
import logfire
from pydantic import Field, BaseModel, ValidationError

from discordbot.typings.timeouts import GITHUB_API_TIMEOUT_SECONDS
from discordbot.cogs.feedback.auth import GitHubAuthError, GitHubCredentials

_API_ROOT: Final[str] = "https://api.github.com"
_API_VERSION: Final[str] = "2022-11-28"

# Who counts as "the developer" when a reply is shown to the person who filed the report.
# The repository is public, so anyone can comment on an issue; relaying a stranger's
# comment as the developer's answer is the worst thing this feature could do.
MAINTAINER_ASSOCIATIONS: Final[frozenset[str]] = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})

# A reply relayed from the panel is posted with this bot's own credential, so GitHub
# attributes it to the token's owner and it would otherwise come back as a maintainer
# answering — the reporter would be shown their own words as the developer's reply.
# The marker is how a relayed line is told apart from a real one on the way back.
REPORTER_COMMENT_MARKER: Final[str] = "<!-- feedback:reporter -->"

_UNPROCESSABLE = 422

# The comments endpoint answers 30 per page by default; 100 is its maximum, and three
# pages is far past any report thread while keeping a slow issue from stalling the panel.
# Which three pages is the load-bearing part; `read_conversation` has it, along with why
# a thread deeper than the bound costs a fourth request the panel waits on.
_COMMENTS_PER_PAGE = 100
_MAX_COMMENT_PAGES = 3

# GitHub numbers its pages in the `Link` header rather than counting them in the body, so
# the entry marked `rel="last"` is where a paged read learns how deep a thread is. The
# page number is matched anywhere inside that entry's URL rather than at its end, since
# nothing promises which query parameter comes last.
_LAST_PAGE_RE: Final[re.Pattern[str]] = re.compile(r'<[^>]*[?&]page=(\d+)[^>]*>;\s*rel="last"')


class GitHubIssuesError(RuntimeError):
    """A GitHub REST call did not produce a usable answer.

    `status_code` carries GitHub's own status when there was one, and is `None` when the
    call never got that far (transport error, or a 2xx body that would not decode).
    Callers that branch on a status read it here rather than matching on the message.

    Attributes:
        status_code: The HTTP status GitHub answered with, if it answered at all.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """Initializes the error with the message and the status behind it."""
        super().__init__(message)
        self.status_code = status_code


# What a closed issue means to the person who reported it. GitHub's own vocabulary is
# `IssueClosedStateReason`, which is exactly `COMPLETED` / `NOT_PLANNED` / `DUPLICATE`.
CloseOutcome = Literal["completed", "not_planned", "duplicate"]


def close_outcome(*, state_reason: str | None) -> CloseOutcome:
    """Maps GitHub's close reason onto what it means for the reporter.

    Anything unrecognised reads as completed, which is what a plain Close has always
    produced: the UI's default button sets `completed`, and an issue closed before GitHub
    recorded a reason at all carries `None`. The two that are named here are the ones that
    would otherwise be reported as work that got done, and neither of them is.

    One function rather than a branch in each caller, because the panel and the close
    notice describe the same issue to the same person; two independent readings of one
    `state_reason` is how they end up saying different things about it.
    """
    if state_reason == "not_planned":
        return "not_planned"
    if state_reason == "duplicate":
        return "duplicate"
    return "completed"


class IssueSnapshot(BaseModel):
    """The state of one issue as the panel needs it."""

    number: int = Field(..., description="The issue number, which is the ticket number.")
    title: str = Field(..., description="Current issue title.")
    state: Literal["open", "closed"] = Field(..., description="Whether the issue is still open.")
    state_reason: str | None = Field(
        ..., description="Why it was closed: completed, not_planned, reopened, or None."
    )
    comment_count: int = Field(..., description="Total comments, maintainer or not.")


class IssueComment(BaseModel):
    """One comment on an issue that belongs in the reporter's view of it."""

    author: str = Field(..., description="GitHub login of whoever wrote it.")
    body: str = Field(..., description="Comment body, as written.")
    created_at: str = Field(..., description="ISO-8601 timestamp the comment was posted.")
    from_reporter: bool = Field(
        ..., description="Whether this is a line the reporter sent through the panel."
    )


def select_conversation(*, comments: list[dict[str, Any]]) -> list[IssueComment]:
    """Keeps only the comments that are part of the reporter's own conversation.

    Two voices survive: a maintainer, and the reporter themselves through the panel.
    Everyone else is dropped, and that is the point rather than tidiness — the
    repository is public, so a passer-by can comment on anyone's report, and showing
    their words as the developer's answer is the worst thing this feature could do.
    Bots go too: a dependency or CI note is not an answer to the person who filed it.
    """
    conversation: list[IssueComment] = []
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        author = comment.get("user") or {}
        body = str(comment.get("body") or "")
        # "Ours" covers both credentials: a GitHub App comments as a Bot, a token comments
        # as the account that owns it. Checking it is what keeps the marker from being a
        # thing any passer-by could paste to have their words shown back to the reporter
        # as the reporter's own — and checking it BEFORE the bot filter is what stops an
        # App's relayed replies from being dropped as bot noise.
        ours = (
            author.get("type") == "Bot"
            or comment.get("author_association") in MAINTAINER_ASSOCIATIONS
        )
        from_reporter = ours and body.lstrip().startswith(REPORTER_COMMENT_MARKER)
        if not from_reporter and author.get("type") == "Bot":
            continue
        if not from_reporter and comment.get("author_association") not in MAINTAINER_ASSOCIATIONS:
            continue
        conversation.append(
            IssueComment(
                author=str(author.get("login") or "maintainer"),
                body=body.replace(REPORTER_COMMENT_MARKER, "").strip(),
                created_at=str(comment.get("created_at") or ""),
                from_reporter=from_reporter,
            )
        )
    return conversation


class GitHubIssues(BaseModel):
    """Opens, edits, and reads the issues that user reports become.

    Attributes:
        credentials: How the bot proves it may act on `repository`; see `auth.py`.
        repository: The `owner/name` slug reports are filed against.
        timeout_seconds: Per-request ceiling. Short on purpose: every caller is either
            on the submit path or on a panel someone is waiting for.
    """

    credentials: GitHubCredentials = Field(..., description="How the bot authorizes a call.")
    repository: str = Field(..., description="The owner/name slug reports are filed against.")
    timeout_seconds: float = Field(
        default=GITHUB_API_TIMEOUT_SECONDS, description="Per-request timeout in seconds."
    )

    async def _authorized_headers(self) -> dict[str, str]:
        """Auth and version headers sent on every request.

        Asynchronous because a GitHub App mints its token on demand; a failure to do so
        arrives as this module's own error, so callers still catch exactly one type.

        Raises:
            GitHubIssuesError: The bot could not authorize the call.
        """
        try:
            authorization = await self.credentials.authorization()
        except GitHubAuthError as exc:
            raise GitHubIssuesError(f"Could not authorize a GitHub call: {exc}") from exc
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": authorization,
            "X-GitHub-Api-Version": _API_VERSION,
        }

    async def _send(
        self, *, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> httpx.Response:
        """Runs one REST call and hands back the answer GitHub gave.

        Split out of `_request` for the one reader that needs something off the response
        that is not in its body: the paged comment read, whose `Link` header is where
        GitHub says how many pages a thread has.

        Raises:
            GitHubIssuesError: The request never reached GitHub, or GitHub answered with
                a non-2xx status.
        """
        url = f"{_API_ROOT}/repos/{self.repository}{path}"
        headers = await self._authorized_headers()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.request(
                    method=method, url=url, headers=headers, json=payload
                )
        except httpx.HTTPError as exc:
            raise GitHubIssuesError(f"{method} {path} could not reach GitHub: {exc}") from exc
        if not response.is_success:
            raise GitHubIssuesError(
                f"{method} {path} answered {response.status_code}: {response.text[:500]}",
                response.status_code,
            )
        return response

    async def _request(
        self, *, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:  # noqa: ANN401 -- the GitHub REST payload shape differs per endpoint
        """Runs one REST call and returns its decoded body.

        Decoding is inside the contract, not next to it: every caller treats
        `GitHubIssuesError` as the one thing this module raises, so a 2xx carrying an
        HTML error page from something in front of GitHub has to arrive as that too,
        rather than as a `JSONDecodeError` nobody upstream is catching.

        Raises:
            GitHubIssuesError: The request never reached GitHub, GitHub answered with a
                non-2xx status, or the answer was not decodable JSON.
        """
        response = await self._send(method=method, path=path, payload=payload)
        try:
            return response.json()
        except ValueError as exc:
            raise GitHubIssuesError(
                f"{method} {path} answered {response.status_code} with undecodable content: "
                f"{response.text[:200]}",
                response.status_code,
            ) from exc

    async def create_issue(self, *, title: str, body: str, labels: list[str]) -> int:
        """Opens an issue and returns its number.

        GitHub answers 422 when it will not accept the request as sent, and a label the
        repository does not carry is one of the reasons. Losing a whole report over a
        piece of metadata is the worse outcome, so a 422 retries bare — and says so,
        because `user-report` is the handle that would rebuild the local store, and a
        silent drop would take that away with nothing to notice it by.

        Raises:
            GitHubIssuesError: The issue could not be created.
        """
        payload: dict[str, Any] = {"title": title, "body": body, "labels": labels}
        try:
            created = await self._request(method="POST", path="/issues", payload=payload)
        except GitHubIssuesError as exc:
            if not labels or exc.status_code != _UNPROCESSABLE:
                raise
            logfire.warn(
                "GitHub refused the labels; opening the issue without them",
                repository=self.repository,
                labels=labels,
                _exc_info=exc,
            )
            created = await self._request(
                method="POST", path="/issues", payload={"title": title, "body": body}
            )
        return int(created["number"])

    async def update_issue(self, *, number: int, title: str, body: str) -> None:
        """Rewrites an issue's title and body.

        Raises:
            GitHubIssuesError: The edit failed.
        """
        await self._request(
            method="PATCH", path=f"/issues/{number}", payload={"title": title, "body": body}
        )

    async def add_labels(self, *, number: int, labels: list[str]) -> None:
        """Adds labels to an existing issue, leaving the ones already on it alone.

        Raises:
            GitHubIssuesError: The labels could not be added.
        """
        await self._request(
            method="POST", path=f"/issues/{number}/labels", payload={"labels": labels}
        )

    async def read_issue(self, *, number: int) -> IssueSnapshot:
        """Reads one issue's current state.

        A payload that does not carry what a snapshot needs is reported as this module's
        own error, for the same reason decoding is: the panel catches one type, and a
        `KeyError` from here would sail past it and take the whole list down.

        Raises:
            GitHubIssuesError: The issue could not be read or did not parse.
        """
        issue = await self._request(method="GET", path=f"/issues/{number}")
        state = "closed" if issue.get("state") == "closed" else "open"
        try:
            return IssueSnapshot(
                number=int(issue["number"]),
                title=str(issue.get("title") or ""),
                state=state,
                state_reason=issue.get("state_reason"),
                comment_count=int(issue.get("comments") or 0),
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise GitHubIssuesError(f"GET /issues/{number} answered an unreadable issue") from exc

    async def _read_comment_page(self, *, number: int, page: int) -> tuple[list[Any], int]:
        """Reads one page of an issue's comments, and how many pages there are.

        The page count comes off the `Link` header rather than the body, which is why
        this goes through `_send`. GitHub leaves `rel="last"` off the last page, so its
        absence means the page in hand is that one, and the answer is the page asked for.

        Raises:
            GitHubIssuesError: The page could not be read, or was not a list of comments.
        """
        response = await self._send(
            method="GET",
            path=f"/issues/{number}/comments?per_page={_COMMENTS_PER_PAGE}&page={page}",
        )
        try:
            comments = response.json()
        except ValueError as exc:
            raise GitHubIssuesError(
                f"GET /issues/{number}/comments answered undecodable content: "
                f"{response.text[:200]}",
                response.status_code,
            ) from exc
        if not isinstance(comments, list):
            raise GitHubIssuesError(f"GET /issues/{number}/comments answered a non-list body")
        last = _LAST_PAGE_RE.search(response.headers.get("link", ""))
        return comments, int(last.group(1)) if last else page

    async def read_conversation(self, *, number: int) -> list[IssueComment]:
        """Reads the comments that belong in the reporter's view of their own report.

        Paged rather than one call: the endpoint answers 30 at a time by default, and the
        panel shows the *newest* replies, so a thread past the first page would hide the
        very answer the reporter came to read while the status still said someone had
        replied.

        `_MAX_COMMENT_PAGES` bounds that walk, and it is applied at the END of the thread:
        the endpoint answers oldest first, so a bound counted from page 1 would drop
        precisely the replies the panel renders. Asking for the other order is not an
        option — measured 2026-08-20 against api.github.com, this endpoint ignores both
        `sort` and `direction` and answers ascending whatever it is passed — so the walk
        starts `_MAX_COMMENT_PAGES` back from the last page instead. A thread inside the
        bound is read whole in the same requests as before; only a deeper one pays for a
        fourth, since page 1's body is thrown away and the page is asked for anyway to
        reach its `Link` header. The issue's own comment count would answer the same
        question for free — the panel holds one two statements earlier — but it is read
        before the walk, so one comment landing in between puts the newest on a page the
        arithmetic then never asks for, which is this bug again through a narrower door.

        Raises:
            GitHubIssuesError: The comments could not be read.
        """
        first, last_page = await self._read_comment_page(number=number, page=1)
        start = max(1, last_page - _MAX_COMMENT_PAGES + 1)
        collected: list[dict[str, Any]] = []
        if start == 1:
            collected.extend(first)
        for page in range(max(start, 2), last_page + 1):
            comments, _ = await self._read_comment_page(number=number, page=page)
            collected.extend(comments)
        return select_conversation(comments=collected)

    async def add_comment(self, *, number: int, body: str) -> None:
        """Posts a comment on an issue.

        Raises:
            GitHubIssuesError: The comment could not be posted.
        """
        await self._request(
            method="POST", path=f"/issues/{number}/comments", payload={"body": body}
        )
