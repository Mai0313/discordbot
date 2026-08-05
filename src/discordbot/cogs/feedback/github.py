"""The GitHub issues surface behind `/feedback`.

REST over `httpx`, which is already a dependency. Not the `gh` CLI (absent from the
runtime image, and it would need its own credential setup) and not an MCP server (an
agent-side tool, not a runtime library). Only five calls are needed, so the client is a
thin object rather than a dependency.

Every call raises `GitHubIssuesError` on a non-2xx answer. Nothing is degraded here: the
submit path decides what a failure means (the report is already stored locally by then),
and the panel decides what an unreadable issue looks like on screen.

A fresh `AsyncClient` per call is deliberate. The volume is a handful of requests per
panel open, and a client held on the cog would outlive the event loop it was built on.
"""

from typing import Any, Final, Literal

import httpx
from pydantic import Field, BaseModel

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


class GitHubIssuesError(RuntimeError):
    """A GitHub REST call answered with something other than success."""


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
        author = comment.get("user") or {}
        body = str(comment.get("body") or "")
        from_reporter = body.lstrip().startswith(REPORTER_COMMENT_MARKER)
        if author.get("type") == "Bot":
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
        token: Credential with issue read/write on `repository`.
        repository: The `owner/name` slug reports are filed against.
        timeout_seconds: Per-request ceiling. Short on purpose: every caller is either
            on the submit path or on a panel someone is waiting for.
    """

    token: str = Field(..., description="Token with issue read/write on the repository.")
    repository: str = Field(..., description="The owner/name slug reports are filed against.")
    timeout_seconds: float = Field(default=15.0, description="Per-request timeout in seconds.")

    @property
    def _headers(self) -> dict[str, str]:
        """Auth and version headers sent on every request."""
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": _API_VERSION,
        }

    async def _request(
        self, *, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:  # noqa: ANN401 -- the GitHub REST payload shape differs per endpoint
        """Runs one REST call and returns its decoded body.

        Raises:
            GitHubIssuesError: The request failed to reach GitHub, or GitHub answered
                with a non-2xx status.
        """
        url = f"{_API_ROOT}/repos/{self.repository}{path}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.request(
                    method=method, url=url, headers=self._headers, json=payload
                )
        except httpx.HTTPError as exc:
            raise GitHubIssuesError(f"{method} {path} could not reach GitHub: {exc}") from exc
        if response.is_success:
            return response.json()
        raise GitHubIssuesError(
            f"{method} {path} answered {response.status_code}: {response.text[:500]}"
        )

    async def create_issue(self, *, title: str, body: str, labels: list[str]) -> int:
        """Opens an issue and returns its number.

        A label the repository does not have makes GitHub reject the whole create, which
        would cost the report over a piece of metadata, so that one case retries bare.

        Raises:
            GitHubIssuesError: The issue could not be created.
        """
        payload: dict[str, Any] = {"title": title, "body": body, "labels": labels}
        try:
            created = await self._request(method="POST", path="/issues", payload=payload)
        except GitHubIssuesError as exc:
            if labels and f"answered {_UNPROCESSABLE}" in str(exc):
                created = await self._request(
                    method="POST", path="/issues", payload={"title": title, "body": body}
                )
            else:
                raise
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

        Raises:
            GitHubIssuesError: The issue could not be read.
        """
        issue = await self._request(method="GET", path=f"/issues/{number}")
        state = "closed" if issue.get("state") == "closed" else "open"
        return IssueSnapshot(
            number=int(issue["number"]),
            title=str(issue.get("title") or ""),
            state=state,
            state_reason=issue.get("state_reason"),
            comment_count=int(issue.get("comments") or 0),
        )

    async def read_conversation(self, *, number: int) -> list[IssueComment]:
        """Reads the comments that belong in the reporter's view of their own report.

        Raises:
            GitHubIssuesError: The comments could not be read.
        """
        comments = await self._request(method="GET", path=f"/issues/{number}/comments")
        return select_conversation(comments=comments)

    async def add_comment(self, *, number: int, body: str) -> None:
        """Posts a comment on an issue.

        Raises:
            GitHubIssuesError: The comment could not be posted.
        """
        await self._request(
            method="POST", path=f"/issues/{number}/comments", payload={"body": body}
        )
