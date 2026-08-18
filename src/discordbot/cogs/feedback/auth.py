"""How the bot proves it may open issues on the reporting repository.

Two ways, and the difference is visible to everyone reading the issues. A GitHub App
files them as `<app>[bot]`, which is what they are; a personal token files them as the
person whose token it is, which reads as the maintainer opening their own reports and,
more practically, means GitHub never notifies them about one — you are not told about
your own actions.

The App path costs one indirection: a JWT signed with the app's private key buys an
installation token that lasts an hour, so the long-lived secret on disk is never what
goes to the API and a leaked request carries something that expires on its own.

This module knows nothing about issues. `github.py` depends on it and not the reverse,
so the credential can be swapped without touching a single call site.
"""

from typing import Any, Final
from pathlib import Path
from datetime import UTC, datetime, timedelta

import jwt
import httpx
from pydantic import Field, BaseModel, PrivateAttr

from discordbot.typings.timeouts import GITHUB_REQUEST_TIMEOUT_SECONDS

_API_ROOT: Final[str] = "https://api.github.com"
_API_VERSION: Final[str] = "2022-11-28"

# GitHub rejects a JWT whose lifetime is over ten minutes. Nine leaves room for the
# backdated `iat` below without crossing that line.
_JWT_LIFETIME_SECONDS: Final[int] = 9 * 60

# GitHub's own advice: backdate `iat` so a clock a little ahead of theirs does not have
# its token refused as issued in the future.
_JWT_BACKDATE_SECONDS: Final[int] = 60

# An installation token lasts an hour. Renewing early means a request never carries one
# that expires while it is in flight.
_RENEW_BEFORE = timedelta(minutes=5)


class GitHubAuthError(RuntimeError):
    """The bot could not prove it may act on the repository."""


class TokenCredentials(BaseModel):
    """A personal or machine-account token, used as-is."""

    token: str = Field(..., description="Token with issue read/write on the repository.")

    async def authorization(self) -> str:
        """The Authorization header value for a GitHub REST call."""
        return f"Bearer {self.token}"


class AppCredentials(BaseModel):
    """A GitHub App installation, which mints its own short-lived tokens.

    Attributes:
        app_id: The app's numeric id, used as the JWT issuer.
        private_key_path: PEM the JWT is signed with. A path rather than the key itself
            because a PEM is multi-line and would have to be escaped into the environment.
        repository: The `owner/name` whose installation is used.
        timeout_seconds: Per-request ceiling for the two auth calls.
    """

    app_id: str = Field(..., description="The GitHub App's numeric id.")
    private_key_path: Path = Field(..., description="Path to the app's private key PEM.")
    repository: str = Field(..., description="The owner/name the app is installed on.")
    timeout_seconds: float = Field(
        default=GITHUB_REQUEST_TIMEOUT_SECONDS, description="Per-request timeout in seconds."
    )

    _installation_id: int | None = PrivateAttr(default=None)
    _token: str = PrivateAttr(default="")
    _token_expires_at: datetime = PrivateAttr(default=datetime.min.replace(tzinfo=UTC))

    def _signed_jwt(self) -> str:
        """Signs the short-lived JWT that identifies the app itself.

        Raises:
            GitHubAuthError: The private key is missing or unusable.
        """
        try:
            private_key = self.private_key_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise GitHubAuthError(f"Cannot read the app private key: {exc}") from exc
        now = datetime.now(tz=UTC)
        try:
            return jwt.encode(
                {
                    "iat": int((now - timedelta(seconds=_JWT_BACKDATE_SECONDS)).timestamp()),
                    "exp": int((now + timedelta(seconds=_JWT_LIFETIME_SECONDS)).timestamp()),
                    "iss": self.app_id,
                },
                private_key,
                algorithm="RS256",
            )
        # Broad on purpose: PyJWT and cryptography raise unrelated types for a key that
        # is the wrong format, encrypted, or truncated, and every one of them means the
        # same thing to the caller.
        except Exception as exc:
            raise GitHubAuthError(f"Cannot sign with the app private key: {exc}") from exc

    async def _call(self, *, method: str, path: str, authorization: str) -> Any:  # noqa: ANN401 -- the GitHub payload shape differs per endpoint
        """Runs one auth-flow request against GitHub.

        Raises:
            GitHubAuthError: The request failed or GitHub refused it.
        """
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": authorization,
            "X-GitHub-Api-Version": _API_VERSION,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.request(
                    method=method, url=f"{_API_ROOT}{path}", headers=headers
                )
        except httpx.HTTPError as exc:
            raise GitHubAuthError(f"{method} {path} could not reach GitHub: {exc}") from exc
        if not response.is_success:
            raise GitHubAuthError(
                f"{method} {path} answered {response.status_code}: {response.text[:300]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise GitHubAuthError(f"{method} {path} answered undecodable content") from exc

    async def _resolve_installation(self, *, app_jwt: str) -> int:
        """Finds the installation covering the reporting repository, once per process.

        Looked up rather than configured: the id is not shown anywhere obvious during
        setup, and one more number to copy by hand is one more way to get it wrong.

        Raises:
            GitHubAuthError: The app is not installed on that repository.
        """
        if self._installation_id is not None:
            return self._installation_id
        installation = await self._call(
            method="GET",
            path=f"/repos/{self.repository}/installation",
            authorization=f"Bearer {app_jwt}",
        )
        try:
            self._installation_id = int(installation["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GitHubAuthError(
                "GitHub did not name an installation for this repository"
            ) from exc
        return self._installation_id

    async def authorization(self) -> str:
        """The Authorization header value, minting or reusing an installation token.

        Raises:
            GitHubAuthError: The token could not be obtained.
        """
        if self._token and datetime.now(tz=UTC) < self._token_expires_at - _RENEW_BEFORE:
            return f"Bearer {self._token}"
        app_jwt = self._signed_jwt()
        installation_id = await self._resolve_installation(app_jwt=app_jwt)
        minted = await self._call(
            method="POST",
            path=f"/app/installations/{installation_id}/access_tokens",
            authorization=f"Bearer {app_jwt}",
        )
        token = str(minted.get("token") or "")
        if not token:
            raise GitHubAuthError("GitHub returned an installation without a token")
        self._token = token
        self._token_expires_at = _parse_expiry(value=minted.get("expires_at"))
        return f"Bearer {self._token}"


def _parse_expiry(*, value: object) -> datetime:
    """Reads GitHub's expiry timestamp, falling back to the documented one-hour life.

    A missing or odd timestamp is not worth failing over: the fallback counts the hour
    from now rather than from when GitHub minted the token, so it lands a moment past the
    real expiry, and `_RENEW_BEFORE` is what absorbs the difference.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(tz=UTC) + timedelta(hours=1)


GitHubCredentials = TokenCredentials | AppCredentials
