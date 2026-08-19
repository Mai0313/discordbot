"""Tests for the shared owned-public-message scaffolding."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from nextcord import File, Embed

from discordbot.utils import owned_message_views
from discordbot.utils.owned_message_views import OwnedPublicView

from tests.helpers.casting import as_message, as_interaction, make_not_found

if TYPE_CHECKING:
    import pytest

PANEL_TIMEOUT_SECONDS = 180


class ResponseStub:
    """Minimal interaction response stub."""

    def __init__(self) -> None:
        """Initializes captured response state."""
        self.deferred = False
        self.sent: list[dict[str, Any]] = []

    async def edit_message(self, **kwargs: Any) -> None:  # noqa: ANN401 -- test double
        """Records an edited response."""
        self.sent.append(kwargs)

    def is_done(self) -> bool:
        """Returns whether this response has been used."""
        return self.deferred or bool(self.sent)


class FollowupStub:
    """Minimal interaction followup stub."""

    def __init__(self) -> None:
        """Initializes captured followup payloads."""
        self.sent: list[dict[str, Any]] = []

    async def send(self, **kwargs: Any) -> MessageStub:  # noqa: ANN401 -- test double
        """Records a followup send."""
        self.sent.append(kwargs)
        return MessageStub()


class MessageStub:
    """Minimal sent message stub."""

    def __init__(self) -> None:
        """Initializes fake message identity."""
        self.id = 123
        self.channel = SimpleNamespace(id=456)
        self.edits: list[dict[str, Any]] = []
        self.deleted = False

    async def edit(self, **kwargs: Any) -> None:  # noqa: ANN401 -- test double
        """Records a message edit."""
        self.edits.append(kwargs)

    async def delete(self) -> None:
        """Records message deletion."""
        self.deleted = True


class DeletedMessageStub(MessageStub):
    """Message stub that has already been deleted remotely."""

    async def edit(self, **kwargs: Any) -> None:  # noqa: ANN401 -- test double
        """Raises the same exception nextcord emits for deleted messages."""
        raise make_not_found(message="missing")


class UserStub:
    """Minimal user stub."""

    def __init__(self, user_id: int = 1, name: str = "alice") -> None:
        """Initializes fake user identity."""
        self.id = user_id
        self.name = name


class InteractionStub:
    """Minimal interaction stub."""

    def __init__(self, user_id: int = 1, name: str = "alice") -> None:
        """Initializes fake Discord interaction pieces."""
        self.user = UserStub(user_id=user_id, name=name)
        self.guild = None
        self.response = ResponseStub()
        self.followup = FollowupStub()
        self.message = MessageStub()


def _panel(owner_id: int = 1) -> OwnedPublicView:
    """Builds a bare owned panel view."""
    return OwnedPublicView(
        owner_id=owner_id,
        timeout_seconds=PANEL_TIMEOUT_SECONDS,
        owner_mismatch_notice="這個面板不是你開的",
    )


async def test_edit_owned_public_message_recovers_when_target_was_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale panel message edit sends a public followup instead of dropping the result."""
    forgotten: list[int] = []
    tracked: list[MessageStub] = []

    async def fake_forget(message_id: int) -> None:
        """Records the stale cleanup row removal."""
        forgotten.append(message_id)

    async def fake_track(message: MessageStub, user_name: str | None = None) -> None:
        """Records the replacement cleanup row."""
        tracked.append(message)

    monkeypatch.setattr(owned_message_views, "forget_public_message", fake_forget)
    monkeypatch.setattr(owned_message_views, "track_public_message", fake_track)
    interaction = InteractionStub()
    interaction.response.deferred = True
    interaction.message = DeletedMessageStub()
    view = _panel()
    chart_file = File(fp=BytesIO(b"chart-bytes"), filename="chart.png")

    await owned_message_views.edit_owned_public_message(
        interaction=as_interaction(fake=interaction),
        embed=Embed(title="面板更新"),
        view=view,
        file=chart_file,
        message=as_message(fake=interaction.message),
    )

    assert interaction.followup.sent[0].get("ephemeral") is not True
    assert interaction.followup.sent[0]["view"] is view
    assert interaction.followup.sent[0]["files"][0] is not chart_file
    assert interaction.followup.sent[0]["files"][0].filename == "chart.png"
    assert interaction.followup.sent[0]["files"][0].fp.read() == b"chart-bytes"
    assert view.message is not interaction.message
    assert forgotten == [interaction.message.id]
    assert tracked == [view.message]


async def test_owned_public_view_timeout_deletes_bound_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idle panel delegates its message removal to shared public-message cleanup."""
    deleted: list[MessageStub] = []

    async def fake_delete(message: MessageStub) -> None:
        """Records delegated public-message deletion."""
        deleted.append(message)

    monkeypatch.setattr(owned_message_views, "delete_public_message", fake_delete)
    message = MessageStub()
    view = _panel()
    view.bind_message(message=as_message(fake=message))

    await view.on_timeout()

    assert deleted == [message]
