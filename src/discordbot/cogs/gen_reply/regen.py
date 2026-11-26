"""Reply regeneration workflow triggered by the feedback view's `🔄` button.

Validates caller permissions and chain limits, serializes concurrent clicks per reply message,
re-runs the original gateway QA prompt with blind-test model exclusion and preserved reasoning
effort, records implicit feedback, and deletes the superseded reply chunks once the replacement
lands.
"""

from typing import Literal, Protocol, cast, runtime_checkable
import contextlib
from collections.abc import Callable

import logfire
from nextcord import (
    User,
    Message,
    NotFound,
    Forbidden,
    Interaction,
    HTTPException,
    PartialMessageable,
)
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation
from nextcord.ext import commands

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import SlowModelSettings
from discordbot.utils.reactions import ReactionStatusChain
from discordbot.utils.llm_errors import extract_friendly_error
from discordbot.utils.asyncio_locks import KeyedLockManager
from discordbot.utils.discord_errors import report_discord_failure
from discordbot.cogs.gen_reply.answer import RegenOptions, max_regens_for
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.cogs.gen_reply.turn_state import dispatched_model, current_answer_streamer
from discordbot.cogs.gen_reply.status_marks import FAILED_EMOJI
from discordbot.cogs.gen_reply.feedback_store import (
    RegenChainState,
    load_reply_turn,
    record_regen_feedback,
    load_regen_chain_state,
    parse_reply_message_ids,
)


@runtime_checkable
class _MessageFetchable(Protocol):
    """Minimal channel surface that can fetch a message by id."""

    async def fetch_message(self, message_id: int, /) -> Message:
        """Fetches one Discord message by id."""
        ...


@runtime_checkable
class _DeletableMessage(Protocol):
    """Minimal message handle that can be deleted."""

    async def delete(self) -> None:
        """Deletes the message."""
        ...


@runtime_checkable
class _PartialMessageChannel(Protocol):
    """Channel surface exposing `get_partial_message` for one-call deletion."""

    def get_partial_message(self, message_id: int, /) -> _DeletableMessage:
        """Returns a partial message handle without an extra REST fetch."""
        ...


class StartRegenTurn(Protocol):
    """Callback that executes one replacement QA turn on a resolved `TurnSurface`."""

    async def __call__(
        self,
        surface: TurnSurface,
        content: str,
        reactions: ReactionStatusChain,
        regen: RegenOptions | None = None,
    ) -> int | None:
        """Runs the turn and returns the replacement reply message id, or None when empty."""
        ...


async def _delete_dangling_reply() -> None:
    """Best-effort deletes any partial reply chunks left behind by a failed regen stream."""
    streamer = current_answer_streamer.get()
    if streamer is None:
        return
    dangling = list(streamer.reply_messages)
    if not dangling and streamer.reply is not None:
        dangling = [streamer.reply]
    for chunk in reversed(dangling):
        with contextlib.suppress(HTTPException):
            await chunk.delete()


class ReplyRegenerator(BaseModel):
    """Validates, serializes, executes, and cleans up `🔄` reply regenerations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bot: SkipValidation[commands.Bot] = Field(
        ..., description="The Discord bot instance used for channel lookups and owner checks."
    )
    config: LLMConfig = Field(
        ..., description="Runtime LLM and feedback configuration governing regen limits."
    )
    slow_candidates: Callable[[], tuple[SlowModelSettings, ...]] = Field(
        ...,
        description=(
            "Zero-arg provider returning configured QA slow candidates, read lazily so "
            "constructing the cog never builds the toolkit eagerly."
        ),
    )
    start_turn: SkipValidation[StartRegenTurn] = Field(
        ..., description="Cog callback executing the replacement QA turn."
    )
    _regen_locks: KeyedLockManager[int] = PrivateAttr(default_factory=KeyedLockManager)

    async def handle(self, interaction: Interaction[commands.Bot]) -> None:
        """Validates permissions/limits, regenerates a QA reply, and removes the old reply after delivery.

        Args:
            interaction (Interaction[commands.Bot]): Component interaction clicking `🔄`.
        """
        if interaction.message is None or interaction.user is None:
            return
        reply_message_id = interaction.message.id
        await interaction.response.defer(ephemeral=True, with_message=True)
        try:
            refusal = await self._validate_access(
                interaction=interaction, reply_message_id=reply_message_id
            )
            if refusal is not None:
                await interaction.edit_original_message(content=refusal)
                return
            async with self._regen_locks.hold(key=reply_message_id):
                await self._run_locked(interaction=interaction, reply_message_id=reply_message_id)
        except Exception as exc:
            # Broad on purpose: deferred button callbacks must finish with a notice on failure.
            logfire.error(
                "Failed to handle reply regen",
                reply_message_id=reply_message_id,
                user_id=interaction.user.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            with contextlib.suppress(HTTPException):
                await interaction.edit_original_message(content="重新生成時發生錯誤，請稍後再試")

    async def _resolve_channel(
        self, interaction: Interaction[commands.Bot], channel_id: int, reply_message_id: int
    ) -> _MessageFetchable | None:
        """Returns the fetchable channel for `channel_id`, or edits the ephemeral notice and returns None."""
        channel: object | None = interaction.channel
        if channel is None or isinstance(channel, PartialMessageable):
            channel = self.bot.get_channel(channel_id)
        if channel is None or isinstance(channel, PartialMessageable):
            try:
                channel = await self.bot.fetch_channel(channel_id)
            except HTTPException as exc:
                report_discord_failure(
                    error=exc,
                    not_found_message=None,
                    forbidden_message="Channel refused lookup for reply regen",
                    other_message="Failed to fetch channel for reply regen",
                    channel_id=channel_id,
                    reply_message_id=reply_message_id,
                )
                if isinstance(exc, NotFound):
                    notice = "找不到原頻道，無法重新生成"
                elif isinstance(exc, Forbidden):
                    notice = "沒有權限讀取原頻道，無法重新生成"
                else:
                    notice = "讀取原頻道時發生錯誤，請稍後再試"
                await interaction.edit_original_message(content=notice)
                return None
        if not isinstance(channel, _MessageFetchable):
            await interaction.edit_original_message(content="這個頻道不支援重新生成")
            return None
        return channel

    async def _fetch_source(
        self,
        interaction: Interaction[commands.Bot],
        channel: _MessageFetchable,
        state: RegenChainState,
    ) -> Message | None:
        """Fetches the original asking message for regen, reporting failures ephemerally."""
        try:
            return await channel.fetch_message(state.source_message_id)
        except HTTPException as exc:
            report_discord_failure(
                error=exc,
                not_found_message=None,
                forbidden_message="Channel refused source message fetch for reply regen",
                other_message="Failed to fetch source message for reply regen",
                channel_id=state.channel_id,
                message_id=state.source_message_id,
                reply_message_id=state.reply_message_id,
            )
            if isinstance(exc, NotFound):
                notice = "找不到原始發問訊息，無法重新生成"
            elif isinstance(exc, Forbidden):
                notice = "沒有權限讀取原始訊息，無法重新生成"
            else:
                notice = "讀取原始訊息時發生錯誤，請稍後再試"
            await interaction.edit_original_message(content=notice)
            return None

    async def _delete_old_chunks(self, channel: _MessageFetchable, state: RegenChainState) -> None:
        """Deletes old reply chunk messages in reverse order after a replacement lands."""
        old_ids = parse_reply_message_ids(
            raw=state.reply_message_ids, fallback_id=state.reply_message_id
        )
        for old_id in reversed(old_ids):
            try:
                if isinstance(channel, _PartialMessageChannel):
                    await channel.get_partial_message(old_id).delete()
                else:
                    old_message = await channel.fetch_message(old_id)
                    await old_message.delete()
            except HTTPException as exc:
                report_discord_failure(
                    error=exc,
                    not_found_message=None,
                    forbidden_message="Channel refused old reply deletion after regen",
                    other_message="Failed to delete old reply chunk after regen",
                    channel_id=state.channel_id,
                    message_id=old_id,
                    reply_message_id=state.reply_message_id,
                )
                if isinstance(exc, Forbidden):
                    break

    async def _execute_turn(
        self,
        interaction: Interaction[commands.Bot],
        source_message: Message,
        state: RegenChainState,
        max_regens: int,
        multi_candidate: bool,
    ) -> int | None:
        """Runs the replacement QA turn, cleaning up dangling previews on failure."""
        remaining = max_regens - (state.chain_depth + 1)
        if multi_candidate:
            progress = (
                f"正在換另一個模型重新生成中⋯（本次後剩餘 {remaining} 次重抽）"
                if remaining > 0
                else "正在換另一個模型重新生成中⋯（最後一次重抽）"
            )
        else:
            progress = "正在重新生成回覆中⋯"
        with contextlib.suppress(HTTPException):
            await interaction.edit_original_message(content=progress)

        surface = TurnSurface.for_message(message=source_message)
        reactions = surface.status_chain(bot_user=self.bot.user)
        forced_effort: Literal["low", "medium", "high"] | None = (
            cast("Literal['low', 'medium', 'high']", state.effort)
            if state.effort in ("low", "medium", "high")
            else None
        )
        regen = RegenOptions(
            regen_of_reply_message_id=state.reply_message_id,
            regen_count=state.chain_depth + 1,
            excluded_models=frozenset(state.tried_models),
            forced_effort=forced_effort,
        )
        dispatched_model.set(None)
        current_answer_streamer.set(None)
        try:
            new_reply_message_id = await self.start_turn(
                surface=surface, content=source_message.content, reactions=reactions, regen=regen
            )
        except Exception as exc:
            message = surface.message
            guild = message.guild
            logfire.error(
                "gen_reply regen failed",
                user_id=message.author.id,
                user_name=message.author.name,
                display_name=message.author.display_name,
                message_id=message.id,
                channel_id=message.channel.id,
                guild_id=surface.guild_id,
                guild_name=guild.name if guild else None,
                reply_message_id=state.reply_message_id,
                model=dispatched_model.get(),
                error_type=type(exc).__name__,
                _exc_info=True,
            )
            await _delete_dangling_reply()
            reactions.advance(emoji=FAILED_EMOJI)
            with contextlib.suppress(HTTPException):
                await interaction.edit_original_message(
                    content=f"重新生成失敗：{extract_friendly_error(exc=exc)}"
                )
            return None
        finally:
            dispatched_model.set(None)
            current_answer_streamer.set(None)
            await reactions.flush()
        if new_reply_message_id is None:
            with contextlib.suppress(HTTPException):
                await interaction.edit_original_message(content="原始訊息內容為空，無法重新生成")
            return None
        return new_reply_message_id

    async def _validate_access(
        self, interaction: Interaction[commands.Bot], reply_message_id: int
    ) -> str | None:
        """Returns a user-facing refusal reason before locking, or None when allowed."""
        if not self.config.feedback_enabled:
            return "目前已停用重新生成功能"
        turn = await load_reply_turn(reply_message_id=reply_message_id)
        if turn is None:
            return "這則回覆已超過可重新生成範圍"
        if turn.surface != "gateway" or turn.route != "QA":
            return "這則回覆不支援重新生成"
        user = interaction.user
        if user is None:
            return "只有原發問者可以重新生成這則回覆"
        is_allowed = user.id == turn.author_id or await self.bot.is_owner(user=cast("User", user))
        if not is_allowed:
            return "只有原發問者可以重新生成這則回覆"
        return None

    async def _run_locked(
        self, interaction: Interaction[commands.Bot], reply_message_id: int
    ) -> None:
        """Re-checks regen state under the per-reply lock, runs replacement QA, and cleans old chunks."""
        user = interaction.user
        if user is None:
            return
        state = await load_regen_chain_state(reply_message_id=reply_message_id)
        candidates = [candidate for candidate in self.slow_candidates() if candidate.weight > 0]
        max_regens = max_regens_for(candidates=candidates)
        refusal = (
            "這則回覆已超過可重新生成範圍"
            if state is None
            else "這則回覆已經重新生成過了"
            if state.already_regenerated
            else "已達重新生成次數上限"
            if state.chain_depth >= max_regens
            else None
        )
        if state is None or refusal is not None:
            await interaction.edit_original_message(
                content=refusal or "這則回覆已超過可重新生成範圍"
            )
            return

        channel = await self._resolve_channel(
            interaction=interaction, channel_id=state.channel_id, reply_message_id=reply_message_id
        )
        if channel is None:
            return
        source_message = await self._fetch_source(
            interaction=interaction, channel=channel, state=state
        )
        if source_message is None:
            return

        new_reply_message_id = await self._execute_turn(
            interaction=interaction,
            source_message=source_message,
            state=state,
            max_regens=max_regens,
            multi_candidate=len(candidates) > 1,
        )
        if new_reply_message_id is None:
            return

        try:
            await record_regen_feedback(
                reply_message_id=reply_message_id,
                user_id=user.id,
                user_name=user.name,
                user_display_name=user.display_name,
            )
        except Exception as exc:
            # Broad on purpose: best-effort implicit feedback must not block old cleanup.
            logfire.warn(
                "Failed to record implicit regen feedback",
                reply_message_id=reply_message_id,
                new_reply_message_id=new_reply_message_id,
                user_id=user.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
        await self._delete_old_chunks(channel=channel, state=state)
        with contextlib.suppress(HTTPException):
            await interaction.edit_original_message(content="已重新生成回覆")
