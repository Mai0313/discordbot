"""The reply slot an expansion cog claims before it has anything to put in it.

An expansion takes seconds: `parse_threads` walks a whole conversation, `parse_douyin`
downloads the clip. In a busy channel that is long enough for other people to talk, and the
finished card then lands several messages below the link it belongs to, which is worst for
exactly the reader it was posted for — someone tagged in it has to work out which link the
card came from. So the cog replies with one line the moment it starts and edits that same
message into the finished expansion: the card sits under the link whatever happened in
between.

A failure leaves NOTHING behind. `discard` removes the placeholder and the reaction on the
source message is the whole report, which is what every expansion cog already does.
"""

import logfire
from nextcord import File, Embed, Message, NotFound, Forbidden, HTTPException, AllowedMentions
from pydantic import Field, BaseModel, ConfigDict, SkipValidation

from discordbot.utils.discord_embeds import embed_spacer_payload

# What Discord answers when the message being replied to no longer exists. It is a generic
# invalid-form-body code, so it means this only on a send that carries a message reference.
_UNSENDABLE_REPLY = 50035


class ExpansionPlaceholder(BaseModel):
    """A posted placeholder and whether the expansion has landed on it yet.

    Attributes:
        message: The placeholder itself, the message every later edit lands on.
        delivered: Whether `deliver` succeeded, which is what stops `discard` from removing
            an expansion the reader can already see.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: SkipValidation[Message] = Field(
        ..., description="The placeholder reply the finished expansion is edited into."
    )
    delivered: bool = Field(
        default=False, description="Whether the expansion landed, making `discard` a no-op."
    )

    async def deliver(
        self, *, content: str | None = None, embeds: list[Embed], files: list[File] | None = None
    ) -> None:
        """Edits the finished expansion onto the placeholder.

        Clearing the placeholder line takes an explicit empty string, never None: an
        expansion brings files (its media, or the embed spacer), which sends the edit as
        multipart, and `http.py::get_message_payload` drops a None content out of that body
        altogether instead of clearing it. It clears on the JSON path, which is exactly what
        makes the difference easy to miss, and `streaming.py::land_failure` carries the same
        note for the same reason.

        The spacer rides as an edit so its `attachments` key drops whatever the placeholder
        held, which is also what lets `files` be the whole of the new attachment list.

        Whatever the edit raises travels out, so the cog's own failure path reports it.

        Args:
            content: The text under the expansion, or None for an expansion that is embeds
                and attachments alone.
            embeds: The finished expansion.
            files: Media attaching natively beside the embeds.
        """
        await self.message.edit(
            content=content or "",
            embeds=embeds,
            allowed_mentions=AllowedMentions.none(),
            **embed_spacer_payload(
                embeds=embeds, is_edit=True, target=self.message, extra_files=files
            ),
        )
        self.delivered = True

    async def discard(self) -> None:
        """Removes an undelivered placeholder, leaving the reaction as the whole report.

        Safe in a `finally`: it swallows its own failure rather than replacing whatever sent
        the expansion down the failure path, and a placeholder already deleted by hand is the
        outcome this wanted anyway.
        """
        if self.delivered:
            return
        try:
            await self.message.delete()
        # Broad on purpose: the expansion has already failed and the reaction says so, so
        # nothing here can improve the outcome or is worth reaching the listener's handler.
        except Exception as error:
            logfire.debug(
                "Could not remove an expansion placeholder",
                message_id=self.message.id,
                error_type=type(error).__name__,
                _exc_info=error,
            )


async def send_expansion_placeholder(
    *, message: Message, text: str
) -> ExpansionPlaceholder | None:
    """Claims the reply slot under `message` with a line saying the expansion is coming.

    A refusal here is the whole expansion's answer as well: a channel that will not take this
    one line will not take the card either, and learning it now costs no fetch. Both refusals
    are ordinary rather than defects, which is why they are classified here instead of
    reaching the listener's last-resort handler, where a misconfigured channel would log an
    error per pasted link.

    Args:
        message: The message carrying the link, which the placeholder replies to.
        text: The line to show until the expansion replaces it.

    Returns:
        The placeholder to deliver onto or discard, or None when the channel refused it and
        the caller should mark the expansion failed without reading anything.
    """
    try:
        placeholder = await message.reply(
            content=text, mention_author=False, allowed_mentions=AllowedMentions.none()
        )
    except Forbidden as error:
        logfire.warn(
            "Missing permission to post an expansion placeholder",
            message_id=message.id,
            channel_id=message.channel.id,
            error_type=type(error).__name__,
            _exc_info=error,
        )
        return None
    except HTTPException as error:
        # A reply to a message that is already gone comes back as 50035, not only as NotFound.
        if not isinstance(error, NotFound) and error.code != _UNSENDABLE_REPLY:
            raise
        logfire.info(
            "The message to expand is gone", message_id=message.id, channel_id=message.channel.id
        )
        return None
    return ExpansionPlaceholder(message=placeholder)
