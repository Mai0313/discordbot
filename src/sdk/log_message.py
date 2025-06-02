import pandas as pd
import logfire
import nextcord
from pydantic import Field, BaseModel, ConfigDict, computed_field
from sqlalchemy import create_engine

from src.types.database import DatabaseConfig


class MessageLogger(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    message: nextcord.Message
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    @computed_field
    @property
    def channel_name_or_author_name(self) -> str:
        if isinstance(self.message.channel, nextcord.DMChannel):
            author_name = self.message.author.nick or self.message.author.name
            return f"DM_{author_name}_{self.message.author.id}"
        channel_name = self.message.channel.name
        if channel_name:
            return f"channel_{channel_name}_{self.message.channel.id}"
        return f"channel_{self.message.channel.id}"

    @computed_field
    @property
    def channel_id_or_author_id(self) -> str:
        if isinstance(self.message.channel, nextcord.DMChannel):
            return f"{self.message.author.id}"
        return f"{self.message.channel.id}"

    async def _save_attachments(self) -> list[str]:
        attachment_urls = []
        for attachment in self.message.attachments:
            attachment_urls.append(attachment.url)
        return attachment_urls

    async def _save_stickers(self) -> list[str]:
        sticker_urls = []
        for sticker in self.message.stickers:
            sticker_urls.append(sticker.url)
        return sticker_urls

    async def _save_messages(self) -> None:
        attachment_paths = await self._save_attachments()
        sticker_paths = await self._save_stickers()
        data_dict = {
            "author": self.message.author.name,
            "author_id": self.channel_id_or_author_id,
            "content": self.message.content,
            "created_at": self.message.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "channel_name": self.channel_name_or_author_name,
            "channel_id": self.channel_id_or_author_id,
            "attachments": ";".join(attachment_paths),
            "stickers": ";".join(sticker_paths),
        }
        logfire.info("Message data", **data_dict)
        message_df = pd.DataFrame([data_dict])
        message_df = message_df.astype(str)

        # 連接到 SQLite 資料庫
        engine = create_engine(self.database.sqlite.sqlite_file_path)

        # 使用 pandas to_sql 寫入 SQLite 資料庫
        message_df.to_sql(
            name=f"{self.channel_name_or_author_name}", con=engine, if_exists="append", index=False
        )

    async def log(self) -> None:
        try:
            if self.message.author.bot:
                return
            await self._save_messages()
        except Exception:
            logfire.error("Failed to log message", _exc_info=True)
