from pathlib import Path
import datetime
from functools import cached_property

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
    @cached_property
    def channel_name_or_author_name(self) -> str:
        if isinstance(self.message.channel, nextcord.DMChannel):
            author_name = self.message.author.nick or self.message.author.name
            return author_name
        return self.message.channel.name or f"{self.message.channel.id}"

    @computed_field
    @cached_property
    def channel_id_or_author_id(self) -> str:
        if isinstance(self.message.channel, nextcord.DMChannel):
            return f"{self.message.author.id}"
        return f"{self.message.channel.id}"

    async def _save_attachments(self, base_dir: Path) -> list[str]:
        saved_paths = []
        for attachment in self.message.attachments:
            filepath = base_dir / attachment.filename
            base_dir.mkdir(parents=True, exist_ok=True)
            await attachment.save(filepath)
            saved_paths.append(str(filepath))
        return saved_paths

    async def _save_stickers(self, base_dir: Path) -> list[str]:
        saved_paths = []
        for sticker in self.message.stickers:
            filepath = base_dir / f"sticker_{sticker.id}.png"
            try:
                base_dir.mkdir(parents=True, exist_ok=True)
                await sticker.save(filepath)
                saved_paths.append(str(filepath))
            except nextcord.NotFound:
                logfire.warn("Sticker is not found", sticker_id=sticker.id)
        return saved_paths

    async def _save_messages(self, attachment_paths: list[str], sticker_paths: list[str]) -> None:
        data_dict = {
            "author": self.message.author.name,
            "author_id": self.message.author.id,
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

        # 確保資料庫目錄存在
        Path(self.database.sqlite.sqlite_file_path).parent.mkdir(parents=True, exist_ok=True)

        # 連接到 SQLite 資料庫
        engine = create_engine(f"sqlite:///{self.database.sqlite.sqlite_file_path}")

        # 使用 pandas to_sql 寫入 SQLite 資料庫
        message_df.to_sql(
            name=f"{self.channel_name_or_author_name}", con=engine, if_exists="append", index=False
        )

    async def log(self) -> None:
        if self.message.author.bot:
            return
        today = datetime.date.today().isoformat()
        base_dir = Path("data") / today / self.channel_name_or_author_name
        attachment_paths = await self._save_attachments(base_dir=base_dir)
        sticker_paths = await self._save_stickers(base_dir=base_dir)
        await self._save_messages(attachment_paths=attachment_paths, sticker_paths=sticker_paths)
