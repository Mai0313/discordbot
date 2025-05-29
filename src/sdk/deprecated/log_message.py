"""This module is deprecated and will be removed in the future.
Please use the new message logging system instead.
New message logging system: ./src/sdk/log_message.py
"""

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
            return f"DM_{author_name}"
        return f"channel_{self.message.channel.name}" or f"channel_{self.message.channel.id}"

    @computed_field
    @cached_property
    def channel_id_or_author_id(self) -> str:
        if isinstance(self.message.channel, nextcord.DMChannel):
            return f"{self.message.author.id}"
        return f"{self.message.channel.id}"

    @computed_field
    @property
    def attachment_path(self) -> Path:
        now = datetime.date.today().isoformat()
        attachment_path = Path("./data/attachments") / now / self.channel_name_or_author_name
        attachment_path.mkdir(parents=True, exist_ok=True)
        return attachment_path

    async def _get_filepath(self, filepath: Path) -> Path:
        # 如果檔名重複，則在檔名後加上數字
        # 例如：file_1.txt -> file_2.txt
        if filepath.exists():
            file_no = len(list(self.attachment_path.glob(f"{filepath.stem}_*{filepath.suffix}")))
            filepath = filepath.with_stem(f"{filepath.stem}_{file_no}")
        return filepath

    async def _save_attachments(self) -> list[str]:
        saved_paths = []
        for attachment in self.message.attachments:
            filepath = self.attachment_path / attachment.filename
            filepath = await self._get_filepath(filepath=filepath)
            await attachment.save(filepath)
            saved_paths.append(filepath.as_posix())
        return saved_paths

    async def _save_stickers(self) -> list[str]:
        saved_paths = []
        for sticker in self.message.stickers:
            filepath = self.attachment_path / sticker.name
            filepath = await self._get_filepath(filepath=filepath)
            try:
                await sticker.save(filepath)
                saved_paths.append(filepath.as_posix())
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
        attachment_paths = await self._save_attachments()
        sticker_paths = await self._save_stickers()
        await self._save_messages(attachment_paths=attachment_paths, sticker_paths=sticker_paths)
