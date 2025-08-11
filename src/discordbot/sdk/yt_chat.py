import pickle
from typing import TYPE_CHECKING
from pathlib import Path
from functools import cached_property
from urllib.parse import urlparse

import dotenv
from pydantic import Field, computed_field
from rich.console import Console
from chat_downloader import ChatDownloader
from pydantic_settings import BaseSettings
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from google.auth.transport.requests import Request

from discordbot.typings.models import (
    ChatItem,
    VideoListResponse,
    LiveChatMessageItem,
    LiveChatMessageListResponse,
)

if TYPE_CHECKING:
    from chat_downloader.sites.common import Chat

dotenv.load_dotenv()
console = Console()


class YoutubeStream(BaseSettings):
    yt_api_key: str = Field(..., validation_alias="YOUTUBE_DATA_API_KEY", exclude=True)
    url: str

    @classmethod
    def _get_credentials(cls) -> Credentials:
        token_file = Path("./data/token.pickle")
        if token_file.exists():
            with open(token_file, "rb") as token:
                credentials: Credentials = pickle.load(token)  # noqa: S301
        else:
            credentials = None
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                try:
                    credentials.refresh(Request())
                except Exception:
                    credentials = None

            else:
                console.print("🔐 需要重新授權...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    client_secrets_file="./data/client_secret.json",
                    scopes=["https://www.googleapis.com/auth/youtube.force-ssl"],
                )
                credentials = flow.run_local_server(port=8080, open_browser=True)
                console.print("✅ 授權完成")

            with open(token_file, "wb") as token:
                pickle.dump(credentials, token)
                console.print("💾 憑證已保存")
        return credentials

    @computed_field
    @cached_property
    def youtube(self) -> Resource:
        credentials = self._get_credentials()
        youtube = build(
            serviceName="youtube",
            version="v3",
            developerKey=self.yt_api_key,
            credentials=credentials,
        )
        return youtube

    def get_chat_id(self) -> str:
        parsed_url = urlparse(url=self.url)
        video_id = parsed_url.query.split("=")[-1]
        video_list = self.youtube.videos().list(part="liveStreamingDetails", id=video_id)
        response_dict = video_list.execute()
        response = VideoListResponse(**response_dict)
        chat_id = response.items[0].live_streaming_details.active_live_chat_id
        return chat_id

    def reply_to_chat(self, message: str) -> None:
        live_chat_id = self.get_chat_id()
        live_message = self.youtube.liveChatMessages()
        chat = live_message.insert(
            part="snippet",
            body={
                "snippet": {
                    "liveChatId": live_chat_id,
                    "type": "textMessageEvent",
                    "textMessageDetails": {"messageText": message},
                }
            },
        ).execute()
        chat = LiveChatMessageItem(**chat)
        console.print(f"📤 已發送: {message}")

    def get_chat_messages(self) -> LiveChatMessageListResponse:
        live_chat_id = self.get_chat_id()
        live_message = self.youtube.liveChatMessages()
        live_messages = live_message.list(
            liveChatId=live_chat_id, part="snippet,authorDetails", pageToken=None
        )
        response = LiveChatMessageListResponse(**live_messages.execute())
        return response

    def get_registered_accounts(self, target_word: str) -> list[str]:
        response = self.get_chat_messages()

        registered_accounts: list[str] = []
        for item in response.items:
            if target_word in item.snippet.display_message:
                registered_accounts.append(item.author_details.display_name)
        unique_accounts = list(set(registered_accounts))
        return unique_accounts

    def get_chat_messages_without_auth(self) -> list[ChatItem]:
        chat_downloader = ChatDownloader(
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/115.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
            }
        )
        chat: Chat = chat_downloader.get_chat(
            url=self.url, chat_type="live", inactivity_timeout=0.2, max_attempts=1
        )
        messages: list[ChatItem] = []
        for message in chat:
            message_item = ChatItem(**message)
            messages.append(message_item)
        return messages

    def get_registered_accounts_without_auth(self, target_word: str) -> list[str]:
        responses = self.get_chat_messages_without_auth()

        registered_accounts: list[str] = []
        for response in responses:
            if target_word in response.message:
                registered_accounts.append(response.author.name)
        unique_accounts = list(set(registered_accounts))
        return unique_accounts
