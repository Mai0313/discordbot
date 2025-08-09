import pickle
from pathlib import Path
from functools import cached_property
from urllib.parse import urlparse

import dotenv
from pydantic import Field, computed_field
from rich.console import Console
from pydantic_settings import BaseSettings
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from google.auth.transport.requests import Request

from discordbot.typings.chat import LiveChatMessageItem, LiveChatMessageListResponse
from discordbot.typings.stream import VideoListResponse

dotenv.load_dotenv()
console = Console()


class YoutubeStream(BaseSettings):
    yt_api_key: str = Field(..., validation_alias="YOUTUBE_DATA_API_KEY", exclude=True)
    url: str

    @computed_field
    @property
    def video_id(self) -> str:
        parsed_url = urlparse(url=self.url)
        video_id = parsed_url.query.split("=")[-1]
        return video_id

    @computed_field
    @cached_property
    def youtube(self) -> Resource:
        credentials = self.get_credentials()
        youtube = build(
            serviceName="youtube",
            version="v3",
            developerKey=self.yt_api_key,
            credentials=credentials,
        )
        return youtube

    def get_credentials(self) -> Credentials:
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

    def get_chat_id(self) -> str:
        video_list = self.youtube.videos().list(part="liveStreamingDetails", id=self.video_id)
        response_dict = video_list.execute()
        response = VideoListResponse(**response_dict)
        chat_id = response.items[0].live_streaming_details.active_live_chat_id
        return chat_id

    def get_chat_messages(self) -> str:
        live_chat_id = self.get_chat_id()
        live_message = self.youtube.liveChatMessages()
        live_messages = live_message.list(
            liveChatId=live_chat_id, part="snippet,authorDetails", pageToken=None
        )
        response = LiveChatMessageListResponse(**live_messages.execute())

        chat_history = ""
        for item in response.items:
            name = item.author_details.display_name
            message = item.snippet.display_message
            chat_history += f"{name}: {message}\n"
        return chat_history

    def get_registered_accounts(self, target_word: str) -> list[str]:
        live_chat_id = self.get_chat_id()
        live_message = self.youtube.liveChatMessages()
        live_messages = live_message.list(
            liveChatId=live_chat_id, part="snippet,authorDetails", pageToken=None
        )
        response = LiveChatMessageListResponse(**live_messages.execute())

        registered_accounts = []
        for item in response.items:
            if target_word in item.snippet.display_message:
                registered_accounts.append(item.author_details.display_name)
        unique_accounts = list(set(registered_accounts))
        return unique_accounts
