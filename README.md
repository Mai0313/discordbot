<div align="center" markdown="1">

# AI-Powered Discord Bot 🤖

[![PyPI version](https://img.shields.io/pypi/v/swebenchv2.svg)](https://pypi.org/project/swebenchv2/)
[![python](https://img.shields.io/badge/-Python_%7C_3.11%7C_3.12%7C_3.13%7C_3.14-blue?logo=python&logoColor=white)](https://www.python.org/downloads/source/)
[![uv](https://img.shields.io/badge/-uv_dependency_management-2C5F2D?logo=python&logoColor=white)](https://docs.astral.sh/uv/)
[![nextcord](https://img.shields.io/badge/-Nextcord-5865F2?logo=discord&logoColor=white)](https://github.com/nextcord/nextcord)
[![openai](https://img.shields.io/badge/-OpenAI-412991?logo=openai&logoColor=white)](https://openai.com)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://docs.pydantic.dev/latest/contributing/#badges)
[![tests](https://github.com/Mai0313/discordbot/actions/workflows/test.yml/badge.svg)](https://github.com/Mai0313/discordbot/actions/workflows/test.yml)
[![code-quality](https://github.com/Mai0313/discordbot/actions/workflows/code-quality-check.yml/badge.svg)](https://github.com/Mai0313/discordbot/actions/workflows/code-quality-check.yml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/Mai0313/discordbot/tree/main?tab=License-1-ov-file)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Mai0313/discordbot/pulls)
[![contributors](https://img.shields.io/github/contributors/Mai0313/discordbot.svg)](https://github.com/Mai0313/discordbot/graphs/contributors)

**English** | [**繁體中文**](./README.zh-TW.md) | [**简体中文**](./README.zh-CN.md)

</div>

A comprehensive Discord Bot built with **nextcord** that provides AI-powered interactions, content processing, and utility features. Features multi-language support and integrated web search capabilities. 🚀⚡🔥

_Suggestions and contributions are always welcome!_

## ✨ Features

### 🤖 AI-Powered Interactions

- **Text Generation**: Support for multiple AI models (OpenAI GPT-4o — default, GPT-5-mini, GPT-5-nano, Claude-3.5-Haiku) with **default streaming** (updates about every 10 characters) and integrated web search
- **Per-User Memory**: Conversation memory is tracked per user; `/clear_memory` clears your memory
- **Image Processing**: Vision model support with automatic image conversion
- **Smart Web Access**: LLM can automatically search the web when needed to provide up-to-date information

### 📊 Content Processing

- **Message Summarization**: Smart channel conversation summaries with user filtering (5, 10, 20, 50 messages)

- **Video Downloading**: Multi-platform support (YouTube, TikTok, Instagram, X, Facebook) with quality options

    - Bilibili compatibility improvements: proper Referer header, safer format fallbacks, and robust error handling
    - Site-specific headers: Referer is applied only for Bilibili to avoid breaking Facebook links
    - Facebook share links (e.g., `facebook.com/share/r/...`) are automatically expanded before downloading, so you can paste whatever the app gives you

- **MapleStory Database**: Search monsters and items with comprehensive drop information

### 🌍 Multi-Language Support

- Traditional Chinese (繁體中文)
- Simplified Chinese (简体中文)
- Japanese (日本語)
- English

### 🔧 Technical Features

- **Main Bot Implementation**: The core bot class `DiscordBot` is implemented in `src/discordbot/cli.py`, extending `nextcord.ext.commands.Bot` with comprehensive initialization, cog loading, and event handling
- Modular Cog-based architecture
- Async/await patterns with nextcord
- Pydantic-based configuration management
- Comprehensive error handling and logging
- Docker support with development containers

## 🎯 Core Commands

| Command           | Description                       | Features                                                                                                                                                                            |
| ----------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/oai`            | Generate AI text response         | Multi-model (GPT-4o default, GPT-5 mini/nano, Claude 3.5 Haiku), **default streaming** (updates ~every 10 characters), optional image input, integrated web search, per-user memory |
| `/clear_memory`   | Clear conversation memory         | Resets your per-user memory used to continue conversations                                                                                                                          |
| `/sum`            | Interactive message summarization | User filter, 5/10/20/50 messages                                                                                                                                                    |
| `/download_video` | Multi-platform video downloader   | Best/High/Medium/Low quality, auto low-quality fallback if >25MB                                                                                                                    |
| `/maple_monster`  | Search MapleStory monster drops   | Detailed stats, images, maps                                                                                                                                                        |
| `/maple_item`     | Search MapleStory item sources    | Drop source mapping                                                                                                                                                                 |
| `/maple_stats`    | MapleStory DB statistics          | Totals, level distribution, popular items                                                                                                                                           |

| `/graph` | Generate images (placeholder) | Framework ready for implementation |
| `/ping` | Bot performance testing | Latency measurement |

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Discord Bot Token
- OpenAI API Key

### Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/Mai0313/discordbot.git
    cd discordbot
    ```

2. **Install dependencies using uv**

    ```bash
    # Install uv if you haven't already
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Install dependencies
    uv sync
    ```

3. **Set up environment variables**

    ```bash
    cp .env.example .env
    # Edit .env with your API keys and configuration
    ```

4. **Run the bot**

    ```bash
    # Recommended (via entry point)
    uv run discordbot

    # Or
    uv run python -m discordbot.cli
    ```

### Docker Setup

```bash
# Using Docker Compose
docker-compose up -d

# Or build manually
docker build -t discordbot .
docker run -d discordbot
```

Note: The Docker image installs `ffmpeg` so yt-dlp can merge separate audio/video streams.

### Optional: Update MapleStory database

```bash
# Install Playwright Chromium (first time only)
uv run playwright install chromium

# Fetch latest MapleStory monsters/items to ./data/monsters.json
uv run update
```

## ⚙️ Configuration

### Required Environment Variables

```env
# Discord Configuration
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_TEST_SERVER_ID=your_test_server_id  # Optional

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# Azure OpenAI (if using Azure)
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com


```

### Optional Environment Variables

```env
# OpenAI (Azure) API version
OPENAI_API_VERSION=2025-04-01-preview

# Local message logging (SQLite)
SQLITE_FILE_PATH=sqlite:///data/messages.db

# Optional external services
POSTGRES_URL=postgresql://postgres:postgres@postgres:5432/postgres
REDIS_URL=redis://redis:6379/0


```

## 📁 Project Structure

```
src/discordbot/
├── cli.py              # Main bot entry point
├── cogs/               # Command modules
│   ├── gen_reply.py    # AI text generation (/oai)
│   ├── summary.py      # Message summarization (/sum)
│   ├── video.py        # Video downloading (/download_video)
│   ├── maplestory.py   # MapleStory database queries
│   ├── gen_image.py    # Image generation (placeholder)
│   └── template.py     # Utilities & /ping
├── sdk/                # Core business logic
│   ├── llm.py          # LLM integration (OpenAI/Azure)
│   ├── log_message.py  # Message logging to SQLite
│   └── yt_chat.py      # YouTube chat helper
├── typings/            # Configuration models
│   ├── config.py       # Discord config
│   └── database.py     # DB configs (SQLite/Postgres/Redis)
└── utils/              # Utility functions
    └── downloader.py   # yt-dlp wrapper
data/
├── monsters.json       # MapleStory monster and drop database
└── downloads/          # Video download storage
```

## 🔍 Key Features Deep Dive

### Multi-Modal AI Support

- **Real-time Streaming**: Text responses stream in real-time, updating every 10 characters for immediate feedback
- Text and image input processing
- Automatic image-to-base64 conversion
- Model-specific constraint handling
- Integrated web search for real-time information

### Video Download Engine

- Support for 10+ platforms
- Quality selection (4K to audio-only)
- File size validation for Discord limits
- Progress tracking and error handling
- Retry mechanism powered by Tenacity: 5 attempts with 1s wait between retries (hard-coded)

#### Bilibili notes

- Referer is applied only for Bilibili; other sites (e.g., Facebook) use minimal headers.
- Ensure your `yt-dlp` is up-to-date.
- `ffmpeg` is required for merging separate video/audio streams. Install it via your package manager.
- The downloader sends a `Referer: https://www.bilibili.com` header and uses safe fallbacks like `bestvideo*+bestaudio/best`.
- If you still see "Requested format is not available", try a lower quality (Medium/Low). Some videos only expose DASH profiles or region/age-limited formats.
- For geo/age/login-gated videos, you may need to provide cookies to yt-dlp (not wired in by default).

#### Facebook notes

- We do not force a Referer for Facebook; minimal headers are used to avoid extractor conflicts.
- Short `facebook.com/share/...` links are auto-expanded and converted to the right reel/watch URL, so you can download directly from the share link.
- Keep `yt-dlp` up-to-date and ensure `ffmpeg` is installed.
- If downloads fail, try a lower quality; private/login/region-limited links may require cookies passed to yt-dlp.

### MapleStory Database

- Comprehensive monster and item database (192+ monsters)
- Interactive search with fuzzy matching
- Multi-language support (Traditional Chinese, Simplified Chinese, Japanese, English)
- Detailed monster statistics and drop information
- Item source tracking with visual displays
- Cached search results for optimal performance

# 🔒 Privacy & Data

This Discord bot complies with Discord’s Terms of Service and Developer Policy.

## 📦 Data Collection and Usage

- **Local Message Logging**: By default, messages in channels where the bot is present are logged to a local SQLite database at `./data/messages.db` (author, content, timestamps, attachments/stickers). This remains on your server and is not shared externally.
- **No Third-Party Sharing**: Aside from calling trusted APIs (e.g., OpenAI) to fulfill requests, data is not shared with third parties.
- **Opt-out**: Server owners can disable logging by removing the logging calls in `src/discordbot/cli.py` or adapting `src/discordbot/sdk/log_message.py`.

## ⚙️ Bot Permissions and Intents

This bot uses certain Discord intents solely to provide its core features:

- **Message Content Intent**: Required for slash-command context, limited keyword handling, and optional local logging as described above.
- **Slash Commands**: Used for interactive and explicit command triggers.
- **Embed Links / File Attachments**: Used to display structured output and allow interaction with visual content.
- **Presence Intent (if applicable)**: May be used to improve responsiveness based on online status; not stored.

Users may opt out of interactions via commands like `!optout` (planned) or by muting the bot.

## 🔐 Data Security

- All API requests are made via secure HTTPS connections.
- No data is persisted to any external service. If local message logging is enabled, messages are stored on your disk in SQLite (`./data/messages.db`) and never leave your server. You can disable logging by removing the logging calls in `src/discordbot/cli.py` or adapting `src/discordbot/sdk/log_message.py`.
- No long-term analytics based on message or user content are performed.

## 📬 Contact and Compliance

If you have privacy concerns or questions about this policy, feel free to:

- Submit an issue via GitHub
- Contact the developer through the repository's listed channels

This bot is designed using **privacy-by-design principles** with a strict minimal-data-handling approach to protect all users.

_Last updated: 2025/08/14_

## 🧪 Testing

- Install test dependencies and run tests:

```bash
uv sync --group test
uv run pytest -q
```

- Coverage outputs:

    - JUnit XML: `./.github/reports/pytest.xml`
    - Coverage XML: `./.github/reports/coverage.xml`
    - Coverage HTML: `./.github/coverage_html_report/index.html`

- Cog unit tests included:

    - TemplateCogs: message reaction and `/ping` embed
    - MessageFetcher: `_format_messages()` and `do_summarize()` (LLM mocked)
    - ReplyGeneratorCogs: `_get_attachment_list()` and `/clear_memory`
    - ImageGeneratorCogs: `/graph` placeholder flow
    - VideoCogs: `/download_video` happy path (downloader mocked)

## Contributors

[![Contributors](https://contrib.rocks/image?repo=Mai0313/discordbot)](https://github.com/Mai0313/discordbot/graphs/contributors)

Made with [contrib.rocks](https://contrib.rocks)

## For More info, check the [Docs](https://mai0313.github.io/discordbot/)
