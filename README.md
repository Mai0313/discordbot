<center>

# AI-Powered Discord Bot 🤖

**English** | [**繁體中文**](./README_cn.md)

[![python](https://img.shields.io/badge/-Python_3.10_%7C_3.11_%7C_3.12-blue?logo=python&logoColor=white)](https://python.org)
[![nextcord](https://img.shields.io/badge/-Nextcord-5865F2?logo=discord&logoColor=white)](https://github.com/nextcord/nextcord)
[![openai](https://img.shields.io/badge/-OpenAI-412991?logo=openai&logoColor=white)](https://openai.com)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![tests](https://github.com/Mai0313/discordbot/actions/workflows/test.yml/badge.svg)](https://github.com/Mai0313/discordbot/actions/workflows/test.yml)
[![code-quality](https://github.com/Mai0313/discordbot/actions/workflows/code-quality-check.yml/badge.svg)](https://github.com/Mai0313/discordbot/actions/workflows/code-quality-check.yml)
[![codecov](https://codecov.io/gh/Mai0313/discordbot/branch/master/graph/badge.svg)](https://codecov.io/gh/Mai0313/discordbot)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/Mai0313/discordbot/tree/master?tab=License-1-ov-file)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Mai0313/discordbot/pulls)
[![contributors](https://img.shields.io/github/contributors/Mai0313/discordbot.svg)](https://github.com/Mai0313/discordbot/graphs/contributors)

</center>

A comprehensive Discord Bot built with **nextcord** that provides AI-powered interactions, content processing, and utility features. Features multi-language support and integrated web search capabilities. 🚀⚡🔥

_Suggestions and contributions are always welcome!_

## ✨ Features

### 🤖 AI-Powered Interactions

- **Text Generation**: Support for multiple AI models (OpenAI GPT-4o — default, GPT-5-mini, GPT-5-nano, Claude-3.5-Haiku) with **default streaming** (updates about every 10 characters) and integrated web search
- **Per-User Memory**: Conversation memory is tracked per user; `/clear_memory` clears your memory
- **Image Processing**: Vision model support with automatic image conversion
- **Image Generation**: Integrated via Responses API tools with **progressive streaming rendering**; generated images are delivered as Discord attachments with rich embeds showing partial progress
- **Smart Web Access**: LLM can automatically search the web when needed to provide up-to-date information

### 📊 Content Processing

- **Message Summarization**: Smart channel conversation summaries with user filtering (5, 10, 20, 50 messages)
- **Video Downloading**: Multi-platform support (YouTube, TikTok, Instagram, X, Facebook) with quality options
    - Bilibili compatibility improvements: proper Referer header, safer format fallbacks, and robust error handling
    - Site-specific headers: Referer is applied only for Bilibili to avoid breaking Facebook links
- **MapleStory Database**: Search monsters and items with comprehensive drop information
- **Auction System**: Complete auction platform with bidding functionality and multi-currency support (楓幣/雪花/台幣)
- **Lottery System**: Multi-platform lottery with Discord button-based join or YouTube chat integration (no reactions); supports per-draw winner count and recreate. Winners are automatically excluded from re-joining the same lottery (until you use "Recreate"). Uses "discord" naming (no legacy "reaction" terminology). The participant list is displayed as a single unified field.
    - Implementation note: Join/Cancel buttons are implemented as subclasses of `nextcord.ui.Button` (`JoinLotteryButton`, `CancelJoinLotteryButton`) for better maintainability and potential persistent view support. Winner checks and duplicate prevention are centralized in the core add/remove functions to keep UI flows simple.
- **Image Generation**: Integrated in `/oai` via the image_generation tool (Responses API). The standalone `/graph` command remains a placeholder for future expansion.

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

| Command           | Description                       | Features                                                                                                                                                                                                                                                                                       |
| ----------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/oai`            | Generate AI text response         | Multi-model (GPT-4o default, GPT-5 mini/nano, Claude 3.5 Haiku), **default streaming** (updates ~every 10 characters), optional image input, integrated web search and **progressive image generation** (Responses API tools), per-user memory                                                 |
| `/clear_memory`   | Clear conversation memory         | Resets your per-user memory used to continue conversations                                                                                                                                                                                                                                     |
| `/sum`            | Interactive message summarization | User filter, 5/10/20/50 messages                                                                                                                                                                                                                                                               |
| `/download_video` | Multi-platform video downloader   | Best/High/Medium/Low quality, auto low-quality fallback if >25MB                                                                                                                                                                                                                               |
| `/maple_monster`  | Search MapleStory monster drops   | Detailed stats, images, maps                                                                                                                                                                                                                                                                   |
| `/maple_item`     | Search MapleStory item sources    | Drop source mapping                                                                                                                                                                                                                                                                            |
| `/maple_stats`    | MapleStory DB statistics          | Totals, level distribution, popular items                                                                                                                                                                                                                                                      |
| `/auction_create` | Create new item auction           | Currency selection (楓幣/雪花/台幣), float prices                                                                                                                                                                                                                                              |
| `/auction_list`   | Browse active auctions            | Dropdown selection, preview                                                                                                                                                                                                                                                                    |
| `/auction_info`   | View auction details              | Current bid, end time, history button                                                                                                                                                                                                                                                          |
| `/auction_my`     | View personal auctions            | Created & leading                                                                                                                                                                                                                                                                              |
| `/lottery`        | Lottery main                      | Dropdown to choose method; Buttons: 🎉 Join, 🚫 Cancel (Discord mode), ✅ Start, 📊 Status (ephemeral), 🔄 Recreate, 🔁 Update Participants (YouTube/host-only); creation message auto-updates unified participant list; winners-per-draw; winners are excluded from re-joining until Recreate |
| `/graph`          | Generate images (placeholder)     | Framework ready for implementation                                                                                                                                                                                                                                                             |
| `/ping`           | Bot performance testing           | Latency measurement                                                                                                                                                                                                                                                                            |

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

# YouTube Data API key (required for YouTube lottery mode)
YOUTUBE_DATA_API_KEY=your_youtube_data_api_key
```

### YouTube Lottery mode setup (optional)

- Provide a Google OAuth client file at `./data/client_secret.json` (Desktop app credentials).
- Ensure `YOUTUBE_DATA_API_KEY` is set.
- On first use of YouTube mode, the bot opens a browser on port 8080 to complete OAuth and saves a token to `./data/token.pickle`.

## 📁 Project Structure

```
src/discordbot/
├── cli.py              # Main bot entry point
├── cogs/               # Command modules
│   ├── gen_reply.py    # AI text generation (/oai)
│   ├── summary.py      # Message summarization (/sum)
│   ├── video.py        # Video downloading (/download_video)
│   ├── maplestory.py   # MapleStory database queries
│   ├── auction.py      # Auction system with bidding
│   ├── lottery.py      # Multi-platform lottery system
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
├── auctions.db         # SQLite database for auction system
└── downloads/          # Video download storage
```

## 🔍 Key Features Deep Dive

### Multi-Modal AI Support

- **Real-time Streaming**: Text responses stream in real-time, updating every 10 characters for immediate feedback
- **Progressive Image Rendering**: Image generation shows partial progress updates for a smooth visual experience
- Text and image input processing
- Automatic image-to-base64 conversion
- Image generation via Responses API tools; outputs are sent as Discord file attachments with embeds for a polished preview
- Model-specific constraint handling
- Integrated web search for real-time information

### Video Download Engine

- Support for 10+ platforms
- Quality selection (4K to audio-only)
- File size validation for Discord limits
- Progress tracking and error handling

#### Bilibili notes

- Referer is applied only for Bilibili; other sites (e.g., Facebook) use minimal headers.
- Ensure your `yt-dlp` is up-to-date.
- `ffmpeg` is required for merging separate video/audio streams. Install it via your package manager.
- The downloader sends a `Referer: https://www.bilibili.com` header and uses safe fallbacks like `bestvideo*+bestaudio/best`.
- If you still see "Requested format is not available", try a lower quality (Medium/Low). Some videos only expose DASH profiles or region/age-limited formats.
- For geo/age/login-gated videos, you may need to provide cookies to yt-dlp (not wired in by default).

#### Facebook notes

- We do not force a Referer for Facebook; minimal headers are used to avoid extractor conflicts.
- Keep `yt-dlp` up-to-date and ensure `ffmpeg` is installed.
- If downloads fail, try a lower quality; private/login/region-limited links may require cookies passed to yt-dlp.

### MapleStory Database & Auction System

- Comprehensive monster and item database (192+ monsters)
- Interactive search with fuzzy matching
- Multi-language support (Traditional Chinese, Simplified Chinese, Japanese, English)
- Detailed monster statistics and drop information
- Item source tracking with visual displays
- Cached search results for optimal performance
- **Separate Auction System Module**:
    - Create item auctions with customizable duration, bidding increments, and currency type selection (楓幣/雪花/台幣)
    - Multi-currency support with "楓幣" (Mesos) as default, "雪花" (Snowflake), and "台幣" (Taiwan Dollar) as alternatives
    - Real-time bidding with interactive UI (💰 Bid, 📊 View Records, 🔄 Refresh)
    - Personal auction management and bid tracking with currency type display
    - Security features preventing self-bidding and duplicate bids
    - SQLite database storage with ACID compliance and backward compatibility

### Lottery System

- **Registration Modes**: Discord button-based join or YouTube chat keyword-based participation
- **Button Controls**: 🎉 Join, 🚫 Cancel, ✅ Start (host-only), 📊 Status (ephemeral), 🔄 Recreate (host-only), 🔁 Update Participants (YouTube/host-only)
- **Live Status**: Pressing 📊 returns an ephemeral status only visible to the requester
- **Auto-Updating Message**: The creation message automatically updates to list participant names as users join/cancel (single unified list)
- **Message Binding**: Button interactions are bound to the creation message so the bot can reliably identify the correct lottery (no emoji reactions involved)
- **YouTube Mode Participant Fetch**: In YouTube mode, participants can be fetched any time by the host using the new 🔁 Update Participants button; the bot also fetches right before drawing when the host presses ✅ Start
- **Winner Exclusion**: Once drawn, winners cannot be re-added to the same lottery (prevents duplicate wins across multiple draws). Use 🔄 Recreate to start a fresh round if you want everyone eligible again.
- **Discord Mode Buttons**: 🎉 Join and 🚫 Cancel buttons are shown only for Discord mode
- **Winners Per Draw**: Configure how many winners to draw per trigger in the creation modal
- **Recreate Functionality (🔄)**: Recreate a fresh lottery with identical settings and restored participants (including prior winners)
- **Security Features**: Creator-only controls, cryptographically secure random selection, participant validation, and automatic winner removal
- **Single-Platform Focus**: Each lottery uses only one registration method to ensure fairness and prevent confusion
- **In-Memory Storage**: Lightweight global variables with defaultdict optimization for runtime data (resets on bot restart for fresh starts)

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

## Contributors

[![Contributors](https://contrib.rocks/image?repo=Mai0313/discordbot)](https://github.com/Mai0313/discordbot/graphs/contributors)

Made with [contrib.rocks](https://contrib.rocks)

## For More info, check the [Docs](https://mai0313.github.io/discordbot/)
