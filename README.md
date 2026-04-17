<div align="center" markdown="1">

# AI-Powered Discord Bot

[![PyPI version](https://img.shields.io/pypi/v/swebenchv2.svg)](https://pypi.org/project/swebenchv2/)
[![python](https://img.shields.io/badge/-Python_%7C_3.12%7C_3.13%7C_3.14-blue?logo=python&logoColor=white)](https://www.python.org/downloads/source/)
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

A feature-rich Discord bot with AI-powered conversations, image and video generation, content parsing, multi-platform video downloading, and a MapleStory game database. Supports multiple languages.

## Features

### AI Chat

Mention the bot (`@bot`) or send a direct message to start a conversation. Powered by Google Gemini, it supports:

- **Text conversations** with real-time streaming responses
- **Media understanding** — attach images or short videos and ask questions about them
- **Image generation & editing** — ask the bot to draw, create, or edit images (attach an image to modify it)
- **Video generation** — ask the bot to generate short videos (cooldown between requests)
- **Chat summarization** — ask the bot to recap the recent conversation
- **Web search** — the bot automatically searches the web when it needs up-to-date information
- **User tagging** — ask the bot to notify or address other participants from the recent conversation (e.g. "let @alice know I'll be late") — it can mention anyone who appeared in the recent chat history
- **Progress reactions** — emoji reactions on your message show real-time processing status (🤔 → 🔀 → 🎨/🎬/📖/❓ → 🆗, or ❌ on error)
- **Reply footer** — each AI response shows the model name, input/output token counts, and estimated USD cost

### Threads Parsing

Paste a Threads.net link and the bot automatically expands it — displaying the post text, images, engagement stats, and downloading any attached videos.

### Video Downloading

Use `/download_video` to download videos from multiple platforms:

- YouTube, TikTok, Instagram, X (Twitter), Facebook, Bilibili
- Quality options: Best, High (1080p), Medium (720p), Low (480p)
- Automatic low-quality fallback if the file exceeds Discord's 25 MB limit
- Facebook share links (`facebook.com/share/r/...`) are automatically expanded

### MapleStory Artale Database

- `/maple_monster` — Search monsters by name, view stats, spawn maps, and drops
- `/maple_equip` — Search equipment by name, view stats and acquisition sources
- `/maple_scroll` — Search scrolls by name and stat bonuses
- `/maple_npc` — Search NPCs by name and location
- `/maple_quest` — Search quests by name, level range, and frequency
- `/maple_map` — Search maps by name, region, and spawning monsters
- `/maple_item` — Search items and find which monsters drop them
- `/maple_stats` — View database statistics
- Interactive search with fuzzy matching and multi-language results

### Multi-Language Support

Commands and responses are available in English, Traditional Chinese, Simplified Chinese, and Japanese.

## Commands

| Command                           | Description                                                           |
| --------------------------------- | --------------------------------------------------------------------- |
| `@bot <message>`                  | Chat with AI (text, images, generation, summarization, web search)    |
| _Threads link_                    | Automatically expands Threads.net posts with media                    |
| `/download_video <url> [quality]` | Download video from YouTube, TikTok, Instagram, X, Facebook, Bilibili |
| `/maple_monster <name>`           | Search MapleStory monsters and drops                                  |
| `/maple_equip <name>`             | Search MapleStory equipment                                           |
| `/maple_scroll <name>`            | Search MapleStory scrolls                                             |
| `/maple_npc <name>`               | Search MapleStory NPCs                                                |
| `/maple_quest <name>`             | Search MapleStory quests                                              |
| `/maple_map <name>`               | Search MapleStory maps                                                |
| `/maple_item <name>`              | Search MapleStory item sources                                        |
| `/maple_stats`                    | View MapleStory database statistics                                   |
| `/help`                           | Show bot usage guide                                                  |
| `/ping`                           | Check bot latency                                                     |

## Self-Hosting

### Prerequisites

- Python 3.12+
- A Discord bot token ([Developer Portal](https://discord.com/developers/applications))
- An OpenAI-compatible API key (e.g. Google Gemini via OpenAI-compatible endpoint)

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/Mai0313/discordbot.git
cd discordbot
cp .env.example .env
# Edit .env with your tokens and API keys
docker-compose up -d
```

The Docker image includes `ffmpeg` for video/audio stream merging.

### Option 2: Local Installation

```bash
git clone https://github.com/Mai0313/discordbot.git
cd discordbot

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your tokens and API keys

# Run the bot
uv run discordbot
```

### Optional: Update MapleStory Artale Database

```bash
uv run update
```

## Configuration

Create a `.env` file (or copy from `.env.example`):

```env
# Required
DISCORD_BOT_TOKEN=your_bot_token
API_KEY=your_api_key
BASE_URL=https://api.openai.com/v1   # or any OpenAI-compatible endpoint

# Optional
DISCORD_TEST_SERVER_ID=your_test_server_id
SQLITE_FILE_PATH=sqlite:///data/messages.db
POSTGRES_URL=postgresql://user:pass@host/db
REDIS_URL=redis://host:6379/0
```

## Platform-Specific Notes

### Bilibili

- `ffmpeg` is required for merging separate video/audio streams (included in Docker image).
- If "Requested format is not available" appears, try a lower quality setting.
- Region/age-restricted videos may require cookies (not configured by default).

### Facebook

- Share links (`facebook.com/share/...`) are automatically expanded before downloading.
- Keep `yt-dlp` up to date for best compatibility.

## Privacy & Data

This bot complies with Discord's Terms of Service and Developer Policy.

- **Message Logging**: Messages in channels where the bot is present are logged locally to SQLite. Data stays on your server and is never shared externally.
- **API Calls**: Text, images, and sender identity (display name, username, and Discord user ID of participants in the active chat context) are sent to the configured LLM API only when the bot is mentioned. User IDs are included so the bot can tag other participants when asked. No data is shared with other third parties.
- **Permissions**: The bot requires Message Content intent for mention-based chat and optional local logging. Slash commands and embed/attachment permissions are used for interactive features.
- **Opt-out**: Server owners can disable message logging by adjusting the bot configuration.

## Troubleshooting

**Bot doesn't respond to commands?**
Check bot permissions and ensure the `applications.commands` scope is enabled.

**Video download fails?**
Make sure `yt-dlp` and `ffmpeg` are up to date. Try a lower quality setting.

**API errors?**
Verify your API key and check that the endpoint URL is correct.

---

Want to contribute? See [CONTRIBUTING.md](./CONTRIBUTING.md).

[![Contributors](https://contrib.rocks/image?repo=Mai0313/discordbot)](https://github.com/Mai0313/discordbot/graphs/contributors)

[Documentation](https://mai0313.github.io/discordbot/) | [Report a Bug](https://github.com/Mai0313/discordbot/issues) | [Discussions](https://github.com/Mai0313/discordbot/discussions)
