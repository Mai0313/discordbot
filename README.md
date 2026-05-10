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

A feature-rich Discord bot with AI-powered conversations, image and video generation, content parsing, multi-platform video downloading, a points economy with casino mini-games, and a MapleStory game database. Supports multiple languages.

## Features

### AI Chat

Mention the bot (`@bot`) or send a direct message to start a conversation. The AI backend is any OpenAI-compatible endpoint (typically a [LiteLLM](https://github.com/BerriAI/litellm) proxy fronting OpenAI, Google Gemini, Anthropic Claude, etc.), and the bot routes each task to a different model — a fast model for intent routing and image captions, a slow reasoning model for replies and summaries, a dedicated image model for generation/editing, and a video model for short clips. Supported features:

- **Text conversations** powered by the OpenAI Responses API with real-time streaming
- **Media understanding** — attach images, stickers, or supported files such as PDFs, text, and JSON, then ask the bot about them; it also reads images embedded in messages or quoted replies (e.g. a parsed Threads post)
- **Image generation & editing** — ask the bot to draw, create, or edit images (attach an image to modify it)
- **Video generation** — ask the bot to generate short videos (cooldown between requests)
- **Chat summarization** — ask the bot to recap the recent conversation
- **Web search & URL reading** — the bot automatically uses model-specific tools (Gemini `googleSearch` + `urlContext`, Claude `web_search` + `web_fetch`, or OpenAI `web_search`) for up-to-date context
- **User tagging** — ask the bot to notify or address other participants from the recent conversation (e.g. "let @alice know I'll be late") — it can mention anyone who appeared in the recent chat history
- **Progress reactions** — emoji reactions on your message show real-time processing status (🤔 → 🔀 → 🎨/🎬/📖/❓ → 🆗, plus 🌐 if the model used web search, or ❌ on error)
- **Reply footer** — each AI response ends with a Discord-quoted line showing the model name, input/output token counts, estimated USD cost (computed from the upstream LiteLLM price table, fetched on demand and cached locally), and how many points the user just earned this turn
- **Auto-unmute** — if a moderator times the bot out, it lifts its own timeout, identifies the moderator from the audit log, and posts a single AI reply in the most recently active channel

### Threads Parsing

Paste a Threads.net link and the bot automatically expands it — displaying the post text, images, engagement stats, and downloading any attached videos. If the link points to a reply, the bot also walks the reply chain and shows the original post plus intermediate replies in the same message, with a grey-scale gradient stripe (light → dark) so each layer is easy to tell apart. Only the post the user pasted has its videos attached; ancestor videos are surfaced as an inline link hint to avoid mixing files across layers.

### Video Downloading

Use `/download_video` to download videos from multiple platforms:

- YouTube, TikTok, Instagram, X (Twitter), Facebook, Bilibili
- Quality options: Best, High (1080p), Medium (720p), Low (480p)
- Automatic low-quality fallback if the file exceeds Discord's 25 MB limit
- Facebook share links (`facebook.com/share/r/...`) are automatically expanded

### Points & Casino Games

The bot keeps a **persistent, cross-server point balance** for every Discord account in a local SQLite file (`data/economy.db`). The same balance follows the user into any guild the bot is in.

**Earning points:** every streaming AI reply awards points equal to its `total_tokens` (input + output), shown in the reply footer. Chatting with the bot is currently the only way to earn — Threads parsing and `/download_video` deliberately don't pay.

**Spending points:** the casino games take a bet up-front and pay out when the round resolves. The dealer is an AI that taunts the bet and reacts to the result with one short line. The dealer's display name in the embed (and in message history seen by `gen_reply`) is the bot's own Discord display name, so it shows up as a familiar identity rather than a generic "dealer" label.

| Slash command      | Game                                                                                                                  |
| ------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `/dice <bet>`      | Three dice vs three dice; bigger total wins (push refunds the bet).                                                   |
| `/blackjack <bet>` | Standard 21 with Hit / Stand buttons. Natural Blackjack pays 1.5×; the dealer drips a sarcastic hint after every hit. |

**Blackjack early settlement:** `Blackjack` means the first two cards are an ace plus a 10-value card. A player natural Blackjack wins immediately and pays 1.5×; a dealer natural Blackjack settles immediately unless the player also has Blackjack, in which case the hand pushes. A regular 21 reached with more cards is not a natural Blackjack and does not skip Hit / Stand.

**Managing points:**

- `/balance` — show your current balance.
- `/leaderboard` — global Top 10 across every server the bot is in (the bot's own house-ledger row is excluded).
- `/give <member> <amount>` — transfer points to another member (no self-transfer, no bots).
- `/house` — show the dealer's accumulated win/loss across `/dice` and `/blackjack`. Because the bot effectively has unlimited funds, the dealer's running balance can go negative when the casino is losing overall.

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

Slash command names, descriptions, and the `/help` guide are localized for English, Traditional Chinese (`zh-TW`), and Japanese (`ja`). AI chat replies follow whichever language the user writes in.

## Commands

| Command                           | Description                                                             |
| --------------------------------- | ----------------------------------------------------------------------- |
| `@bot <message>`                  | Chat with AI (text, media/files, generation, summarization, web search) |
| _Threads link_                    | Automatically expands Threads.net posts with media                      |
| `/download_video <url> [quality]` | Download video from YouTube, TikTok, Instagram, X, Facebook, Bilibili   |
| `/balance`                        | Show your current point balance (cross-server)                          |
| `/leaderboard`                    | Global Top 10 point holders                                             |
| `/give <member> <amount>`         | Transfer points to another member                                       |
| `/dice <bet>`                     | Roll three dice against the AI dealer                                   |
| `/blackjack <bet>`                | Play one round of 21 with Hit / Stand buttons; natural Blackjack settles immediately |
| `/house`                          | Show the dealer's accumulated win/loss across `/dice` and `/blackjack`  |
| `/maple_monster <name>`           | Search MapleStory monsters and drops                                    |
| `/maple_equip <name>`             | Search MapleStory equipment                                             |
| `/maple_scroll <name>`            | Search MapleStory scrolls                                               |
| `/maple_npc <name>`               | Search MapleStory NPCs                                                  |
| `/maple_quest <name>`             | Search MapleStory quests                                                |
| `/maple_map <name>`               | Search MapleStory maps                                                  |
| `/maple_item <name>`              | Search MapleStory item sources                                          |
| `/maple_stats`                    | View MapleStory database statistics                                     |
| `/help`                           | Show bot usage guide                                                    |
| `/ping`                           | Check bot latency                                                       |

## Self-Hosting

### Prerequisites

- Python 3.12+
- A Discord bot token ([Developer Portal](https://discord.com/developers/applications))
- An OpenAI-compatible endpoint + API key — either a single provider (OpenAI, Gemini via its OpenAI-compatible endpoint, etc.) or a [LiteLLM](https://github.com/BerriAI/litellm) proxy fronting multiple providers

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
uv run python scripts/artale_data.py
```

This scrapes `artalemaplestory.com` and writes JSON files into `data/maplestory/`.

## Configuration

Create a `.env` file (or copy from `.env.example`):

```env
# Required
DISCORD_BOT_TOKEN=your_bot_token
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1   # or any OpenAI-compatible endpoint

# Optional
DISCORD_TEST_SERVER_ID=your_test_server_id
```

### Slash Command Sync

All slash commands register globally on first start (no per-guild pinning). Discord's global propagation can take up to an hour for brand-new commands; subsequent edits to existing commands typically appear within a few minutes. If a new command does not show up on your client right away, try `Ctrl+R` to refresh, or wait a bit.

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

- **Message Logging**: Messages in channels where the bot is present are logged locally to SQLite (`data/messages.db`). Data stays on your server and is never shared externally.
- **Points Database**: Per-user point balances live in a separate local SQLite file (`data/economy.db`). Only the Discord user ID, the most recently seen username, and balance counters are stored. Balances are shared across every server the bot runs in.
- **API Calls**: Text, images, supported file attachments, embedded media, and sender identity (display name, username, and Discord user ID of participants in the active chat context) are sent to the configured LLM API only when the bot is responding, such as when it is mentioned in a guild or messaged in DM. User IDs are included so the bot can tag other participants when asked. No data is shared with other third parties.
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
