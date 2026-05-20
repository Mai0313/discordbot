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

A self-hosted Discord bot for AI chat, image and video generation, Threads link expansion, video downloads, virtual currency, casino mini-games, and MapleStory Artale lookups. It runs on nextcord, stores runtime data in local SQLite files, and talks to an OpenAI-compatible LLM endpoint such as LiteLLM.

## Features

- **AI chat**: mention the bot in a server or send a DM. It can answer questions, summarize recent chat, inspect supported attachments, generate or edit images, generate short videos, and use model-provided web tools when available.
- **Threads parser**: paste a Threads.net or Threads.com URL and the bot expands the post, media, and reply chain.
- **Video downloader**: `/download_video` downloads videos from YouTube, TikTok, Instagram, X, Facebook, Bilibili, and other yt-dlp supported sites, with automatic low-quality retry for large files.
- **Virtual currency**: users earn 虛擬歡樂豆 from messages and AI replies, can check in daily, transfer balances, buy VIP, borrow until the daily Taipei reset, and view leaderboards.
- **Casino games**: multiplayer `/blackjack` and `/dragon_gate` lobbies with AI dealer banter, public result embeds, and automatic cleanup.
- **MapleStory Artale database**: `/maple_*` commands search monsters, equipment, scrolls, NPCs, quests, maps, item drops, and database stats.
- **Localized commands**: slash command metadata and `/help` are localized for English, Traditional Chinese, and Japanese. AI replies follow the user's language.

## Commands

| Command                                           | What it does                                                                              |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `@bot <message>`                                  | Chat with the AI. Attach supported files or images when you want the bot to inspect them. |
| _Threads URL_                                     | Automatically expands Threads posts and media.                                            |
| `/download_video <url> [quality]`                 | Downloads a video and sends it back to Discord.                                           |
| `/balance`                                        | Privately shows your 虛擬歡樂豆 balance, VIP status, and loan status.                     |
| `/checkin`                                        | Claims the daily check-in reward.                                                         |
| `/vip`                                            | Buys permanent VIP perks.                                                                 |
| `/leaderboard`                                    | Shows the global top balances.                                                            |
| `/loss_leaderboard`                               | Shows today's accumulated casino losses.                                                  |
| `/borrow <amount>`                                | Borrows 虛擬歡樂豆 until the next Asia/Taipei daily reset.                                |
| `/repay <amount>`                                 | Repays outstanding loan principal from your balance.                                      |
| `/give <member> <amount>`                         | Transfers 虛擬歡樂豆 to another member.                                                   |
| `/admin refund_tax\|collect_tax`                  | Admin-only manual balance adjustments.                                                    |
| `/blackjack <bet>`                                | Opens a multiplayer Blackjack lobby.                                                      |
| `/dragon_gate`                                    | Opens a multiplayer 射龍門 table backed by the shared jackpot pool.                       |
| `/house`                                          | Shows the Blackjack dealer ledger.                                                        |
| `/maple_monster`, `/maple_equip`, `/maple_scroll` | Search MapleStory Artale monsters, equipment, and scrolls.                                |
| `/maple_npc`, `/maple_quest`, `/maple_map`        | Search NPCs, quests, and maps.                                                            |
| `/maple_item`, `/maple_stats`                     | Search item drop sources and database stats.                                              |
| `/help`                                           | Shows the in-Discord guide.                                                               |
| `/ping`                                           | Checks bot latency.                                                                       |

## Self-Hosting

### Prerequisites

- Python 3.12 or newer
- A Discord bot token from the [Discord Developer Portal](https://discord.com/developers/applications)
- An OpenAI-compatible API key and base URL. LiteLLM is the expected choice when you want to route OpenAI, Gemini, Claude, and other providers behind one endpoint.
- `ffmpeg` for video downloads. The Docker image already includes it.

### Docker

```bash
git clone https://github.com/Mai0313/discordbot.git
cd discordbot
cp .env.example .env
# edit .env
docker compose up -d
```

### Local

```bash
git clone https://github.com/Mai0313/discordbot.git
cd discordbot
uv sync
cp .env.example .env
# edit .env
uv run discordbot
```

To refresh the bundled MapleStory Artale data:

```bash
uv run python scripts/artale_data.py
```

## Configuration

Create `.env` from `.env.example` and set the required values:

```env
DISCORD_BOT_TOKEN=your_discord_bot_token
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
```

`OPENAI_BASE_URL` may point at OpenAI directly or at an OpenAI-compatible gateway such as LiteLLM.

## Data And Privacy

This bot stores runtime data locally under `data/`.

- `messages.db`: human messages and this bot's replies, used for chat history and summaries.
- `economy.db`: user-scoped 虛擬歡樂豆 balances, VIP flags, loans, check-ins, casino daily counters, and cached Discord account names / avatar URLs.
- `global_state.db`: bot-wide shared state such as jackpot pools.
- `game_cleanup.db`: Discord guild/channel names, user names, channel IDs, and message IDs for public game or economy responses that should be cleaned up after restart.
- `model_prices.json`: cached LiteLLM pricing metadata used for AI reply cost estimates.
- `downloads/` and `threads/`: temporary media scratch folders.

When the bot responds with AI, relevant text, supported attachments, embedded media, and participant identity from the active context are sent to the configured LLM endpoint. Data is not sent to any other service by this project.

## Troubleshooting

- **Slash commands do not appear**: make sure the invite includes `applications.commands`. Global command propagation can take time, especially for new commands.
- **AI replies fail**: check `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and the model routing configured in the cogs.
- **Video download fails**: update `yt-dlp` and make sure `ffmpeg` is installed. Try a lower quality setting.
- **Permission errors**: the bot needs Message Content intent for mention-based chat and local message logging, plus normal permissions for embeds, attachments, reactions, and slash commands.

## Development

Contributor setup, code conventions, tests, and release notes live in [CONTRIBUTING.md](./CONTRIBUTING.md).

[Documentation](https://mai0313.github.io/discordbot/) | [Report a Bug](https://github.com/Mai0313/discordbot/issues) | [Discussions](https://github.com/Mai0313/discordbot/discussions)

[![Contributors](https://contrib.rocks/image?repo=Mai0313/discordbot)](https://github.com/Mai0313/discordbot/graphs/contributors)
