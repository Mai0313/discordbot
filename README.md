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

- **AI chat**: mention the bot in a server or send a DM. It can answer questions, summarize recent chat, inspect supported attachments, watch a linked YouTube video, generate or edit images, generate short videos from a prompt or attached images, edit a referenced video, continue long replies as follow-up reply messages, and use model-provided web tools when available. It also builds a private per-user long-term memory of your preferences in the background — privacy-scoped by source, so something told in one server never surfaces in another (only your tone preferences and clearly harmless general facts carry over) — manageable with `/memory show` and `/memory regenerate`.
- **Threads parser**: paste a Threads.net or Threads.com URL and the bot expands the post, media, and reply chain. Mention the bot alongside the link instead and it reads the post and answers about it.
- **Video downloader**: `/download_video` downloads videos from YouTube, TikTok, Instagram, X, Facebook, Bilibili, and other yt-dlp supported sites. Douyin is supported too, watermark free and including photo posts. Files too large to upload are served as a link instead.
- **Virtual currency and finance**: users earn 虛擬歡樂豆 from messages, can check in daily, transfer balances, buy VIP, use long-term personal credit or central-bank loans, and view leaderboards.
- **Simulated stock market**: `/stock` opens one public market message with DB-managed virtual companies; selecting a stock, trading with float-supply, borrow, and per-user 49% long holding caps, position summaries, recent trades, liquidity-based slippage, periodically refreshed news, and the 7D chart all update that same public message. Only the opener can operate its controls.
- **Casino games**: multiplayer `/games blackjack` and `/games dragon_gate` lobbies. Blackjack is dealt by the casino system (deterministic H17), the bot itself joins each round as a player driven by its own deterministic strategy (fractional-Kelly betting and EV-based play), and `/casino` / `/pocat` surface the casino ledger and the bot's wallet. Solo `/games fishing` adds a gear-and-cast money sink with an N to UR rarity ladder and a biggest-catch leaderboard.
- **MapleStory Artale database**: `/maplestory` subcommands search monsters, equipment, scrolls, NPCs, quests, maps, item drops, and database stats.
- **Localized commands**: slash command metadata and `/help` are localized for English, Traditional Chinese, and Japanese. AI replies follow the user's language.

## Commands

| Command                                                          | What it does                                                                                                           |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `@bot <message>`                                                 | Chat with the AI. Attach supported files or images when you want the bot to inspect them.                              |
| _Threads URL_                                                    | Automatically expands Threads posts and media, unless the bot is mentioned (then it answers about the post instead).   |
| `/download_video <url> [quality]`                                | Downloads a video and sends it back to Discord. A Douyin photo post comes back as images.                              |
| `/balance [member]`                                              | Privately shows a member's 虛擬歡樂豆 balance, debt, stock holdings, net worth, and VIP status.                        |
| `/checkin`                                                       | Claims the daily check-in reward.                                                                                      |
| `/vip`                                                           | Buys permanent VIP perks.                                                                                              |
| `/leaderboard`                                                   | Shows the global top balances.                                                                                         |
| `/loss_leaderboard`                                              | Shows today's accumulated casino losses.                                                                               |
| `/credit status\|borrow\|call\|repay`                            | Handles personal credit requests, 180-second approval/rejection/cancel buttons, repayment, collection, and status.     |
| `/central_bank status\|borrow\|call\|repay`                      | Handles central-bank loan requests, 180-second approval/rejection/cancel buttons, repayment, collection, and capacity. |
| `/stock`                                                         | Opens one public stock market message that edits in place for details, trading, news, and history.                     |
| `/give <member> <amount>`                                        | Transfers 虛擬歡樂豆 to another member or bot.                                                                         |
| `/admin refund_tax\|collect_tax`                                 | Admin-only manual balance adjustments for members or bots.                                                             |
| `/games blackjack <bet>`                                         | Opens a multiplayer Blackjack lobby; `bet` accepts comma-formatted numbers, and `0` means all in.                      |
| `/games dragon_gate`                                             | Opens a multiplayer 射龍門 table backed by the shared jackpot pool.                                                    |
| `/games fishing`                                                 | Opens your personal fishing panel to buy gear and cast for graded fish; a currency sink.                               |
| `/casino`                                                        | Shows the casino system's cumulative profit and loss.                                                                  |
| `/pocat`                                                         | Shows the bot player's own wallet (shortcut for `/balance @bot`).                                                      |
| `/maplestory monster`, `/maplestory equip`, `/maplestory scroll` | Search MapleStory Artale monsters, equipment, and scrolls.                                                             |
| `/maplestory npc`, `/maplestory quest`, `/maplestory map`        | Search NPCs, quests, and maps.                                                                                         |
| `/maplestory item`, `/maplestory stats`                          | Search item drop sources and database stats.                                                                           |
| `/memory show\|regenerate`                                       | Privately shows or rebuilds what the bot remembers about you; regenerate is scheduled in the background.               |
| `/help`                                                          | Shows the in-Discord guide.                                                                                            |
| `/ping`                                                          | Checks bot latency.                                                                                                    |

## Self-Hosting

### Prerequisites

- Python 3.12 or newer
- A Discord bot token from the [Discord Developer Portal](https://discord.com/developers/applications)
- An OpenAI-compatible API key and base URL. LiteLLM is the expected choice when you want to route OpenAI, Gemini, Claude, and other providers behind one endpoint.
- `ffmpeg` for video downloads and CJK-capable fonts for generated board images. The Docker image already includes both.

### Docker

```bash
git clone https://github.com/Mai0313/discordbot.git
cd discordbot
cp .env.example .env
# edit .env
mkdir -p data
docker compose up -d
```

The container runs as UID 1000 so files under `data/` stay owned by your host user; if your host user has a different UID, override `user:` in `docker-compose.yaml`.

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
GEMINI_API_KEY=your_google_ai_studio_key
```

`OPENAI_BASE_URL` may point at OpenAI directly or at an OpenAI-compatible gateway such as LiteLLM. `GEMINI_API_KEY` is a Google AI Studio key used directly (not through the gateway) for video generation, Gemini Files API attachment uploads, YouTube video answers, and deep research; leave it unset to disable those features.

For local central-bank approval testing, set `ECONOMY_ALLOW_CENTRAL_BANK_SELF_APPROVAL=true`. Keep it unset or `false` in production.

Per-user long-term memory is always on; users manage their own memory with `/memory show` and `/memory regenerate`. Every remembered fact is tagged with where it was learned, and anything private stays confined to that server or DM.

## Data And Privacy

This bot stores runtime data locally under `data/`; SQLite databases live in `data/database/`.

- `database/messages.db`: human messages and this bot's replies, used for chat history and summaries.
- `database/economy.db`: `user_wallet` spendable balances and gross totals, `user_account` cached Discord account names / avatar URLs plus VIP, admin, central banker, check-in, and leaderboard flags, long-term loan requests / contracts, casino daily counters, plus bot-wide jackpot pools and the casino ledger.
- `database/stock.db`: DB-managed simulated stock profiles, float supply, price ticks, positions, trade operations, ordered trade legs, and AI-or-fallback stock news.
- `database/games.db`: per-player Blackjack round history, the fishing catalog plus per-user gear, bait, and catch history, and cleanup tracking (guild/channel names, user names, channel IDs, and message IDs) for public expiring responses that should be removed after restart.
- Temporary media downloads use the project-root `tmp/` scratch folder (not under `data/`) and are deleted after sending.
- `memories/`: per-user long-term memory as plaintext markdown files in one folder per Discord user id, built in the background from your conversations and injected into future AI replies.

When the bot responds with AI, relevant text, supported attachments, embedded media, and participant identity from the active context are sent to the configured LLM endpoint. Data is not sent to any other service by this project.

## Troubleshooting

- **Slash commands do not appear**: make sure the invite includes `applications.commands`. Global command propagation can take time, especially for new commands.
- **AI replies fail**: check `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and the model routing configured in the cogs.
- **Video download fails**: update `yt-dlp` and make sure `ffmpeg` is installed. Try a lower quality setting.
- **Permission errors**: the bot needs Message Content intent for mention-based chat and local message logging, plus normal permissions for embeds, attachments, reactions, and slash commands.

## Development

Contributor setup, code conventions, tests, and release notes live in [CONTRIBUTING.md](./.github/CONTRIBUTING.md).

[Documentation](https://mai0313.github.io/discordbot/) | [Report a Bug](https://github.com/Mai0313/discordbot/issues) | [Discussions](https://github.com/Mai0313/discordbot/discussions)

[![Contributors](https://contrib.rocks/image?repo=Mai0313/discordbot)](https://github.com/Mai0313/discordbot/graphs/contributors)
