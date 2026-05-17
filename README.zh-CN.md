<div align="center" markdown="1">

# AI 智能 Discord 机器人

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

[**English**](./README.md) | [**繁體中文**](./README.zh-TW.md) | **简体中文**

</div>

一个自托管 Discord 机器人，提供 AI 聊天、图片与视频生成、Threads 链接展开、视频下载、虚拟欢乐豆、赌场小游戏，以及 MapleStory Artale 查询。它基于 nextcord 运行，用本地 SQLite 保存 runtime data，并连接 OpenAI-compatible LLM endpoint，例如 LiteLLM。

## 功能

- **AI 聊天**：在 server 标记机器人或发送 DM。它可以回答问题、总结近期聊天、检查支持的附件、生成或编辑图片、生成短视频，并在可用时使用 model-provided web tools。
- **Threads 解析**：贴上 Threads.net 或 Threads.com URL，机器人会展开贴文、媒体与 reply chain。
- **视频下载**：`/download_video` 可从 YouTube、TikTok、Instagram、X、Facebook、Bilibili，以及其他 yt-dlp 支持的网站下载视频，文件太大时会自动 retry 低画质。
- **虚拟欢乐豆**：用户可从消息与 AI 回复获得虚拟欢乐豆，可每日签到、转账、购买 VIP、借款到台北时间每日重置，并查看排行榜。
- **赌场游戏**：多人 `/blackjack` 与 `/dragon_gate` lobby，带 AI dealer 对话、公开结果 embed 与自动清理。
- **MapleStory Artale 数据库**：`/maple_*` 指令可查询怪物、装备、卷轴、NPC、任务、地图、掉落来源与数据库统计。
- **本地化指令**：slash command metadata 与 `/help` 支持英文、繁体中文、日文。AI 回复会跟随用户语言。

## 指令

| 指令                                              | 功能                                                       |
| ------------------------------------------------- | ---------------------------------------------------------- |
| `@bot <message>`                                  | 和 AI 聊天。需要机器人检查文件或图片时，可附上支持的附件。 |
| _Threads URL_                                     | 自动展开 Threads 贴文与媒体。                              |
| `/download_video <url> [quality]`                 | 下载视频并传回 Discord。                                   |
| `/balance`                                        | 私密显示你的虚拟欢乐豆余额、VIP 状态与借款状态。           |
| `/checkin`                                        | 领取每日签到奖励。                                         |
| `/vip`                                            | 购买永久 VIP 权益。                                        |
| `/leaderboard`                                    | 显示全域余额排行榜。                                       |
| `/loss_leaderboard`                               | 显示今日赌场输钱排行榜。                                   |
| `/borrow <amount>`                                | 借虚拟欢乐豆，到下一次 Asia/Taipei 每日重置为止。          |
| `/repay <amount>`                                 | 用余额偿还未还本金。                                       |
| `/give <member> <amount>`                         | 转账虚拟欢乐豆给其他成员。                                 |
| `/admin refund_tax\|collect_tax`                  | admin-only 手动余额调整。                                  |
| `/blackjack <bet>`                                | 开一个多人 Blackjack lobby。                               |
| `/dragon_gate`                                    | 开一个由共享 jackpot pool 支撑的多人射龙门桌。             |
| `/house`                                          | 显示 Blackjack dealer ledger。                             |
| `/maple_monster`, `/maple_equip`, `/maple_scroll` | 查询 MapleStory Artale 怪物、装备与卷轴。                  |
| `/maple_npc`, `/maple_quest`, `/maple_map`        | 查询 NPC、任务与地图。                                     |
| `/maple_item`, `/maple_stats`                     | 查询物品掉落来源与数据库统计。                             |
| `/help`                                           | 显示 Discord 内的使用指南。                                |
| `/ping`                                           | 检查 bot latency。                                         |

## 自托管

### 前置需求

- Python 3.12 或更新版本
- 来自 [Discord Developer Portal](https://discord.com/developers/applications) 的 Discord bot token
- OpenAI-compatible API key 与 base URL。若想把 OpenAI、Gemini、Claude 和其他 provider 放在同一个 endpoint 后面，建议使用 LiteLLM。
- 视频下载需要 `ffmpeg`。Docker image 已经内置。

### Docker

```bash
git clone https://github.com/Mai0313/discordbot.git
cd discordbot
cp .env.example .env
# edit .env
docker compose up -d
```

### 本地

```bash
git clone https://github.com/Mai0313/discordbot.git
cd discordbot
uv sync
cp .env.example .env
# edit .env
uv run discordbot
```

刷新内置的 MapleStory Artale data：

```bash
uv run python scripts/artale_data.py
```

## 配置

从 `.env.example` 建立 `.env`，并设置必要值：

```env
DISCORD_BOT_TOKEN=your_discord_bot_token
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
```

`OPENAI_BASE_URL` 可以直接指向 OpenAI，也可以指向 LiteLLM 这类 OpenAI-compatible gateway。

## 数据与隐私

这个 bot 会把 runtime data 存在本地 `data/` 底下。

- `messages.db`：human messages 与 bot 自己的回复，用于聊天历史与摘要。
- `economy.db`：虚拟欢乐豆余额、VIP flag、借款、签到、赌场交易与 jackpot state。
- `game_cleanup.db`：公开 game 或 economy response 的 Discord channel ID 与 message ID，用于 bot 重启后的清理。
- `model_prices.json`：缓存的 LiteLLM pricing metadata，用于 AI 回复费用估算。
- `downloads/` 与 `threads/`：临时 media scratch folders。

当 bot 需要用 AI 回复时，当前上下文中的相关文字、支持的附件、embedded media 与参与者身份会送到你配置的 LLM endpoint。本项目不会把这些资料送到其他服务。

## 故障排除

- **Slash commands 没出现**：确认邀请链接包含 `applications.commands`。Global command propagation 可能需要一些时间，尤其是新增指令。
- **AI 回复失败**：检查 `OPENAI_API_KEY`、`OPENAI_BASE_URL`，以及 cogs 中配置的 model routing。
- **视频下载失败**：更新 `yt-dlp`，并确认已安装 `ffmpeg`。也可以尝试较低的 quality。
- **权限错误**：mention-based chat 与本地消息记录需要 Message Content intent，embed、attachment、reaction 与 slash command 也需要一般 Discord 权限。

## 开发

Contributor setup、code conventions、tests 与 release notes 请见 [CONTRIBUTING.md](./CONTRIBUTING.md)。

[文档](https://mai0313.github.io/discordbot/) | [报告问题](https://github.com/Mai0313/discordbot/issues) | [讨论](https://github.com/Mai0313/discordbot/discussions)

[![Contributors](https://contrib.rocks/image?repo=Mai0313/discordbot)](https://github.com/Mai0313/discordbot/graphs/contributors)
