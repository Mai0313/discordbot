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

- **AI 聊天**：在 server 标记机器人或发送 DM。它可以回答问题、总结近期聊天、检查支持的附件、观看贴上的 YouTube 视频、生成或编辑图片、生成短视频、以接续 reply 消息延续长回复，并在可用时使用 model-provided web tools。它还会在后台慢慢积累对你个人偏好的长期记忆（跨服务器、仅自己可见），可用 `/memory show`、`/memory clear` 与 `/memory regenerate` 管理。
- **Threads 解析**：贴上 Threads.net 或 Threads.com URL，机器人会展开贴文、媒体与 reply chain。
- **视频下载**：`/download_video` 可从 YouTube、TikTok、Instagram、X、Facebook、Bilibili，以及其他 yt-dlp 支持的网站下载视频，文件太大时会自动 retry 低画质。
- **虚拟欢乐豆与金融系统**：用户可从消息获得虚拟欢乐豆，可每日签到、转账、购买 VIP、使用长期个人信贷或央行借款，并查看排行榜。
- **模拟股市**：`/stock` 开启一则公开 market message，内含 DB-managed virtual companies；选股、受 float supply、borrow cap 与单人 49% long holding cap 限制的交易、仓位摘要、近期交易记录、liquidity-based slippage、定期刷新新闻与 7 日图表都在同一则公开 message 内 edit 切换，只有发起 `/stock` 的 user 可以操作 controls。
- **赌场游戏**：多人 `/games blackjack` 与 `/games dragon_gate` lobby。Blackjack 庄家改为赌场系统 (deterministic H17)，bot 本身会以玩家身份入桌并由独立的确定性策略 (fractional-Kelly 下注与 EV 决策) 决策，`/casino` 与 `/pocat` 分别显示赌场账本与 bot 玩家钱包。单人 `/games fishing` 则是买钓具抛竿、回收欢乐豆的 sink 玩法，鱼分 N 到 UR 稀有度并有最大单笔渔获排行榜。
- **MapleStory Artale 数据库**：`/maplestory` 子命令可查询怪物、装备、卷轴、NPC、任务、地图、掉落来源与数据库统计。
- **本地化指令**：slash command metadata 与 `/help` 支持英文、繁体中文、日文。AI 回复会跟随用户语言。

## 指令

| 指令                                                             | 功能                                                                     |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `@bot <message>`                                                 | 和 AI 聊天。需要机器人检查文件或图片时，可附上支持的附件。               |
| _Threads URL_                                                    | 自动展开 Threads 贴文与媒体。                                            |
| `/download_video <url> [quality]`                                | 下载视频并传回 Discord。                                                 |
| `/balance [member]`                                              | 私密显示成员的虚拟欢乐豆余额、债务、stock holdings、净资产与 VIP 状态。  |
| `/checkin`                                                       | 领取每日签到奖励。                                                       |
| `/vip`                                                           | 购买永久 VIP 权益。                                                      |
| `/leaderboard`                                                   | 显示全域余额排行榜。                                                     |
| `/loss_leaderboard`                                              | 显示今日赌场输钱累计排行榜。                                             |
| `/credit status\|borrow\|call\|repay`                            | 处理个人信贷申请、180 秒批准/拒绝/取消按钮、还款、催收与状态。           |
| `/central_bank status\|borrow\|call\|repay`                      | 处理央行借款申请、180 秒批准/拒绝/取消按钮、还款、催收与可放贷额度。     |
| `/stock`                                                         | 公开股票市场消息，明细、交易、新闻、记录都在同一则 message edit。        |
| `/give <member> <amount>`                                        | 转账虚拟欢乐豆给其他成员或 bot。                                         |
| `/admin refund_tax\|collect_tax`                                 | admin-only 手动调整成员或 bot 余额。                                     |
| `/games blackjack <bet>`                                         | 开一个多人 Blackjack lobby；`bet` 可输入含逗号的数字，`0` 就是 all in。  |
| `/games dragon_gate`                                             | 开一个由共享 jackpot pool 支撑的多人射龙门桌。                           |
| `/games fishing`                                                 | 打开个人钓鱼面板，买钓竿与鱼饵抛竿，是回收欢乐豆的 sink 玩法。           |
| `/casino`                                                        | 显示赌场系统累积 P&L (跨服务器)。                                        |
| `/pocat`                                                         | 显示 bot 玩家自己的钱包 (等同 `/balance @bot`)。                         |
| `/maplestory monster`, `/maplestory equip`, `/maplestory scroll` | 查询 MapleStory Artale 怪物、装备与卷轴。                                |
| `/maplestory npc`, `/maplestory quest`, `/maplestory map`        | 查询 NPC、任务与地图。                                                   |
| `/maplestory item`, `/maplestory stats`                          | 查询物品掉落来源与数据库统计。                                           |
| `/memory show\|clear\|regenerate`                                | 私密查看、清除或重建 bot 对你记住的内容（regenerate 会排程在后台执行）。 |
| `/help`                                                          | 显示 Discord 内的使用指南。                                              |
| `/ping`                                                          | 检查 bot latency。                                                       |

## 自托管

### 前置需求

- Python 3.12 或更新版本
- 来自 [Discord Developer Portal](https://discord.com/developers/applications) 的 Discord bot token
- OpenAI-compatible API key 与 base URL。若想把 OpenAI、Gemini、Claude 和其他 provider 放在同一个 endpoint 后面，建议使用 LiteLLM。
- 视频下载需要 `ffmpeg`，生成 board 图片需要支持 CJK 的 fonts。Docker image 已经内置两者。

### Docker

```bash
git clone https://github.com/Mai0313/discordbot.git
cd discordbot
cp .env.example .env
# edit .env
mkdir -p data
docker compose up -d
```

容器以 UID 1000 运行,让 `data/` 下的文件保持由你的主机用户拥有;如果你的主机用户 UID 不是 1000,请在 `docker-compose.yaml` 中覆写 `user:`。

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

本地测试央行批准流程时，可以设置 `ECONOMY_ALLOW_CENTRAL_BANK_SELF_APPROVAL=true`。正式环境请保持未设置或 `false`。

个人长期记忆永远开启；用户可以用 `/memory show`、`/memory clear` 与 `/memory regenerate` 管理自己的记忆。

## 数据与隐私

这个 bot 会把 runtime data 存在本地 `data/` 底下；SQLite 数据库集中在 `data/database/`。

- `database/messages.db`：human messages 与 bot 自己的回复，用于聊天历史与摘要。
- `database/economy.db`：`user_wallet` 存每位用户的可用余额与 gross totals，`user_account` 存 cached Discord account name / avatar URL、VIP、admin、央行成员、签到与 leaderboard flags，另存长期信贷申请与契约、赌场每日统计，以及 bot-wide jackpot pool 与 casino ledger。
- `database/stock.db`：DB-managed 模拟 stock profile、float supply、price tick、position、trade operation、ordered trade leg 与 AI-or-fallback stock news。
- `database/games.db`：每位玩家的 Blackjack 对局历史、钓鱼目录与每位用户的装备、鱼饵与渔获记录，以及公开 expiring response 的清理追踪（guild/channel 名称、user name、channel ID 与 message ID），用于 bot 重启后的清理。
- 临时 media 下载使用项目根目录的 `tmp/` scratch folder（不在 `data/` 底下），发送完成后即删除。
- `memories/`：每个 Discord user id 一个文件夹的纯文本 markdown 个人长期记忆，由你的对话在后台积累，并在后续 AI 回复时注入。

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
