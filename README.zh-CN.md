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

功能丰富的 Discord 机器人，具备 AI 智能对话、图片与视频生成、内容解析、多平台视频下载、点数系统（每日签到、可选 VIP、每日重置的贷款）、赌场小游戏，以及枫之谷游戏数据库。支持多国语言。

## 功能

### AI 聊天

标记机器人（`@bot`）或发送私信即可开始对话。AI 后端是任何 OpenAI 兼容端点（通常搭配 [LiteLLM](https://github.com/BerriAI/litellm) proxy 对接 OpenAI、Google Gemini、Anthropic Claude 等），机器人会根据任务切换模型 — 用快速 model 做意图分类与图片 caption、慢速推理 model 做对话与摘要、专用 image model 做图片生成/编辑、video model 做短片生成。支持功能：

- **文字对话** — 通过 OpenAI Responses API 实时流式响应
- **多媒体理解** — 附上图片、贴图，或 PDF、文字、JSON 等支持的文件即可向机器人提问；也能读取消息或引用回复中内嵌的图片（例如已解析的 Threads 帖子 embed）
- **图片生成与编辑** — 请机器人根据文字描述绘制或创作图片（附上图片即可进行编辑修改）
- **视频生成** — 请机器人生成短视频（请求之间有冷却时间）
- **聊天摘要** — 请机器人总结近期对话内容
- **网络搜索与 URL 读取** — 机器人会自动使用模型对应的工具（Gemini 的 `googleSearch` + `urlContext`、Claude 的 `web_search` + `web_fetch`，或 OpenAI 的 `web_search`）获取最新信息
- **用户标记** — 请机器人通知或转告近期对话中的其他成员（例如「帮我跟 @alice 说我会晚到」），只要该成员曾出现在近期聊天记录中即可被标记
- **进度反应** — 以 emoji reaction 显示实时处理状态（🤔 → 🔀 → 🎨/🎬/📖/❓ → 🆗，模型用到 web search 时加 🌐，出错时显示 ❌）
- **回复脚注** — 每次 AI 回复会在末端以 Discord 引用格式（`>`）显示 model 名称、input/output token 数量、预估 USD 费用（从上游 LiteLLM 的 price table 计算，首次查询时下载并缓存在本地），以及这一轮赚到的点数
- **自动解除 timeout** — 如果有 moderator 把机器人 timeout，机器人会自动解除 timeout、从 audit log 识别出对方是谁，并在最近有人聊天的频道回一句 AI 生成的回复

### Threads 解析

贴上 Threads.net 链接，机器人会自动展开帖子 — 显示文字内容、图片、互动数据，并下载附带的视频。如果链接指向某条回复，机器人会把整条回复链一起展开，原始帖子与中间每一层回复都会出现在同一条消息里，并用灰阶渐变色带（由浅到深）区分层级。只有用户贴的那一篇会把视频下载并附上来，上层回复若有视频则改用点击链接提示，避免多层附件混在一起无法区分。

### 视频下载

使用 `/download_video` 从多个平台下载视频：

- YouTube、TikTok、Instagram、X (Twitter)、Facebook、Bilibili
- 画质选项：最佳、高画质 (1080p)、中等 (720p)、低画质 (480p)
- 文件超过 Discord 25 MB 限制时自动降为低画质
- Facebook 分享链接（`facebook.com/share/r/...`）会自动展开

### 点数系统与赌场小游戏

机器人会用本地 SQLite (`data/economy.db`) 持久保存每位 Discord 用户的点数余额，**点数跨服务器共享**，同一个账号在任何 guild 看到的余额都一样。

**获得点数：** 每则非 bot 用户消息都会获得 5,000 点数。AI 流式回复会再追加以 `total_tokens` (input + output) 计算的 bonus，实际数字会显示在回复 footer。`/checkin` 每天可领 100,000 点数，连续 7 天为一个 cycle（线性加成：第 1 天 1×、第 7 天 4×）。Threads 解析与 `/download_video` 不会在基础消息奖励之外再付额外 action reward。

**花用点数：** 赌场游戏会先开 lobby。房主可以单人开始，也可以等其他玩家加入；只有房主可以开始 table。Blackjack bet 会在玩家加入时检查，开始时重新确认余额，等 table 结算时才套用本局正负结果。如果 Blackjack 房主输入超过目前余额的 bet，lobby 的 table stake 会 clamp 成房主实际 all-in 金额，后续玩家默认跟这个金额，不会被原本过大的输入值强制全下；只有余额为 0 或负数时才会拒绝玩家。射龙门则跑在**跨桌共享的全局彩金池**上 (`jackpot_pool` 中 `game_id="dragon_gate"` 的单一 row，首次启动由系统种入 100,000 点 on the house)。入场费固定 5,000 (进场时直接拨进彩金池)，最低下注 10,000，下注上限就是当下彩金池总额，每一手下注的赔付即时写入玩家账户与彩金池，不再等桌结束才结算。Dragon Gate loss 会 clamp 在余额 0，彩金池只收到实际扣到的金额，归零玩家会自动离桌，其他玩家继续玩。任何玩家都可中途按「离桌」退出；若离桌或 timeout 时该玩家的桌内累计仍为正，那部分净赢会逆向退回彩金池 (「逆赢不拿」)。桌结束的条件是彩金池被刷光、所有玩家都离桌或归零、或 180 秒无人互动。庄家是个 AI，开局会嘴一下整桌下注，所有 Blackjack 玩家结束后会显示思考中并决定庄家 hit / stand，结算时会依结果嘴或夸玩家。Embed 上「庄家」的显示名称直接用机器人自己的 Discord display name，所以未来 `gen_reply` 看历史消息时会把这些对白认作自己过去的发言，而不是某个无名 dealer。游戏结算 footer 会显示每位玩家的本局 delta、结算后余额，以及 Blackjack 庄家的 decision path；`/house` 看的是 Blackjack 庄家的 ledger，射龙门的对手是彩金池而不是庄家，所以不会影响 `/house` 数字。

**VIP：** `/vip` 一次性花费 10,000,000 点数购买永久 VIP 标记。VIP 会获得 1.5x Blackjack payout、2x 签到基础点数、2x 借款上限。

**管理员调整：** economy admin 会存在 `user_account.is_admin`，用 `uv run python scripts/manage_admin.py grant|revoke|list` 管理。admin 可以用 `/admin refund_tax` 退税加点，也可以用 `/admin collect_tax` 收税扣点；收税会 clamp 在余额 0，不会扣成负数。

可互动的 game/lobby message 使用 180 秒无互动 timeout，这里的三分钟是指没有任何 component interaction，不是从开桌固定倒数。公开 response 进入不可互动状态后，会在送出或结算三分钟后自动删除，包含赌场游戏 final embed、余额不足拒绝开局回复、`/leaderboard`、`/loss_leaderboard`、`/house`、`/give` 转账结果 embed，以及 `/admin refund_tax` / `/admin collect_tax` 成功结果 embed。`/balance`、`/borrow`、`/repay`、`/checkin`、`/vip` 和 admin 错误回复只有调用者看得到。游戏 response 的 message ID 会存在本地，bot 重启后会在下次 startup 删掉上次留下的进行中或已结算游戏 embed。

| Slash command       | 玩法                                                                                                                                                                                                                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `/blackjack <下注>` | 支持多人 21 点 lobby，含 Join / Leave / Start button。玩家动作包含 **Hit / Stand / Double Down / Split / Surrender**；只有庄家明牌 A 且可买 Insurance 时才会显示保险 button。所有玩家结束后牌桌会先显示庄家思考中，再由 AI 庄家决定 hit / stand（Soft 17 交由 AI 判断）。            |
| `/dragon_gate`      | 支持多人射龙门 lobby, 跑**跨桌全局彩金池** (cross-table 累积)。入场费固定 5,000 进彩金池, 最低下注 10,000, 上限 = 当下彩金池, 每手即时结算; 玩家可中途按「离桌」退出 (逆赢不拿); 180 秒无互动 timeout 整桌结束。射进龙门从彩金池赢 1 倍；loss 最多扣到玩家目前余额，归零后自动离桌。 |

**21 点玩家动作：** Double Down 加倍下注后只再抽一张并强制 stand。Split 仅同 rank pair 可分；**Split 后禁止 Double**（No DAS），Split Aces 两手各只能再拿一张，且 21 点算一般 1 倍赢（不是天生 Blackjack 1.5 倍）。Late Surrender 退回一半本金；庄家 peek 出 Blackjack 后就无法投降。

**21 点提前结算与 peek：** `Blackjack` 指的是起手两张牌就是 A + 10 点牌，赔 1.5 倍。庄家明牌 A 时会先进入 Insurance phase（保险注 = 原注一半，庄家 peek 到 Blackjack 赔 2:1），Insurance Yes / No button 只会在这个 phase 显示；明牌 10 点则 silent peek。peek 命中 Blackjack 会直接结算整桌，玩家同样也是天生 Blackjack 才平手。任意凑到 21 不算 natural Blackjack，不会跳过动作阶段。

**管理点数：**

- `/balance` — private 查看自己的余额、未还本金 (如果有) 与 VIP 状态。
- `/checkin` — 领取今日签到奖励；回复是 ephemeral 只有自己看得到。每天 00:00 Asia/Taipei 结算，连续 7 天为一 cycle，超过或漏签会 reset 回第 1 天。
- `/vip` — private 花 10,000,000 点数购买永久 VIP。
- `/leaderboard` — 机器人所有服务器的全域 Top 10（庄家自己的账户会被排除）。
- `/loss_leaderboard` — 今日 00:00 Asia/Taipei 之后赌场净输最多的全域 Top 10。
- `/borrow <金额>` — private 依 Discord 账号年龄借点数，超过今日剩余额度时会自动借到剩余额度。**每天 00:00 Asia/Taipei 本金自动归零**，没有利息。
- `/repay <金额>` — private 从目前余额偿还未还本金。
- `/give <成员> <金额>` — 把点数转给其他人（不能转给自己或机器人）。
- `/house` — 查看庄家在赌场游戏累积的输赢。庄家资金无上限，所以 ledger balance 可以是负数（代表整体玩家从庄家手里赢走的点数比较多）。
- `/admin refund_tax|collect_tax` — admin-only 退税加点/收税扣点，admin 用 `scripts/manage_admin.py` 管理。

借款后，每次 income event（message reward / chat reward / 赌场 payout）会先自动拿 50% 还本金，剩下才进钱包。`/give` 的收款方不会被自动扣去还债。

### 枫之谷 Artale 数据库

- `/maple_monster` — 按名称搜索怪物，查看属性、出没地图与掉落物
- `/maple_equip` — 按名称搜索装备，查看属性与获取方式
- `/maple_scroll` — 按名称搜索卷轴与附加属性
- `/maple_npc` — 按名称搜索 NPC 与所在位置
- `/maple_quest` — 按名称搜索任务、等级范围与频率
- `/maple_map` — 按名称搜索地图、区域与出没怪物
- `/maple_item` — 搜索物品并查看哪些怪物会掉落
- `/maple_stats` — 查看数据库统计信息
- 支持模糊搜索与多语言显示

### 多语言支持

Slash command 的名称、描述，以及 `/help` 使用指南目前支持英文、繁体中文 (`zh-TW`) 与日文 (`ja`)。AI 对话回复则会跟随用户输入的语言。

## 指令

| 指令                             | 说明                                                                          |
| -------------------------------- | ----------------------------------------------------------------------------- |
| `@bot <消息>`                    | 与 AI 对话（文字、媒体/文件、生成、摘要、网络搜索）                           |
| _Threads 链接_                   | 自动展开 Threads.net 帖子与媒体                                               |
| `/download_video <网址> [画质]`  | 从 YouTube、TikTok、Instagram、X、Facebook、Bilibili 下载视频                 |
| `/balance`                       | private 查看你目前的点数余额、贷款与 VIP 状态（跨服务器）                     |
| `/checkin`                       | 领取今日签到奖励（ephemeral；7 天 streak 加成，每天 Taipei 00:00 重置）       |
| `/vip`                           | private 购买永久 VIP（1.5x Blackjack payout、2x 签到、2x 借款上限）           |
| `/leaderboard`                   | 全域点数 Top 10                                                               |
| `/loss_leaderboard`              | 今日输最多 Top 10（每天 Taipei 00:00 重置）                                   |
| `/borrow <金额>`                 | private 借点数；超过上限时自动借到今日剩余额度                                |
| `/repay <金额>`                  | private 从余额偿还未还本金                                                    |
| `/give <成员> <点数>`            | 把点数转给其他成员                                                            |
| `/admin refund_tax\|collect_tax` | admin-only 退税加点/收税扣点                                                  |
| `/blackjack <下注>`              | 开一个 21 点 lobby，含 Hit / Stand / Double / Split / Surrender 与庄家 A 保险 |
| `/dragon_gate`                   | 开一桌射龙门 lobby，跑跨桌全局彩金池 (loss clamp 到 0、含离桌按钮)            |
| `/house`                         | 查看庄家在赌场游戏累积的输赢                                                  |
| `/maple_monster <名称>`          | 搜索枫之谷怪物与掉落物                                                        |
| `/maple_equip <名称>`            | 搜索枫之谷装备                                                                |
| `/maple_scroll <名称>`           | 搜索枫之谷卷轴                                                                |
| `/maple_npc <名称>`              | 搜索枫之谷 NPC                                                                |
| `/maple_quest <名称>`            | 搜索枫之谷任务                                                                |
| `/maple_map <名称>`              | 搜索枫之谷地图                                                                |
| `/maple_item <名称>`             | 搜索枫之谷物品来源                                                            |
| `/maple_stats`                   | 查看枫之谷数据库统计                                                          |
| `/help`                          | 显示机器人使用指南                                                            |
| `/ping`                          | 测试机器人延迟                                                                |

## 自托管

### 前置要求

- Python 3.12+
- Discord 机器人 Token（[开发者门户](https://discord.com/developers/applications)）
- OpenAI 兼容端点与 API 密钥 — 可以是单一 provider（OpenAI、Gemini 通过 OpenAI 兼容端点等），或是 [LiteLLM](https://github.com/BerriAI/litellm) proxy 对接多个 provider

### 方式一：Docker（推荐）

```bash
git clone https://github.com/Mai0313/discordbot.git
cd discordbot
cp .env.example .env
# 编辑 .env 填入你的 Token 和 API 密钥
docker-compose up -d
```

Docker 镜像已包含 `ffmpeg`，可处理视频/音频流合并。

### 方式二：本地安装

```bash
git clone https://github.com/Mai0313/discordbot.git
cd discordbot

# 安装 uv（Python 包管理器）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv sync

# 配置环境
cp .env.example .env
# 编辑 .env 填入你的 Token 和 API 密钥

# 运行机器人
uv run discordbot
```

### 可选：更新枫之谷 Artale 数据库

```bash
uv run python scripts/artale_data.py
```

此 script 会从 `artalemaplestory.com` 抓取数据，并将 JSON 文件写入 `data/maplestory/`。

## 配置

创建 `.env` 文件（或从 `.env.example` 复制）：

```env
# 必需
DISCORD_BOT_TOKEN=你的机器人token
OPENAI_API_KEY=你的api密钥
OPENAI_BASE_URL=https://api.openai.com/v1   # 或任何 OpenAI 兼容端点
```

### Slash Command Sync

所有 slash command 都走 Discord 全域注册（没有 per-guild pin）。第一次新增指令时 Discord 端 propagate 最多需要 1 小时；后续编辑既有指令通常几分钟内就会生效。如果新指令在你的 client 上没立刻出现，先试试 `Ctrl+R` 重整，或稍等一下。

## 各平台注意事项

### Bilibili

- 需要 `ffmpeg` 来合并分离的视频/音频流（Docker 镜像已内置）。
- 若出现「Requested format is not available」，请尝试较低的画质设置。
- 区域/年龄限制的视频可能需要提供 cookies（默认未配置）。

### Facebook

- 分享链接（`facebook.com/share/...`）会在下载前自动展开。
- 请保持 `yt-dlp` 为最新版本以获得最佳兼容性。

## 隐私与数据

本机器人遵守 Discord 服务条款与开发者政策。

- **消息记录**：机器人所在频道的消息会记录到本地 SQLite (`data/messages.db`)。数据仅存在你的服务器，不会外传。
- **点数数据库**：每位用户的点数余额存储在另一个本地 SQLite 文件 (`data/economy.db`)，会记录 Discord user ID、最近一次看到的 username、avatar URL，以及余额相关计数。余额会跨机器人运行的所有服务器共享。
- **游戏清理数据库**：等待清理的公开 game / economy response 只会把 Discord channel ID 与 message ID 存在 `data/game_cleanup.db`，让 bot 重启后能删除残留的 cleanup target。
- **API 调用**：文字、图片、支持的文件附件、内嵌媒体，以及发送者身份（当前对话上下文中参与者的 display name、username 与 Discord user ID）仅在机器人需要回复时才会发送至配置的 LLM API，例如在 guild 被标记或收到 DM 时。user ID 会一并传入，让机器人在被要求时可以标记其他成员。不会与其他第三方分享数据。
- **权限**：机器人需要 Message Content 意图用于标记聊天和可选的本地记录。斜线指令与嵌入/附件权限用于交互功能。
- **停用**：服务器管理员可通过调整机器人配置来停用消息记录。

## 常见问题

**机器人不响应指令？**
检查机器人权限，确保已启用 `applications.commands` 范围。

**视频下载失败？**
确认 `yt-dlp` 和 `ffmpeg` 为最新版本。尝试较低的画质设置。

**API 错误？**
验证 API 密钥并确认端点 URL 正确。

---

想要贡献？请参阅 [CONTRIBUTING.md](./CONTRIBUTING.md)。

[![Contributors](https://contrib.rocks/image?repo=Mai0313/discordbot)](https://github.com/Mai0313/discordbot/graphs/contributors)

[文档](https://mai0313.github.io/discordbot/) | [报告问题](https://github.com/Mai0313/discordbot/issues) | [讨论](https://github.com/Mai0313/discordbot/discussions)
