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

功能丰富的 Discord 机器人，具备 AI 智能对话、图片与视频生成、内容解析、多平台视频下载，以及枫之谷游戏数据库。支持多国语言。

## 功能

### AI 聊天

标记机器人（`@bot`）即可开始对话，由 Google Gemini 提供支持：

- **文字对话** — 实时流式响应
- **图片理解** — 附上图片即可向机器人提问
- **图片生成** — 请机器人根据文字描述绘制或创作图片
- **视频生成** — 请机器人生成短视频（请求之间有冷却时间）
- **聊天摘要** — 请机器人总结近期对话内容
- **网络搜索** — 机器人在需要最新信息时会自动搜索网络
- **进度反应** — 以 emoji reaction 显示实时处理状态（🤔 → 🔀 → 🎨/🎬/📖/❓ → 🆗）

### Threads 解析

贴上 Threads.net 链接，机器人会自动展开帖子 — 显示文字内容、图片、互动数据，并下载附带的视频。

### 视频下载

使用 `/download_video` 从多个平台下载视频：

- YouTube、TikTok、Instagram、X (Twitter)、Facebook、Bilibili
- 画质选项：最佳、高画质 (1080p)、中等 (720p)、低画质 (480p)
- 文件超过 Discord 25 MB 限制时自动降为低画质
- Facebook 分享链接（`facebook.com/share/r/...`）会自动展开

### 枫之谷数据库

- `/maple_monster` — 按名称搜索怪物，查看属性、出没地图与掉落物
- `/maple_item` — 搜索物品并查看哪些怪物会掉落
- `/maple_stats` — 查看数据库统计信息
- 支持模糊搜索与多语言显示

### 多语言支持

指令与响应支持英文、繁体中文、简体中文和日文。

## 指令

| 指令                            | 说明                                                          |
| ------------------------------- | ------------------------------------------------------------- |
| `@bot <消息>`                   | 与 AI 对话（文字、图片、生成、摘要、网络搜索）                |
| _Threads 链接_                  | 自动展开 Threads.net 帖子与媒体                               |
| `/download_video <网址> [画质]` | 从 YouTube、TikTok、Instagram、X、Facebook、Bilibili 下载视频 |
| `/maple_monster <名称>`         | 搜索枫之谷怪物与掉落物                                        |
| `/maple_item <名称>`            | 搜索枫之谷物品来源                                            |
| `/maple_stats`                  | 查看枫之谷数据库统计                                          |
| `/ping`                         | 测试机器人延迟                                                |

## 自托管

### 前置要求

- Python 3.12+
- Discord 机器人 Token（[开发者门户](https://discord.com/developers/applications)）
- OpenAI 兼容 API 密钥（例如 Google Gemini 通过 OpenAI 兼容端点）

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

### 可选：更新枫之谷数据库

```bash
uv run update
```

## 配置

创建 `.env` 文件（或从 `.env.example` 复制）：

```env
# 必需
DISCORD_BOT_TOKEN=你的机器人token
API_KEY=你的api密钥
BASE_URL=https://api.openai.com/v1   # 或任何 OpenAI 兼容端点

# 可选
DISCORD_TEST_SERVER_ID=你的测试服务器id
SQLITE_FILE_PATH=sqlite:///data/messages.db
POSTGRES_URL=postgresql://user:pass@host/db
REDIS_URL=redis://host:6379/0
```

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

- **消息记录**：机器人所在频道的消息会记录到本地 SQLite。数据仅存在你的服务器，不会外传。
- **API 调用**：文字和图片仅在机器人被标记时才会发送至配置的 LLM API。不会与其他第三方分享数据。
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
