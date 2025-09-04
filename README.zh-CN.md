<center>

# AI 智能 Discord 机器人 🤖

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

[**English**](./README.md) | [**繁體中文**](./README.zh-TW.md) | **简体中文**

</center>

基于 **nextcord** 框架开发的全功能 Discord 机器人，提供 AI 智能对话、内容处理和实用工具功能。支持多语言界面与整合式网络搜索功能。🚀⚡🔥

_欢迎提供建议和贡献!_

## ✨ 主要功能

### 🤖 AI 智能互动

- **文字生成**：支持多种 AI 模型（OpenAI GPT-4o — 默认、GPT-5-mini、GPT-5-nano、Claude-3.5-Haiku）与整合式网络搜索，**默认流模式**（约每 10 个字更新一次）
- **用户记忆**：以用户为单位记录对话记忆；`/clear_memory` 会清除你的用户记忆
- **图像处理**：视觉模型支持，自动图像格式转换
- **图像生成**：通过 Responses API 的工具整合，**支持渐进式流渲染**，生成的图片会以 Discord 附件 + Embed 的方式美化呈现，并显示部分进度
- **智能网络访问**：LLM 可于需要时自动搜索网络，提供最新信息

### 📊 内容处理

- **消息摘要**：智能频道对话摘要，支持用户筛选（5、10、20、50 条消息）
- **视频下载**：多平台支持（YouTube、TikTok、Instagram、X、Facebook），提供质量选项
    - Bilibili 兼容性改善：加入正确 Referer 标头、更安全的格式回退、与更稳健的错误处理
    - 网站专属标头：Referer 仅在 Bilibili 套用，以避免影响 Facebook 链接
- **枫之谷数据库**：查询怪物和物品详细掉落信息
- **竞标系统**：完整的拍卖平台与竞标功能，支持多货币类型（枫币/雪花/台币） - **已重构为模块化架构**，提升维护性
- **抽奖系统**：多平台抽奖功能，支持 Discord「按钮报名」或 YouTube 聊天室整合（完全不使用表情反应），可设定每次抽出人数并支持重新建立。已中奖者在同一活动期间会被自动排除，直到你使用「重新建立」开新活动。已统一采用 `discord` 命名，不再使用旧的 `reaction` 名称。参与者名单统一以单一字段显示。
    - 实现说明：报名/取消按钮以 `nextcord.ui.Button` 的子类别形式实现（`JoinLotteryButton`、`CancelJoinLotteryButton`），UI 仅处理互动；中奖与重复检查集中在核心新增/移除函数中以保持逻辑简洁。
- **图像生成**：已整合至 `/oai` 指令（使用 image_generation 工具）。独立的 `/graph` 仍为未来扩充的预留指令。

### 🌍 多语言支持

- 繁体中文
- 简体中文
- 日本語
- English

### 🔧 技术特色

- **主要机器人实现**：核心机器人类别 `DiscordBot` 在 `src/discordbot/cli.py` 中实现，继承 `nextcord.ext.commands.Bot` 并包含完整的初始化、Cog 加载和事件处理
- 模块化 Cog 架构设计
- 异步处理配合 nextcord
- Pydantic 基础配置管理
- 完整错误处理与日志记录
- Docker 支持与开发容器

## 🎯 核心指令

| 指令            | 功能说明         | 特色功能                                                                                                                                                                              |
| --------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/oai`          | 生成 AI 文字回应 | 多模型支持（默认 GPT-4o、另含 GPT-5 mini/nano、Claude 3.5 Haiku）、**默认流模式**（约每 10 个字更新）、图像输入、自动网络搜索与**渐进式图像生成**（Responses API 工具）、用户记忆 |
| `/clear_memory` | 清除对话记忆     | 重置你的用户记忆，让下次对话从头开始                                                                                                                                                |

| `/sum` | 互动式消息摘要 | 用户筛选、5/10/20/50 条消息选项 |
| `/download_video` | 多平台视频下载器 | 最佳/高/中/低质量；若超过 25MB 自动降为低画质 |
| `/maple_monster` | 搜索枫之谷怪物掉落 | 详细怪物信息 |
| `/maple_item` | 搜索枫之谷物品来源 | 掉落来源追踪 |
| `/maple_stats` | 枫之谷数据库统计 | 数据概览和热门物品 |
| `/auction_create` | 创建物品拍卖 | 互动表单、货币类型选择 |
| `/auction_list` | 浏览进行中拍卖 | 实时更新、下拉选单 |
| `/auction_info` | 查看拍卖详情 | 当前竞标、竞标历史 |
| `/auction_my` | 查看个人拍卖 | 我的拍卖与领先竞标 |
| `/lottery` | 抽奖主指令 | 下拉选择方式；按钮：🎉 报名、🚫 取消（Discord 模式）、✅ 开始（主持人）、📊 状态（仅自己可见）、🔄 重新建立、🔁 更新参与者（YouTube/仅主持人）；建立消息会自动更新统一的参与者名单；可设定每次抽出人数；同一活动已中奖者不再可加入（除非重新建立） |
| `/graph` | 生成图像（预留） | 框架已准备实现 |
| `/ping` | 机器人效能测试 | 延迟测量 |

## 🚀 快速开始

### 系统需求

- Python 3.10 或更高版本
- Discord 机器人 Token
- OpenAI API 密钥

### 安装步骤

1. **克隆项目**

    ```bash
    git clone https://github.com/Mai0313/discordbot.git
    cd discordbot
    ```

2. **使用 uv 安装依赖**

    ```bash
    # 如果尚未安装 uv
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # 安装项目依赖
    uv sync
    ```

3. **设定环境变量**

    ```bash
    cp .env.example .env
    # 编辑 .env 文件，填入你的 API 密钥和设定
    ```

4. **启动机器人**

    ```bash
    # 推荐（通过 entry point）
    uv run discordbot

    # 或
    uv run python -m discordbot.cli
    ```

### Docker 部署

```bash
# 使用 Docker Compose
docker-compose up -d

# 或手动建立
docker build -t discordbot .
docker run -d discordbot
```

注意：Docker 映像已安装 `ffmpeg`，以便 yt-dlp 可合并视频/音频流。

### 可选：更新枫之谷数据库

```bash
# 安装 Playwright Chromium（首次）
uv run playwright install chromium

# 抓取最新怪物/物品数据到 ./data/monsters.json
uv run update
```

## ⚙️ 配置设定

### 必要环境变量

```env
# Discord 设定
DISCORD_BOT_TOKEN=你的_discord_机器人_token
DISCORD_TEST_SERVER_ID=你的_测试_服务器_id  # 可选

# OpenAI 设定
OPENAI_API_KEY=你的_openai_api_密钥
OPENAI_BASE_URL=https://api.openai.com/v1

# Azure OpenAI（如果使用 Azure）
AZURE_OPENAI_API_KEY=你的_azure_密钥
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com


```

### 可选环境变量

```env
# 如果使用 Azure OpenAI
OPENAI_API_VERSION=2025-04-01-preview

# 消息记录（SQLite）
SQLITE_FILE_PATH=sqlite:///data/messages.db

# 其他服务（如有使用）
POSTGRES_URL=postgresql://postgres:postgres@postgres:5432/postgres
REDIS_URL=redis://redis:6379/0

# YouTube Data API 密钥（YouTube 抽奖模式需要）
YOUTUBE_DATA_API_KEY=你的_youtube_data_api_密钥
```

### YouTube 抽奖模式设定（可选）

- 在 `./data/client_secret.json` 放置 Google OAuth 凭证（桌面应用）。
- 设定 `YOUTUBE_DATA_API_KEY`。
- 第一次使用 YouTube 模式时会在 8080 端口开启浏览器进行授权，完成后凭证会储存至 `./data/token.pickle`。

## 📁 项目结构

```
src/discordbot/
├── cli.py              # 主要机器人入口点
├── cogs/               # 指令模块
│   ├── gen_reply.py    # AI 文字生成 (/oai)
│   ├── summary.py      # 消息摘要 (/sum)
│   ├── video.py        # 视频下载 (/download_video)
│   ├── maplestory.py   # 枫之谷数据库查询
│   ├── auction.py      # 竞标系统（重构模块化）
│   │   ├── models.py   # Pydantic 数据模型
│   │   ├── database.py # 数据库操作
│   │   ├── views.py    # UI 组件（Views, Modals, Buttons）
│   │   └── utils.py    # 工具函数与辅助功能
│   ├── lottery.py      # 多平台抽奖系统
│   ├── gen_image.py    # 图像生成（预留）
│   └── template.py     # 系统工具与延迟测试
├── sdk/                # 核心业务逻辑
│   ├── llm.py          # LLM 整合（OpenAI/Azure）
│   ├── log_message.py  # 消息记录（写入 SQLite）
│   └── yt_chat.py      # YouTube 聊天辅助
├── typings/            # 配置模型
│   ├── config.py       # Discord 设定
│   └── database.py     # DB 设定（SQLite/Postgres/Redis）
└── utils/              # 工具函数
    └── downloader.py   # yt-dlp 包装
data/
├── monsters.json       # 枫之谷怪物与掉落数据库
├── auctions.db         # 竞标系统 SQLite 数据库
└── downloads/          # 视频下载储存
```

## 🔍 核心功能深度解析

### 多模态 AI 支持

- **实时流**：文字回应采用实时流，每 10 个字更新一次，提供实时反馈
- **渐进式图片渲染**：图片生成显示部分进度更新，提供流畅的视觉体验
- 文字和图像输入处理
- 自动图像转 base64 格式
- 通过 Responses API 工具生成图像；输出以 Discord 文件附件与嵌入 (Embed) 呈现，提供优雅的预览效果
- 模型特定限制处理
- 自动网络搜索回应功能

### 视频下载引擎

- 支持 10+ 个平台
- 质量选择（4K 到纯音频）
- Discord 文件大小限制验证
- 进度追踪和错误处理
- 使用 Tenacity 的重试机制：重试 5 次，每次间隔 1 秒（写死设定）

#### Bilibili 使用注意

- Referer 仅在 Bilibili 套用；其他站（如 Facebook）使用最小标头。
- 请确保 `yt-dlp` 为最新版本。
- 需要安装 `ffmpeg` 以合并分离的视频/音频流（多数 B 站视频为分离流）。
- 下载器会附带 `Referer: https://www.bilibili.com` 并使用像 `bestvideo*+bestaudio/best` 的安全格式回退。
- 若仍出现「Requested format is not available」，可尝试选择较低画质（中/低）。部分视频仅提供特定 DASH 配置或受区域/年龄限制。
- 对于需登录/年龄/区域限制的视频，可能需要提供 cookies 给 yt-dlp（目前未预设接线）。

#### Facebook 使用注意

- 我们不会对 Facebook 强制加入 Referer；为避免与抽取器冲突，仅使用最小必要标头。
- 请维持 `yt-dlp` 为最新，并确认已安装 `ffmpeg`。
- 若下载失败，尝试较低画质；对于私密/登录/区域限制的链接，可能需要提供 cookies 给 yt-dlp。

### 枫之谷数据库

- 完整的怪物和物品数据库（192+ 个怪物）
- 支持模糊搜索的互动式查询
- 多语言支持（繁体中文、简体中文、日文、英文）
- 详细的怪物属性和掉落信息
- 物品来源追踪与可视化显示
- 缓存搜索结果以优化效能

### 竞标系统

- **重构模块化架构**：
    - **models.py**：Auction 和 Bid 实体的 Pydantic 数据模型，具备完整验证
    - **database.py**：AuctionDatabase 类别，提供完整 CRUD 操作、迁移支持和服务器隔离
    - **views.py**：UI 组件，包含 Views、Modals 和 Buttons 用于互动式拍卖管理
    - **utils.py**：工具函数，用于 embed 创建、验证和辅助操作
    - **auction.py**：主要 cog 实现，使用模块化组件
- **完整拍卖平台**：
    - 创建物品拍卖，可自订持续时间、竞标增额和货币类型选择（枫币/雪花/台币）
    - 多货币支持，默认为「枫币」，另提供「雪花」和「台币」选项
    - 实时竞标与互动界面（💰 出价、📊 查看记录、🔄 刷新）
    - 个人拍卖管理与竞标追踪，包含货币类型显示
    - 防止自我竞标和重复出价的安全机制
    - SQLite 数据库储存，具备 ACID 合规性和向后兼容性

### 抽奖系统

- **报名模式**：Discord 按钮报名或 YouTube 聊天室关键字报名
- **控制按钮**：🎉 报名、🚫 取消、✅ 开始（仅主持人）、📊 状态（仅本人可见）、🔄 重新建立、🔁 更新参与者（YouTube/仅主持人）
- **实时状态**：按 📊 取得仅自己可见的活动状态
- **自动更新消息**：建立消息会自动更新参与者名单（用户报名/取消时实时反映，统一于单一字段）
- **消息绑定**：按钮互动会绑定建立消息，机器人可藉此辨识对应的抽奖（完全不使用表情反应）
- **YouTube 模式名单抓取**：在 YouTube 模式下，主持人可随时按下 🔁 更新参与者 以依设定关键字从聊天室抓取参与者；同时在按下 ✅ 开始 之前也会再次抓取一次
- **中奖者排除**：在同一活动中，一旦被抽中，将不会再被加入该活动的参与名单（避免连续抽到同一人）。如需让所有人再次有资格，请使用 🔄 重新建立。
- **Discord 模式按钮**：🎉 报名 与 🚫 取消 仅在 Discord 模式显示
- **每次抽出人数**：建立表单可设定每次抽出的人数
- **重新建立（🔄）**：建立新的抽奖并沿用设定，并自动带回所有参与者（含先前中奖者）
- **安全机制**：仅主持人控制、加密安全的随机选择、参与者验证
- **单一平台**：每次抽奖只选择一种报名方式
- **内存储存**：轻量级 in-memory 结构，重启后重置

## 🛠️ 开发指南

### 本地开发

```bash
# 安装开发依赖
uv sync --dev

# 执行测试
uv sync --group test
uv run pytest -q

# 代码质量检查
uv run ruff check
uv run ruff format

# 建立文档
uv run mkdocs serve
```

### 🧪 测试说明

- 测试框架：`pytest`（含 `xdist` 并行化、`pytest-asyncio` 异步测试、覆盖率设定在 `pyproject.toml`）。
- 测试路径：所有测试位于 `tests/`，涵盖各个 cog 与核心工具。
- 新增的 Cog 单元测试包含：
    - `TemplateCogs`：消息反应与 `/ping` 延迟 Embed
    - `MessageFetcher`（摘要）：`_format_messages()` 与 `do_summarize()`（模拟 LLM）
    - `ReplyGeneratorCogs`：`_get_attachment_list()` 与 `/clear_memory`
    - `ImageGeneratorCogs`：`/graph`（预留流程）
    - `VideoCogs`：`/download_video` 乐观流程（模拟下载器）

执行完整测试并产生报表：

```bash
uv run pytest -q
# 覆盖率报表位置：./.github/reports 与 ./.github/coverage_html_report
```

### 贡献指南

1. Fork 此项目
2. 建立功能分支（`git checkout -b feature/新功能`）
3. 提交变更（`git commit -m '新增某项功能'`）
4. 推送到分支（`git push origin feature/新功能`）
5. 建立 Pull Request

### 代码规范

- 遵循 PEP 8 命名惯例
- 使用 Pydantic 模型进行数据验证
- 所有函数需要类型提示
- 使用 Google 风格的 docstring
- 最大行长度 99 字符

## 📚 API 参考

### 主要 SDK 模块

#### `src/sdk/llm.py`

```python
# AI 文字生成（范例代码已更新为新架构）
# 现在通过 Discord 的 /oai 指令使用

# AI 回应与自动网络搜索
# 现在整合在 /oai 指令中，LLM 会自动判断是否需要搜索网络

# 网络搜索功能已整合至 AI 回应中
# 无需单独调用，LLM 会自动处理
```

## 🚀 部署

### 生产环境部署

1. **环境准备**

    ```bash
    # 设定生产环境变量
    export DISCORD_BOT_TOKEN="生产环境token"
    export OPENAI_API_KEY="生产环境密钥"
    ```

2. **Docker 部署**

    ```bash
    docker-compose -f docker-compose.yaml up -d
    ```

3. **监控设定**

    - 使用 Logfire 进行日志监控
    - 设定健康检查端点
    - 配置错误通知

## 🔧 疑难排解

### 常见问题

**Q: 机器人无法回应指令**
A: 检查机器人权限，确保已启用「应用程序指令」范围

**Q: OpenAI API 错误**
A: 验证 API 密钥和额度，检查模型可用性

**Q: 视频下载失败**
A: 确认 yt-dlp 版本为最新，检查平台支持状况

**Q: 数据库连接错误**
A: 检查文件路径权限，确保目录存在

### 日志分析

```bash
# 检视机器人日志
tail -f logs/bot.log

# 检查错误日志
grep ERROR logs/bot.log
```

## 📈 效能优化

### 建议配置

- **内存**：最少 512MB，建议 1GB
- **储存空间**：最少 2GB（用于视频下载和数据储存）
- **网络**：稳定的互联网连接
- **CPU**：多核心处理器，支持大量并发请求

### 优化技巧

1. 使用 Redis 缓存频繁查询
2. 定期清理旧的下载文件
3. 配置适当的 API 请求限制
4. 使用连接池优化数据库连接

## 🔒 隐私与数据

本 Discord 机器人遵守 Discord 服务条款与开发者政策。

### 数据收集与使用

- **本地消息记录**：默认情况下，机器人在所在频道的消息会记录到本机 SQLite（`./data/messages.db`），包含作者、内容、时间戳与附件/贴图链接。数据仅存在你的服务器，不会外传。
- **不与第三方分享**：除了为完成请求所需的受信任 API（例如 OpenAI）之外，不会与第三方分享数据。
- **如何停用**：服务器拥有者可在 `src/discordbot/cli.py` 移除记录调用，或依需求调整 `src/discordbot/sdk/log_message.py`。

### 机器人权限与意图

本机器人仅功能需求申请以下权限：

- **消息内容意图**：用于斜线指令情境、少量关键字处理与上述本地记录（可调整）
- **斜线指令**：用于互动式指令处理
- **文件附件**：用于处理 AI 视觉功能中的图像和下载用户请求的内容
- **嵌入链接**：用于格式化丰富回应和搜索结果

### 数据安全

- 所有 API 通讯使用加密的 HTTPS 连接
- 不会将数据发送至任何外部服务。若启用本地消息记录，消息仅储存在你的磁盘（SQLite：`./data/messages.db`），不会外传。你可以在 `src/discordbot/cli.py` 移除记录调用或调整 `src/discordbot/sdk/log_message.py` 以停用。
- 不进行基于用户内容的长期分析。

### 联络与合规

如果您对隐私有疑虑或对数据处理有疑问：

- 通过 [GitHub Issues](https://github.com/Mai0313/discordbot/issues) 回报问题
- 通过项目储存库联络开发团队

本机器人采用隐私设计原则和最小化数据处理，以确保用户隐私保护。

## 📄 授权条款

本项目采用 MIT 授权条款。详细信息请参阅 [LICENSE](LICENSE) 文件。

## 👥 贡献者

[![Contributors](https://contrib.rocks/image?repo=Mai0313/discordbot)](https://github.com/Mai0313/discordbot/graphs/contributors)

使用 [contrib.rocks](https://contrib.rocks) 制作

## 📞 联络方式

- 📧 Email: [项目维护者邮箱]
- 💬 Discord: [Discord 服务器链接]
- 🐛 Issue: [GitHub Issues](https://github.com/Mai0313/discordbot/issues)
- 💡 讨论: [GitHub Discussions](https://github.com/Mai0313/discordbot/discussions)

## 🔗 相关资源

- [官方文档](https://mai0313.github.io/discordbot/)
- [Nextcord 文档](https://docs.nextcord.dev/)
- [OpenAI API 文档](https://platform.openai.com/docs)
- [Discord 开发者文档](https://discord.com/developers/docs)

---

<center>

**⭐ 如果这个项目对你有帮助，请给我们一个星星!**

</center>
