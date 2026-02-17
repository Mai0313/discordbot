<div align="center" markdown="1">

# AI 智能 Discord 機器人 🤖

[![PyPI version](https://img.shields.io/pypi/v/swebenchv2.svg)](https://pypi.org/project/swebenchv2/)
[![python](https://img.shields.io/badge/-Python_%7C_3.11%7C_3.12%7C_3.13%7C_3.14-blue?logo=python&logoColor=white)](https://www.python.org/downloads/source/)
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

[**English**](./README.md) | **繁體中文** | [**简体中文**](./README.zh-CN.md)

</div>

基於 **nextcord** 框架開發的全功能 Discord 機器人，提供 AI 智能對話、內容處理和實用工具功能。支援多語言介面與整合式網路搜尋功能。🚀⚡🔥

_歡迎提供建議和貢獻!_

## ✨ 主要功能

### 🤖 AI 智能互動

- **標記聊天**：只需標記機器人（@機器人）就能開始對話 — 支援文字和圖片輸入，整合多種 AI 模型（OpenAI GPT-4o — 預設、GPT-5-mini、GPT-5-nano、Claude-3.5-Haiku）與**預設串流模式**（約每 10 個字更新一次）
- **圖像處理**：視覺模型支援，自動圖像格式轉換
- **智慧網路存取**：LLM 可於需要時自動搜尋網路，提供最新資訊

### 📊 內容處理

- **影片下載**：多平台支援（YouTube、TikTok、Instagram、X、Facebook），提供品質選項

    - Bilibili 相容性改善：加入正確 Referer 標頭、更安全的格式回退、與更穩健的錯誤處理
    - 網站專屬標頭：Referer 僅在 Bilibili 套用，以避免影響 Facebook 連結
    - Facebook 分享短連結（例如 `facebook.com/share/r/...`）會自動展開後再下載，你可以直接貼上 App 內複製的網址

- **楓之谷資料庫**：查詢怪物和物品詳細掉落資訊

### 🌍 多語言支援

- 繁體中文
- 簡體中文
- 日本語
- English

### 🔧 技術特色

- **主要機器人實現**：核心機器人類別 `DiscordBot` 在 `src/discordbot/cli.py` 中實現，繼承 `nextcord.ext.commands.Bot` 並包含完整的初始化、Cog 載入和事件處理
- 模組化 Cog 架構設計
- 非同步處理配合 nextcord
- Pydantic 基礎配置管理
- 完整錯誤處理與日誌記錄
- Docker 支援與開發容器

## 🎯 核心指令

| 指令    | 功能說明       | 特色功能                                                                                                                       |
| ------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `@標記` | 標記機器人聊天 | 多模型 AI（預設 GPT-4o、另含 GPT-5 mini/nano、Claude 3.5 Haiku）、**預設串流模式**（約每 10 個字更新）、圖像輸入、自動網路搜尋 |

```
                                                                                            |
```

| `/download_video` | 多平台影片下載器 | 最佳/高/中/低品質；若超過 25MB 自動降為低畫質 |
| `/maple_monster` | 搜尋楓之谷怪物掉落 | 詳細怪物資訊 |
| `/maple_item` | 搜尋楓之谷物品來源 | 掉落來源追蹤 |
| `/maple_stats` | 楓之谷資料庫統計 | 資料概覽和熱門物品 |

| `/graph` | 生成圖像（預留） | 框架已準備實現 |
| `/ping` | 機器人效能測試 | 延遲測量 |

## 🚀 快速開始

### 系統需求

- Python 3.10 或更高版本
- Discord 機器人 Token
- OpenAI API 金鑰

### 安裝步驟

1. **克隆專案**

    ```bash
    git clone https://github.com/Mai0313/discordbot.git
    cd discordbot
    ```

2. **使用 uv 安裝依賴**

    ```bash
    # 如果尚未安裝 uv
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # 安裝專案依賴
    uv sync
    ```

3. **設定環境變數**

    ```bash
    cp .env.example .env
    # 編輯 .env 檔案，填入你的 API 金鑰和設定
    ```

4. **啟動機器人**

    ```bash
    # 推薦（透過 entry point）
    uv run discordbot

    # 或
    uv run python -m discordbot.cli
    ```

### Docker 部署

```bash
# 使用 Docker Compose
docker-compose up -d

# 或手動建立
docker build -t discordbot .
docker run -d discordbot
```

注意：Docker 映像已安裝 `ffmpeg`，以便 yt-dlp 可合併視訊/音訊串流。

### 可選：更新楓之谷資料庫

```bash
# 安裝 Playwright Chromium（首次）
uv run playwright install chromium

# 抓取最新怪物/物品資料到 ./data/monsters.json
uv run update
```

## ⚙️ 配置設定

### 必要環境變數

```env
# Discord 設定
DISCORD_BOT_TOKEN=你的_discord_機器人_token
DISCORD_TEST_SERVER_ID=你的_測試_伺服器_id  # 可選

# OpenAI 設定
OPENAI_API_KEY=你的_openai_api_金鑰
OPENAI_BASE_URL=https://api.openai.com/v1

# Azure OpenAI（如果使用 Azure）
AZURE_OPENAI_API_KEY=你的_azure_金鑰
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com


```

### 可選環境變數

```env
# 如果使用 Azure OpenAI
OPENAI_API_VERSION=2025-04-01-preview

# 訊息記錄（SQLite）
SQLITE_FILE_PATH=sqlite:///data/messages.db

# 其他服務（如有使用）
POSTGRES_URL=postgresql://postgres:postgres@postgres:5432/postgres
REDIS_URL=redis://redis:6379/0


```

## 📁 專案結構

```
src/discordbot/
├── cli.py              # 主要機器人入口點
├── cogs/               # 指令模組
│   ├── gen_reply.py    # AI 文字生成 (@標記)
│   ├── video.py        # 影片下載 (/download_video)
│   ├── maplestory.py   # 楓之谷資料庫查詢
│   ├── log_msg.py      # 訊息記錄（寫入 SQLite）
│   └── template.py     # 系統工具與延遲測試
├── typings/            # 配置模型
│   ├── config.py       # Discord 設定
│   └── database.py     # DB 設定（SQLite/Postgres/Redis）
└── utils/              # 工具函數
    ├── downloader.py   # yt-dlp 包裝
    └── llm.py          # LLM 整合（OpenAI/Azure）
data/
├── monsters.json       # 楓之谷怪物與掉落資料庫
└── downloads/          # 影片下載儲存
```

## 🔍 核心功能深度解析

### 多模態 AI 支援

- **即時串流**：文字回應採用即時串流，每 10 個字更新一次，提供即時回饋
- 文字和圖像輸入處理
- 自動圖像轉 base64 格式
- 模型特定限制處理
- 自動網路搜尋回應功能

### 影片下載引擎

- 支援 10+ 個平台
- 品質選擇（4K 到純音訊）
- Discord 檔案大小限制驗證
- 進度追蹤和錯誤處理
- 使用 Tenacity 的重試機制：重試 5 次，每次間隔 1 秒（寫死設定）

#### Bilibili 使用注意

- Referer 僅在 Bilibili 套用；其他站（如 Facebook）使用最小標頭。
- 請確保 `yt-dlp` 為最新版本。
- 需要安裝 `ffmpeg` 以合併分離的視訊/音訊串流（多數 B 站影片為分離流）。
- 下載器會附帶 `Referer: https://www.bilibili.com` 並使用像 `bestvideo*+bestaudio/best` 的安全格式回退。
- 若仍出現「Requested format is not available」，可嘗試選擇較低畫質（中/低）。部分影片僅提供特定 DASH 配置或受區域/年齡限制。
- 對於需登入/年齡/區域限制的影片，可能需要提供 cookies 給 yt-dlp（目前未預設接線）。

#### Facebook 使用注意

- 我們不會對 Facebook 強制加入 Referer；為避免與抽取器衝突，僅使用最小必要標頭。
- `facebook.com/share/...` 的短連結會自動展開並轉成正確的 reel/watch 網址，免去了手動開啟瀏覽器的步驟。
- 請維持 `yt-dlp` 為最新，並確認已安裝 `ffmpeg`。
- 若下載失敗，嘗試較低畫質；對於私密/登入/區域限制的連結，可能需要提供 cookies 給 yt-dlp。

### 楓之谷資料庫

- 完整的怪物和物品資料庫（192+ 個怪物）
- 支援模糊搜尋的互動式查詢
- 多語言支援（繁體中文、簡體中文、日文、英文）
- 詳細的怪物屬性和掉落資訊
- 物品來源追蹤與視覺化顯示
- 快取搜尋結果以優化效能

## 🛠️ 開發指南

### 本地開發

```bash
# 安裝開發依賴
uv sync --dev

# 執行測試
uv sync --group test
uv run pytest -q

# 程式碼品質檢查
uv run ruff check
uv run ruff format

# 建立文檔
uv run mkdocs serve
```

### 🧪 測試說明

- 測試框架：`pytest`（含 `xdist` 平行化、`pytest-asyncio` 非同步測試、覆蓋率設定在 `pyproject.toml`）。
- 測試路徑：所有測試位於 `tests/`，涵蓋各個 cog 與核心工具。
- 新增的 Cog 單元測試包含：
    - `TemplateCogs`：訊息反應與 `/ping` 延遲 Embed
    - `ReplyGeneratorCogs`：`_get_attachment_list()` 與訊息處理
    - `VideoCogs`：`/download_video` 樂觀流程（模擬下載器）

執行完整測試並產生報表：

```bash
uv run pytest -q
# 覆蓋率報表位置：./.github/reports 與 ./.github/coverage_html_report
```

### 貢獻指南

1. Fork 此專案
2. 建立功能分支（`git checkout -b feature/新功能`）
3. 提交變更（`git commit -m '新增某項功能'`）
4. 推送到分支（`git push origin feature/新功能`）
5. 建立 Pull Request

### 程式碼規範

- 遵循 PEP 8 命名慣例
- 使用 Pydantic 模型進行資料驗證
- 所有函數需要型別提示
- 使用 Google 風格的 docstring
- 最大行長度 99 字元

## 📚 API 參考

### 主要 SDK 模組

#### `src/discordbot/utils/llm.py`

```python
# AI 文字生成（範例代碼已更新為新架構）
# 現在透過 Discord 的 @標記 功能使用

# AI 回應與自動網路搜尋
# 現在整合在 @標記 功能中，LLM 會自動判斷是否需要搜尋網路

# 網路搜尋功能已整合至 AI 回應中
# 無需單獨呼叫，LLM 會自動處理
```

## 🚀 部署

### 生產環境部署

1. **環境準備**

    ```bash
    # 設定生產環境變數
    export DISCORD_BOT_TOKEN="生產環境token"
    export OPENAI_API_KEY="生產環境金鑰"
    ```

2. **Docker 部署**

    ```bash
    docker-compose -f docker-compose.yaml up -d
    ```

3. **監控設定**

    - 使用 Logfire 進行日誌監控
    - 設定健康檢查端點
    - 配置錯誤通知

## 🔧 疑難排解

### 常見問題

**Q: 機器人無法回應指令**
A: 檢查機器人權限，確保已啟用「應用程式指令」範圍

**Q: OpenAI API 錯誤**
A: 驗證 API 金鑰和額度，檢查模型可用性

**Q: 影片下載失敗**
A: 確認 yt-dlp 版本為最新，檢查平台支援狀況

**Q: 資料庫連接錯誤**
A: 檢查檔案路徑權限，確保目錄存在

### 日誌分析

```bash
# 檢視機器人日誌
tail -f logs/bot.log

# 檢查錯誤日誌
grep ERROR logs/bot.log
```

## 📈 效能優化

### 建議配置

- **記憶體**：最少 512MB，建議 1GB
- **儲存空間**：最少 2GB（用於影片下載和資料儲存）
- **網路**：穩定的網際網路連接
- **CPU**：多核心處理器，支援大量並發請求

### 優化技巧

1. 使用 Redis 快取頻繁查詢
2. 定期清理舊的下載檔案
3. 配置適當的 API 請求限制
4. 使用連接池優化資料庫連接

## 🔒 隱私與資料

本 Discord 機器人遵守 Discord 服務條款與開發者政策。

### 資料收集與使用

- **本地訊息記錄**：預設情況下，機器人在所在頻道的訊息會記錄到本機 SQLite（`./data/messages.db`），包含作者、內容、時間戳與附件/貼圖連結。資料僅存在你的伺服器，不會外傳。
- **不與第三方分享**：除了為完成請求所需的受信任 API（例如 OpenAI）之外，不會與第三方分享資料。
- **如何停用**：伺服器擁有者可在 `src/discordbot/cli.py` 移除記錄呼叫，或依需求調整 `src/discordbot/cogs/log_msg.py`。

### 機器人權限與意圖

本機器人僅為功能需求申請以下權限：

- **訊息內容意圖**：用於斜線指令情境、少量關鍵字處理與上述本地記錄（可調整）
- **斜線指令**：用於互動式指令處理
- **檔案附件**：用於處理 AI 視覺功能中的圖像和下載用戶請求的內容
- **嵌入連結**：用於格式化豐富回應和搜尋結果

### 資料安全

- 所有 API 通訊使用加密的 HTTPS 連接
- 不會將資料發送至任何外部服務。若啟用本地訊息記錄，訊息僅儲存在你的磁碟（SQLite：`./data/messages.db`），不會外傳。你可以在 `src/discordbot/cli.py` 移除記錄呼叫或調整 `src/discordbot/cogs/log_msg.py` 以停用。
- 不進行基於用戶內容的長期分析。

### 聯絡與合規

如果您對隱私有疑慮或對資料處理有疑問：

- 透過 [GitHub Issues](https://github.com/Mai0313/discordbot/issues) 回報問題
- 透過專案儲存庫聯絡開發團隊

本機器人採用隱私設計原則和最小化資料處理，以確保用戶隱私保護。

## 📄 授權條款

本專案採用 MIT 授權條款。詳細資訊請參閱 [LICENSE](LICENSE) 檔案。

## 👥 貢獻者

[![Contributors](https://contrib.rocks/image?repo=Mai0313/discordbot)](https://github.com/Mai0313/discordbot/graphs/contributors)

使用 [contrib.rocks](https://contrib.rocks) 製作

## 📞 聯絡方式

- 📧 Email: [專案維護者郵箱]
- 💬 Discord: [Discord 伺服器連結]
- 🐛 Issue: [GitHub Issues](https://github.com/Mai0313/discordbot/issues)
- 💡 討論: [GitHub Discussions](https://github.com/Mai0313/discordbot/discussions)

## 🔗 相關資源

- [官方文檔](https://mai0313.github.io/discordbot/)
- [Nextcord 文檔](https://docs.nextcord.dev/)
- [OpenAI API 文檔](https://platform.openai.com/docs)
- [Discord 開發者文檔](https://discord.com/developers/docs)

---

<center>

**⭐ 如果這個專案對你有幫助，請給我們一個星星!**

</center>
