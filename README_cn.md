<center>

# AI 智能 Discord 機器人 🤖

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

[**English**](./README.md) | **繁體中文**

</center>

基於 **nextcord** 框架開發的全功能 Discord 機器人，提供 AI 智能對話、內容處理和實用工具功能。支援多語言介面與整合式網路搜尋功能。🚀⚡🔥

_歡迎提供建議和貢獻!_

## ✨ 主要功能

### 🤖 AI 智能互動

- **文字生成**：支援多種 AI 模型（GPT-5、GPT-5-mini、GPT-5-nano、Claude-3.5-Haiku）與整合式網路搜尋
- **圖像處理**：視覺模型支援，自動圖像格式轉換
- **智慧網路存取**：LLM 可於需要時自動搜尋網路，提供最新資訊

### 📊 內容處理

- **訊息摘要**：智能頻道對話摘要，支援用戶篩選（5、10、20、50 則訊息）
- **影片下載**：多平台支援（YouTube、TikTok、Instagram、X、Facebook），提供品質選項
- **楓之谷資料庫**：查詢怪物和物品詳細掉落資訊
- **競標系統**：完整的拍賣平台與競標功能，支援多貨幣類型（楓幣/雪花/台幣）
- **抽獎系統**：多平台抽獎功能，支援 Discord 反應與 YouTube 聊天室整合，具備動畫抽獎流程
- **圖像生成**：框架已準備（預留實現）

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

| 指令   | 功能說明         | 特色功能                                            |
| ------ | ---------------- | --------------------------------------------------- |
| `/oai` | 生成 AI 文字回應 | 多模型支援（GPT-5、Claude）、圖像輸入、自動網路搜尋 |

| `/sum` | 互動式訊息摘要 | 用戶篩選、5/10/20/50 則訊息選項 |
| `/download_video` | 多平台影片下載器 | 最佳/高/中/低品質；若超過 25MB 自動降為低畫質 |
| `/maple_monster` | 搜尋楓之谷怪物掉落 | 詳細怪物資訊 |
| `/maple_item` | 搜尋楓之谷物品來源 | 掉落來源追蹤 |
| `/maple_stats` | 楓之谷資料庫統計 | 資料概覽和熱門物品 |
| `/auction_create` | 創建物品拍賣 | 互動表單、貨幣類型選擇 |
| `/auction_list` | 瀏覽進行中拍賣 | 即時更新、下拉選單 |
| `/auction_info` | 查看拍賣詳情 | 當前競標、競標歷史 |
| `/auction_my` | 查看個人拍賣 | 我的拍賣與領先競標 |
| `/lottery` | 抽獎主指令 | 下拉選擇方式；主持人以 ✅ 開始，📊 狀態，🔄 重新建立抽獎；建立表單可設定每次抽出人數 |
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
│   ├── gen_reply.py    # AI 文字生成 (/oai)
│   ├── summary.py      # 訊息摘要 (/sum)
│   ├── video.py        # 影片下載 (/download_video)
│   ├── maplestory.py   # 楓之谷資料庫查詢
│   ├── auction.py      # 競標系統
│   ├── lottery.py      # 多平台抽獎系統
│   ├── gen_image.py    # 圖像生成（預留）
│   └── template.py     # 系統工具與延遲測試
├── sdk/                # 核心業務邏輯
│   ├── llm.py          # LLM 整合（OpenAI/Azure）
│   ├── log_message.py  # 訊息記錄（寫入 SQLite）
│   └── yt_chat.py      # YouTube 聊天輔助
├── typings/            # 配置模型
│   ├── config.py       # Discord 設定
│   └── database.py     # DB 設定（SQLite/Postgres/Redis）
└── utils/              # 工具函數
    └── downloader.py   # yt-dlp 包裝
data/
├── monsters.json       # 楓之谷怪物與掉落資料庫
├── auctions.db         # 競標系統 SQLite 資料庫
└── downloads/          # 影片下載儲存
```

## 🔍 核心功能深度解析

### 多模態 AI 支援

- 文字和圖像輸入處理
- 自動圖像轉 base64 格式
- 模型特定限制處理
- 自動網路搜尋回應功能

### 影片下載引擎

- 支援 10+ 個平台
- 品質選擇（4K 到純音訊）
- Discord 檔案大小限制驗證
- 進度追蹤和錯誤處理

### 楓之谷資料庫

- 完整的怪物和物品資料庫（192+ 個怪物）
- 支援模糊搜尋的互動式查詢
- 多語言支援（繁體中文、簡體中文、日文、英文）
- 詳細的怪物屬性和掉落資訊
- 物品來源追蹤與視覺化顯示
- 快取搜尋結果以優化效能

### 競標系統

- **完整拍賣平台**：
    - 創建物品拍賣，可自訂持續時間、競標增額和貨幣類型選擇（楓幣/雪花/台幣）
    - 多貨幣支援，預設為「楓幣」，另提供「雪花」和「台幣」選項
    - 即時競標與互動介面（💰 出價、📊 查看記錄、🔄 刷新）
    - 個人拍賣管理與競標追蹤，包含貨幣類型顯示
    - 防止自我競標和重複出價的安全機制
    - SQLite 資料庫儲存，具備 ACID 合規性和向後相容性

### 抽獎系統

- **雙重報名模式**：支援 Discord 反應式**或** YouTube 聊天室關鍵字報名（防止跨平台重複報名）
- **動畫抽獎流程**：互動式轉盤動畫，15步視覺效果確保透明公正的抽獎過程
- **每次抽出人數**：建立表單可設定每次抽出的人數；每次點選 ✅ 將一次抽出該人數
- **重新建立抽獎（🔄）**：僅主持人可用，會建立一個全新的抽獎，沿用原設定，並自動帶回之前所有參與者（包含先前中獎者）
- **全面狀態監控**：即時參與者追蹤，完整顯示所有參與者名單，跨平台統計分析，逗號分隔緊湊顯示
- **進階參與者管理**：自動透過 Discord 反應或 YouTube 聊天關鍵字註冊，具備重複防護和平台驗證
- **安全機制**：僅限發起人控制、密碼學安全隨機選擇、參與者驗證和自動移除中獎者
- **單一平台焦點**：每次抽獎只能使用一種報名方式，確保公平性並避免混亂
- **內存存儲**：輕量級全局變數配合 defaultdict 優化，運行時數據管理（機器人重啟時重置，確保全新開始）
- **互動 UI 元件**：創建表單、動畫抽獎界面、詳細狀態顯示和即時更新
- **更精簡的內部實作**：以更簡潔的記憶體結構與工具函式取代重複邏輯，在不改變功能的前提下降低維護成本

## 🛠️ 開發指南

### 本地開發

```bash
# 安裝開發依賴
uv sync --dev

# 執行測試
uv run pytest

# 程式碼品質檢查
uv run ruff check
uv run ruff format

# 建立文檔
uv run mkdocs serve
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

#### `src/sdk/llm.py`

```python
# AI 文字生成（範例代碼已更新為新架構）
# 現在透過 Discord 的 /oai 指令使用

# AI 回應與自動網路搜尋
# 現在整合在 /oai 指令中，LLM 會自動判斷是否需要搜尋網路

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
- **如何停用**：伺服器擁有者可在 `src/discordbot/cli.py` 移除記錄呼叫，或依需求調整 `src/discordbot/sdk/log_message.py`。

### 機器人權限與意圖

本機器人僅為功能需求申請以下權限：

- **訊息內容意圖**：用於斜線指令情境、少量關鍵字處理與上述本地記錄（可調整）
- **斜線指令**：用於互動式指令處理
- **檔案附件**：用於處理 AI 視覺功能中的圖像和下載用戶請求的內容
- **嵌入連結**：用於格式化豐富回應和搜尋結果

### 資料安全

- 所有 API 通訊使用加密的 HTTPS 連接
- 臨時資料處理在安全的短暫環境中進行
- 沒有用戶資料會在即時請求-回應週期之外持續存在

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
