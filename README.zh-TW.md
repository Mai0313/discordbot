<div align="center" markdown="1">

# AI 智能 Discord 機器人

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

[**English**](./README.md) | **繁體中文** | [**简体中文**](./README.zh-CN.md)

</div>

功能豐富的 Discord 機器人，具備 AI 智能對話、圖片與影片生成、內容解析、多平台影片下載、點數系統與賭場小遊戲，以及楓之谷遊戲資料庫。支援多國語言。

## 功能

### AI 聊天

標記機器人（`@bot`）或傳送私訊即可開始對話。AI 後端是任何 OpenAI 相容端點（通常會搭配 [LiteLLM](https://github.com/BerriAI/litellm) proxy 對接 OpenAI、Google Gemini、Anthropic Claude 等），機器人會依任務切換模型 — 用快速 model 做意圖分類與圖片 caption、慢速推理 model 做對話與摘要、專用 image model 做圖片生成/編輯、video model 做短片生成。支援功能：

- **文字對話** — 透過 OpenAI Responses API 串流即時回應
- **多媒體理解** — 附上圖片、貼圖，或 PDF、文字、JSON 等支援的檔案即可向機器人提問；也能讀取訊息或引用回覆中內嵌的圖片（例如已解析的 Threads 貼文 embed）
- **圖片生成與編輯** — 請機器人根據文字描述繪製或創作圖片（附上圖片即可進行編輯修改）
- **影片生成** — 請機器人生成短影片（請求之間有冷卻時間）
- **聊天摘要** — 請機器人總結近期對話內容
- **網路搜尋與 URL 讀取** — 機器人會自動使用模型對應的工具（Gemini 的 `googleSearch` + `urlContext`、Claude 的 `web_search` + `web_fetch`，或 OpenAI 的 `web_search`）取得最新資訊
- **使用者標記** — 請機器人通知或轉告近期對話中的其他成員（例如「幫我跟 @alice 說我會晚到」），只要該成員曾出現在近期聊天紀錄中即可被標記
- **進度反應** — 以 emoji reaction 顯示即時處理狀態（🤔 → 🔀 → 🎨/🎬/📖/❓ → 🆗，模型有用到 web search 時加 🌐，錯誤時顯示 ❌）
- **回覆附註** — 每次 AI 回覆會在末端以 Discord 引用格式（`>`）顯示 model 名稱、input/output token 數量、預估 USD 費用（從上游 LiteLLM 的 price table 計算，首次查詢時下載並快取在本機），以及這一輪賺到的點數
- **自動解除 timeout** — 若有 moderator 把機器人 timeout，機器人會自動解除 timeout、從 audit log 識別出對方是誰，並在最近有人聊天的頻道回一句 AI 生成的回覆

### Threads 解析

貼上 Threads.net 連結，機器人會自動展開貼文 — 顯示文字內容、圖片、互動數據，並下載附帶的影片。若連結指向某則回覆，機器人會把整條回覆鏈一併展開，原始貼文與中間每一層回覆都會出現在同一則訊息裡，並用灰階漸層色帶（由淡到深）區分層級。只有使用者貼的那一篇會把影片下載並附上來，上層回覆若有影片則改用點擊連結提示，避免多層附件混在一起無法分辨。

### 影片下載

使用 `/download_video` 從多個平台下載影片：

- YouTube、TikTok、Instagram、X (Twitter)、Facebook、Bilibili
- 品質選項：最佳、高畫質 (1080p)、中等 (720p)、低畫質 (480p)
- 檔案超過 Discord 25 MB 限制時自動降為低畫質
- Facebook 分享連結（`facebook.com/share/r/...`）會自動展開

### 點數系統與賭場小遊戲

機器人會用本機 SQLite (`data/economy.db`) 持久保存每位 Discord 使用者的點數餘額，**點數跨伺服器共用**，同一個帳號在任何 guild 看到的餘額都一樣。

**獲得點數：** 每次 AI 串流回覆會以 `total_tokens` (input + output) 為點數獎勵，實際數字會顯示在回覆 footer。目前只有跟機器人聊天會給點數 — Threads 解析與 `/download_video` 刻意都不付點。

**花用點數：** 賭場遊戲會先扣下注額，結算時再依結果回補。莊家是個 AI，開局會嘴一下注金額，結算時會依結果嘴或誇玩家。Embed 上「莊家」的顯示名稱直接用機器人自己的 Discord display name，所以未來 `gen_reply` 看歷史訊息時會把這些對白認作自己過去的發言，而不是某個無名 dealer。

| Slash command       | 玩法                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| `/dice <下注>`      | 三顆骰子 vs 三顆骰子，總和大者贏（平手退錢）。                                                  |
| `/blackjack <下注>` | 標準 21 點，附 Hit / Stand button；天生 Blackjack 賠 1.5 倍；玩家每次 hit 莊家會給帶刺的 hint。 |

**21 點提前結算規則：** `Blackjack` 指的是起手兩張牌就是 A + 10 點牌。玩家起手 Blackjack 會直接贏並賠 1.5 倍；莊家起手 Blackjack 會直接結算，除非玩家同時也是 Blackjack，否則玩家輸。這不是任意湊到 21 點都會提前結束，只有起手 natural Blackjack 才會跳過 Hit / Stand。

**管理點數：**

- `/balance` — 查看自己的餘額。
- `/leaderboard` — 機器人所有伺服器的全域 Top 10（莊家自己的帳戶會被排除）。
- `/give <成員> <金額>` — 把點數轉給其他人（不能轉給自己或機器人）。
- `/house` — 查看莊家在 `/dice` 與 `/blackjack` 累積的輸贏。莊家資金無上限，所以餘額可以是負數（代表整體玩家從莊家手上贏走的點數比較多）。

### 楓之谷 Artale 資料庫

- `/maple_monster` — 依名稱搜尋怪物，查看屬性、出沒地圖與掉落物
- `/maple_equip` — 依名稱搜尋裝備，查看屬性與取得方式
- `/maple_scroll` — 依名稱搜尋捲軸與附加屬性
- `/maple_npc` — 依名稱搜尋 NPC 與所在位置
- `/maple_quest` — 依名稱搜尋任務、等級範圍與頻率
- `/maple_map` — 依名稱搜尋地圖、區域與出沒怪物
- `/maple_item` — 搜尋物品並查看哪些怪物會掉落
- `/maple_stats` — 查看資料庫統計資訊
- 支援模糊搜尋與多語言顯示

### 多語言支援

Slash command 的名稱、描述，以及 `/help` 使用指南目前支援英文、繁體中文 (`zh-TW`) 與日文 (`ja`)。AI 對話回覆則會跟隨使用者輸入的語言。

## 指令

| 指令                            | 說明                                                                       |
| ------------------------------- | -------------------------------------------------------------------------- |
| `@bot <訊息>`                   | 與 AI 對話（文字、媒體/檔案、生成、摘要、網路搜尋）                        |
| _Threads 連結_                  | 自動展開 Threads.net 貼文與媒體                                            |
| `/download_video <網址> [品質]` | 從 YouTube、TikTok、Instagram、X、Facebook、Bilibili 下載影片              |
| `/balance`                      | 查看你目前的點數餘額（跨伺服器）                                           |
| `/leaderboard`                  | 全域點數 Top 10                                                            |
| `/give <成員> <點數>`           | 把點數轉給其他成員                                                         |
| `/dice <下注>`                  | 跟 AI 莊家擲三顆骰子比大小                                                 |
| `/blackjack <下注>`             | 跟 AI 莊家玩一局 21 點（含 Hit / Stand button；起手 Blackjack 會直接結算） |
| `/house`                        | 查看莊家在 `/dice` 與 `/blackjack` 累積的輸贏                              |
| `/maple_monster <名稱>`         | 搜尋楓之谷怪物與掉落物                                                     |
| `/maple_equip <名稱>`           | 搜尋楓之谷裝備                                                             |
| `/maple_scroll <名稱>`          | 搜尋楓之谷捲軸                                                             |
| `/maple_npc <名稱>`             | 搜尋楓之谷 NPC                                                             |
| `/maple_quest <名稱>`           | 搜尋楓之谷任務                                                             |
| `/maple_map <名稱>`             | 搜尋楓之谷地圖                                                             |
| `/maple_item <名稱>`            | 搜尋楓之谷物品來源                                                         |
| `/maple_stats`                  | 查看楓之谷資料庫統計                                                       |
| `/help`                         | 顯示機器人使用指南                                                         |
| `/ping`                         | 測試機器人延遲                                                             |

## 自架設

### 前置需求

- Python 3.12+
- Discord 機器人 Token（[開發者入口](https://discord.com/developers/applications)）
- OpenAI 相容端點與 API 金鑰 — 可以是單一 provider（OpenAI、Gemini 透過 OpenAI 相容端點等），或是 [LiteLLM](https://github.com/BerriAI/litellm) proxy 對接多個 provider

### 方式一：Docker（推薦）

```bash
git clone https://github.com/Mai0313/discordbot.git
cd discordbot
cp .env.example .env
# 編輯 .env 填入你的 Token 和 API 金鑰
docker-compose up -d
```

Docker 映像已包含 `ffmpeg`，可處理影片/音訊串流合併。

### 方式二：本機安裝

```bash
git clone https://github.com/Mai0313/discordbot.git
cd discordbot

# 安裝 uv（Python 套件管理器）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安裝依賴
uv sync

# 設定環境
cp .env.example .env
# 編輯 .env 填入你的 Token 和 API 金鑰

# 啟動機器人
uv run discordbot
```

### 可選：更新楓之谷 Artale 資料庫

```bash
uv run python scripts/artale_data.py
```

此 script 會從 `artalemaplestory.com` 抓取資料，並將 JSON 檔案寫入 `data/maplestory/`。

## 設定

建立 `.env` 檔案（或從 `.env.example` 複製）：

```env
# 必要
DISCORD_BOT_TOKEN=你的機器人token
OPENAI_API_KEY=你的api金鑰
OPENAI_BASE_URL=https://api.openai.com/v1   # 或任何 OpenAI 相容端點

# 可選
DISCORD_TEST_SERVER_ID=你的測試伺服器id
```

### Slash Command Sync

所有 slash command 都走 Discord 全域註冊（沒有 per-guild pin）。第一次新增指令時 Discord 端 propagate 最多需要 1 小時；後續編輯既有指令通常幾分鐘內就會生效。如果新指令在你的 client 上沒立刻出現，先試試 `Ctrl+R` 重整，或稍等一下。

## 各平台注意事項

### Bilibili

- 需要 `ffmpeg` 來合併分離的影片/音訊串流（Docker 映像已內建）。
- 若出現「Requested format is not available」，請嘗試較低的畫質設定。
- 區域/年齡限制的影片可能需要提供 cookies（預設未設定）。

### Facebook

- 分享連結（`facebook.com/share/...`）會在下載前自動展開。
- 請保持 `yt-dlp` 為最新版本以獲得最佳相容性。

## 隱私與資料

本機器人遵守 Discord 服務條款與開發者政策。

- **訊息記錄**：機器人所在頻道的訊息會記錄到本機 SQLite (`data/messages.db`)。資料僅存在你的伺服器，不會外傳。
- **點數資料庫**：每位使用者的點數餘額儲存在另一個本機 SQLite 檔案 (`data/economy.db`)，只記錄 Discord user ID、最近一次看到的 username，以及餘額相關計數。餘額會跨機器人運行的所有伺服器共用。
- **API 呼叫**：文字、圖片、支援的檔案附件、內嵌媒體，以及發送者身份（目前對話上下文中參與者的 display name、username 與 Discord user ID）僅在機器人需要回覆時才會發送至設定的 LLM API，例如在 guild 被標記或收到 DM 時。user ID 會一併傳入，讓機器人在被要求時可以標記其他成員。不會與其他第三方分享資料。
- **權限**：機器人需要 Message Content 意圖用於標記聊天和可選的本地記錄。斜線指令與嵌入/附件權限用於互動功能。
- **停用**：伺服器擁有者可透過調整機器人設定來停用訊息記錄。

## 常見問題

**機器人不回應指令？**
檢查機器人權限，確保已啟用 `applications.commands` 範圍。

**影片下載失敗？**
確認 `yt-dlp` 和 `ffmpeg` 為最新版本。嘗試較低的畫質設定。

**API 錯誤？**
驗證 API 金鑰並確認端點 URL 正確。

---

想要貢獻？請參閱 [CONTRIBUTING.md](./CONTRIBUTING.md)。

[![Contributors](https://contrib.rocks/image?repo=Mai0313/discordbot)](https://github.com/Mai0313/discordbot/graphs/contributors)

[文件](https://mai0313.github.io/discordbot/) | [回報問題](https://github.com/Mai0313/discordbot/issues) | [討論](https://github.com/Mai0313/discordbot/discussions)
