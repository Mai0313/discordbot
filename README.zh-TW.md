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

功能豐富的 Discord 機器人，具備 AI 智能對話、圖片與影片生成、內容解析、多平台影片下載、虛擬歡樂豆系統（每日簽到、可選 VIP、每日重置的貸款）、賭場小遊戲，以及楓之谷遊戲資料庫。支援多國語言。

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
- **回覆附註** — 每次 AI 回覆會在末端以 Discord 引用格式（`>`）顯示 model 名稱、input/output token 數量、預估 USD 費用（從上游 LiteLLM 的 price table 計算，首次查詢時下載並快取在本機），以及這一輪賺到的虛擬歡樂豆
- **自動解除 timeout** — 若有 moderator 把機器人 timeout，機器人會自動解除 timeout、從 audit log 識別出對方是誰，並在最近有人聊天的頻道回一句 AI 生成的回覆

### Threads 解析

貼上 Threads.net 連結，機器人會自動展開貼文 — 顯示文字內容、圖片、互動數據，並下載附帶的影片。若連結指向某則回覆，機器人會把整條回覆鏈一併展開，原始貼文與中間每一層回覆都會出現在同一則訊息裡，並用灰階漸層色帶（由淡到深）區分層級。只有使用者貼的那一篇會把影片下載並附上來，上層回覆若有影片則改用點擊連結提示，避免多層附件混在一起無法分辨。

### 影片下載

使用 `/download_video` 從多個平台下載影片：

- YouTube、TikTok、Instagram、X (Twitter)、Facebook、Bilibili
- 品質選項：最佳、高畫質 (1080p)、中等 (720p)、低畫質 (480p)
- 檔案超過 Discord 25 MB 限制時自動降為低畫質
- Facebook 分享連結（`facebook.com/share/r/...`）會自動展開

### 虛擬歡樂豆系統與賭場小遊戲

機器人會用本機 SQLite (`data/economy.db`) 持久保存每位 Discord 使用者的虛擬歡樂豆餘額，**虛擬歡樂豆跨伺服器共用**，同一個帳號在任何 guild 看到的餘額都一樣。

**獲得虛擬歡樂豆：** 每則非 bot 使用者訊息都會獲得 5,000 虛擬歡樂豆。AI 串流回覆會再追加以 `total_tokens` (input + output) 計算的 bonus，實際數字會顯示在回覆 footer。`/checkin` 每天可領 100,000 虛擬歡樂豆，連續 7 天為一個 cycle（線性加成：第 1 天 1×、第 7 天 4×）。Threads 解析與 `/download_video` 不會在基礎訊息獎勵之外再付額外 action reward。

**花用虛擬歡樂豆：** 賭場遊戲會先開 lobby。房主可以單人開始，也可以等其他玩家加入；只有房主可以開始 table。Blackjack bet 會在玩家加入時檢查，開始時重新確認餘額，等 table 結算時才套用本局正負結果。如果 Blackjack 房主輸入超過目前餘額的 bet，lobby 的 table stake 會 clamp 成房主實際 all-in 金額，後續玩家預設跟這個金額，不會被原本過大的輸入值強制全下；只有餘額為 0 或負數時才會拒絕玩家。射龍門則跑在**跨桌共用的全域彩金池**上 (`jackpot_pool` 中 `game_id="dragon_gate"` 的單一 row，首次啟動由系統種入 100,000 點 on the house)。入場費固定 5,000 (進場時直接撥進彩金池)，最低下注 10,000，下注上限就是當下彩金池總額，每一手下注的賠付即時寫入玩家帳戶與彩金池，不再等桌結束才結算。Dragon Gate loss 會 clamp 在餘額 0，彩金池只收到實際扣到的金額，歸零玩家會自動離桌，其他玩家繼續玩。任何玩家都可中途按「離桌」退出；若離桌或 timeout 時該玩家的桌內累計仍為正，那部分淨贏會逆向退回彩金池 (「逆贏不拿」)。桌結束的條件是彩金池被刷光、所有玩家都離桌或歸零、或 180 秒無人互動。莊家是個 AI，開局會嘴一下整桌下注，所有 Blackjack 玩家結束後會顯示思考中並決定莊家 hit / stand，結算時會依結果嘴或誇玩家。Embed 上「莊家」的顯示名稱直接用機器人自己的 Discord display name，所以未來 `gen_reply` 看歷史訊息時會把這些對白認作自己過去的發言，而不是某個無名 dealer。遊戲結算 footer 會顯示每位玩家的本局 delta、結算後餘額，以及 Blackjack 莊家的 decision path；`/house` 看的是 Blackjack 莊家的 ledger，射龍門的對手是彩金池而不是莊家，所以不會影響 `/house` 數字。

**VIP：** `/vip` 一次性花費 10,000,000 虛擬歡樂豆購買永久 VIP 標記。VIP 會獲得 1.5x Blackjack payout、2x 簽到基礎點數、2x 借款上限。

**管理員調整：** economy admin 會存在 `user_account.is_admin`，用 `uv run python scripts/manage_admin.py grant|revoke|list` 管理。admin 可以用 `/admin refund_tax` 退稅加點，也可以用 `/admin collect_tax` 收稅扣點；收稅會 clamp 在餘額 0，不會扣成負數。

可互動的 game/lobby message 使用 180 秒無互動 timeout，這裡的三分鐘是指沒有任何 component interaction，不是從開桌固定倒數。公開 response 進入不可互動狀態後，會在送出或結算三分鐘後自動刪除，包含賭場遊戲 final embed、餘額不足拒絕開局回覆、`/leaderboard`、`/loss_leaderboard`、`/house`、`/give` 轉帳結果 embed，以及 `/admin refund_tax` / `/admin collect_tax` 成功結果 embed。`/balance`、`/borrow`、`/repay`、`/checkin`、`/vip` 和 admin 錯誤回覆只有呼叫者看得到。遊戲 response 的 message ID 會存在本機，bot 重啟後會在下次 startup 刪掉上次留下的進行中或已結算遊戲 embed。

| Slash command       | 玩法                                                                                                                                                                                                                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `/blackjack <下注>` | 支援多人 21 點 lobby，含 Join / Leave / Start button。玩家動作包含 **Hit / Stand / Double Down / Split / Surrender**；只有莊家明牌 A 且可買 Insurance 時才會顯示保險 button。所有玩家結束後牌桌會先顯示莊家思考中，再由 AI 莊家決定 hit / stand（Soft 17 交由 AI 判斷）。            |
| `/dragon_gate`      | 支援多人射龍門 lobby, 跑**跨桌全域彩金池** (cross-table 累積)。入場費固定 5,000 進彩金池, 最低下注 10,000, 上限 = 當下彩金池, 每手即時結算; 玩家可中途按「離桌」退出 (逆贏不拿); 180 秒無互動 timeout 整桌結束。射進龍門從彩金池贏 1 倍；loss 最多扣到玩家目前餘額，歸零後自動離桌。 |

**21 點玩家動作：** Double Down 加倍下注後只再抽一張並強制 stand。Split 僅同 rank pair 可分；**Split 後禁止 Double**（No DAS），Split Aces 兩手各只能再拿一張，且 21 點算一般 1 倍贏（不是天生 Blackjack 1.5 倍）。Late Surrender 退回一半本金；莊家 peek 出 Blackjack 後就無法投降。

**21 點提前結算與 peek：** `Blackjack` 指的是起手兩張牌就是 A + 10 點牌，賠 1.5 倍。莊家會拿一張 hidden hole card 和一張 visible up-card；玩家操作期間牌桌會顯示成 `🂠 <明牌>`，明確告訴玩家有暗牌，但不 reveal 點數。莊家明牌 A 時會先進入 Insurance phase（保險注 = 原注一半，莊家 peek 到 Blackjack 賠 2:1），Insurance Yes / No button 只會在這個 phase 顯示；明牌 10 點則 silent peek。peek 命中 Blackjack 會直接結算整桌，玩家同樣也是天生 Blackjack 才平手。任意湊到 21 不算 natural Blackjack，不會跳過動作階段。

**管理虛擬歡樂豆：**

- `/balance` — private 查看自己的餘額、未還本金 (有的話) 與 VIP 狀態。
- `/checkin` — 領取今日簽到獎勵；回覆是 ephemeral 只有自己看得到。每天 00:00 Asia/Taipei 結算，連續 7 天為一 cycle，超過或漏簽會 reset 回第 1 天。
- `/vip` — private 花 10,000,000 虛擬歡樂豆購買永久 VIP。
- `/leaderboard` — 機器人所有伺服器的全域 Top 10（莊家自己的帳戶會被排除）。
- `/loss_leaderboard` — 今日 00:00 Asia/Taipei 之後賭場淨輸最多的全域 Top 10。
- `/borrow <金額>` — private 依 Discord 帳號年齡借虛擬歡樂豆，超過今日剩餘額度時會自動借到剩餘額度。**每天 00:00 Asia/Taipei 本金自動歸零**，沒有利息。
- `/repay <金額>` — private 從目前餘額償還未還本金。
- `/give <成員> <金額>` — 把虛擬歡樂豆轉給其他人（不能轉給自己或機器人）。
- `/house` — 查看莊家在賭場遊戲累積的輸贏。莊家資金無上限，所以 ledger balance 可以是負數（代表整體玩家從莊家手上贏走的虛擬歡樂豆比較多）。
- `/admin refund_tax|collect_tax` — admin-only 退稅加點/收稅扣點，admin 用 `scripts/manage_admin.py` 管理。

借款後，每次 income event（message reward / chat reward / 賭場 payout）會先自動拿 50% 還本金，剩下才進錢包。`/give` 的收款方不會被自動扣去還債。

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

| 指令                             | 說明                                                                          |
| -------------------------------- | ----------------------------------------------------------------------------- |
| `@bot <訊息>`                    | 與 AI 對話（文字、媒體/檔案、生成、摘要、網路搜尋）                           |
| _Threads 連結_                   | 自動展開 Threads.net 貼文與媒體                                               |
| `/download_video <網址> [品質]`  | 從 YouTube、TikTok、Instagram、X、Facebook、Bilibili 下載影片                 |
| `/balance`                       | private 查看你目前的虛擬歡樂豆餘額、貸款與 VIP 狀態（跨伺服器）               |
| `/checkin`                       | 領取今日簽到獎勵（ephemeral；7 天 streak 加成，每天 Taipei 00:00 重置）       |
| `/vip`                           | private 購買永久 VIP（1.5x Blackjack payout、2x 簽到、2x 借款上限）           |
| `/leaderboard`                   | 全域虛擬歡樂豆 Top 10                                                         |
| `/loss_leaderboard`              | 今日輸最多 Top 10（每天 Taipei 00:00 重置）                                   |
| `/borrow <金額>`                 | private 借虛擬歡樂豆；超過上限時自動借到今日剩餘額度                          |
| `/repay <金額>`                  | private 從餘額償還未還本金                                                    |
| `/give <成員> <虛擬歡樂豆>`      | 把虛擬歡樂豆轉給其他成員                                                      |
| `/admin refund_tax\|collect_tax` | admin-only 退稅加點/收稅扣點                                                  |
| `/blackjack <下注>`              | 開一個 21 點 lobby，含 Hit / Stand / Double / Split / Surrender 與莊家 A 保險 |
| `/dragon_gate`                   | 開一桌射龍門 lobby，跑跨桌全域彩金池 (loss clamp 到 0、含離桌按鈕)            |
| `/house`                         | 查看莊家在賭場遊戲累積的輸贏                                                  |
| `/maple_monster <名稱>`          | 搜尋楓之谷怪物與掉落物                                                        |
| `/maple_equip <名稱>`            | 搜尋楓之谷裝備                                                                |
| `/maple_scroll <名稱>`           | 搜尋楓之谷捲軸                                                                |
| `/maple_npc <名稱>`              | 搜尋楓之谷 NPC                                                                |
| `/maple_quest <名稱>`            | 搜尋楓之谷任務                                                                |
| `/maple_map <名稱>`              | 搜尋楓之谷地圖                                                                |
| `/maple_item <名稱>`             | 搜尋楓之谷物品來源                                                            |
| `/maple_stats`                   | 查看楓之谷資料庫統計                                                          |
| `/help`                          | 顯示機器人使用指南                                                            |
| `/ping`                          | 測試機器人延遲                                                                |

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
- **虛擬歡樂豆資料庫**：每位使用者的虛擬歡樂豆餘額儲存在另一個本機 SQLite 檔案 (`data/economy.db`)，會記錄 Discord user ID、最近一次看到的 username、avatar URL，以及餘額相關計數。餘額會跨機器人運行的所有伺服器共用。
- **遊戲清理資料庫**：等待清理的公開 game / economy response 只會把 Discord channel ID 與 message ID 存在 `data/game_cleanup.db`，讓 bot 重啟後能刪除殘留的 cleanup target。
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
