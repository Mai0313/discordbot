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

一個自架 Discord bot，提供 AI chat、圖片與影片生成、Threads 連結展開、影片下載、虛擬歡樂豆、賭場小遊戲，以及 MapleStory Artale 查詢。它基於 nextcord 執行，用本機 SQLite 保存 runtime data，並連接 OpenAI-compatible LLM endpoint，例如 LiteLLM。

## 功能

- **AI chat**：在 server tag bot 或傳送 DM。它可以回答問題、總結近期聊天、檢查支援的附件、觀看貼上的 YouTube 影片、生成或編輯圖片、生成短影片、以接續 reply 訊息延續長回覆，並在可用時使用 model-provided web tools。它還會在背景慢慢累積對你個人偏好的長期記憶（跨伺服器、僅自己可見），可用 `/memory show`、`/memory clear` 與 `/memory regenerate` 管理。
- **Threads 解析**：貼上 Threads.net 或 Threads.com URL，bot 會展開貼文、media 與 reply chain。
- **影片下載**：`/download_video` 可從 YouTube、TikTok、Instagram、X、Facebook、Bilibili，以及其他 yt-dlp 支援的網站下載影片，檔案太大時會自動 retry 低畫質。
- **虛擬歡樂豆與金融系統**：使用者可從訊息獲得虛擬歡樂豆，可每日簽到、轉帳、購買 VIP、使用長期個人信貸或央行借款，並查看排行榜。
- **模擬股市**：`/stock` 開啟一則公開 market message，內含 DB-managed virtual companies；選股、受 float supply、borrow cap 與單人 49% long holding cap 限制的交易、部位摘要、近期交易紀錄、liquidity-based slippage、定期刷新新聞與 7 日圖表都在同一則公開 message 內 edit 切換，只有發起 `/stock` 的 user 可以操作 controls。
- **賭場遊戲**：多人 `/games blackjack` 與 `/games dragon_gate` lobby。Blackjack 莊家改為賭場系統 (deterministic H17)，bot 本身會以玩家身份入桌並由獨立的確定性策略 (fractional-Kelly 下注與 EV 決策) 決策，`/casino` 與 `/pocat` 分別顯示賭場帳本與 bot 玩家錢包。單人 `/games fishing` 則是買釣具拋竿、回收歡樂豆的 sink 玩法，魚分 N 到 UR 稀有度並有最大單筆漁獲排行榜。
- **MapleStory Artale 資料庫**：`/maplestory` 子命令可查詢怪物、裝備、卷軸、NPC、任務、地圖、掉落來源與資料庫統計。
- **本地化指令**：slash command metadata 與 `/help` 支援英文、繁體中文、日文。AI 回覆會跟隨使用者語言。

## 指令

| 指令                                                             | 功能                                                                     |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `@bot <message>`                                                 | 和 AI chat。需要 bot 檢查檔案或圖片時，可附上支援的附件。                |
| _Threads URL_                                                    | 自動展開 Threads 貼文與 media。                                          |
| `/download_video <url> [quality]`                                | 下載影片並傳回 Discord。                                                 |
| `/balance [member]`                                              | 私密顯示成員的虛擬歡樂豆餘額、債務、stock holdings、淨資產與 VIP 狀態。  |
| `/checkin`                                                       | 領取每日簽到獎勵。                                                       |
| `/vip`                                                           | 購買永久 VIP 權益。                                                      |
| `/leaderboard`                                                   | 顯示全域餘額排行榜。                                                     |
| `/loss_leaderboard`                                              | 顯示今日賭場輸局累計排行榜。                                             |
| `/credit status\|borrow\|call\|repay`                            | 處理個人信貸申請、180 秒批准/拒絕/取消按鈕、還款、催收與狀態。           |
| `/central_bank status\|borrow\|call\|repay`                      | 處理央行借款申請、180 秒批准/拒絕/取消按鈕、還款、催收與可放貸額度。     |
| `/stock`                                                         | 公開股票市場訊息，明細、交易、新聞、紀錄都在同一則 message edit。        |
| `/give <member> <amount>`                                        | 轉帳虛擬歡樂豆給其他成員或 bot。                                         |
| `/admin refund_tax\|collect_tax`                                 | admin-only 手動調整成員或 bot 餘額。                                     |
| `/games blackjack <bet>`                                         | 開一個多人 Blackjack lobby；`bet` 可輸入含逗號的數字，`0` 就是 all in。  |
| `/games dragon_gate`                                             | 開一個由共享 jackpot pool 支撐的多人射龍門桌。                           |
| `/games fishing`                                                 | 打開個人釣魚面板，買釣竿與魚餌拋竿，是回收歡樂豆的 sink 玩法。           |
| `/casino`                                                        | 顯示賭場系統累積 P&L (跨伺服器)。                                        |
| `/pocat`                                                         | 顯示 bot 玩家自己的錢包 (等同 `/balance @bot`)。                         |
| `/maplestory monster`, `/maplestory equip`, `/maplestory scroll` | 查詢 MapleStory Artale 怪物、裝備與卷軸。                                |
| `/maplestory npc`, `/maplestory quest`, `/maplestory map`        | 查詢 NPC、任務與地圖。                                                   |
| `/maplestory item`, `/maplestory stats`                          | 查詢物品掉落來源與資料庫統計。                                           |
| `/memory show\|clear\|regenerate`                                | 私密查看、清除或重建 bot 對你記住的內容（regenerate 會排程在背景執行）。 |
| `/help`                                                          | 顯示 Discord 內的使用指南。                                              |
| `/ping`                                                          | 檢查 bot latency。                                                       |

## 自架

### 前置需求

- Python 3.12 或更新版本
- 來自 [Discord Developer Portal](https://discord.com/developers/applications) 的 Discord bot token
- OpenAI-compatible API key 與 base URL。若想把 OpenAI、Gemini、Claude 和其他 provider 放在同一個 endpoint 後面，建議使用 LiteLLM。
- 影片下載需要 `ffmpeg`，產生 board 圖片需要支援 CJK 的 fonts。Docker image 已經內建兩者。

### Docker

```bash
git clone https://github.com/Mai0313/discordbot.git
cd discordbot
cp .env.example .env
# edit .env
mkdir -p data
docker compose up -d
```

容器以 UID 1000 執行,讓 `data/` 下的檔案維持由你的主機使用者擁有;如果你的主機使用者 UID 不是 1000,請在 `docker-compose.yaml` 中覆寫 `user:`。

### 本機

```bash
git clone https://github.com/Mai0313/discordbot.git
cd discordbot
uv sync
cp .env.example .env
# edit .env
uv run discordbot
```

刷新內建的 MapleStory Artale data：

```bash
uv run python scripts/artale_data.py
```

## 設定

從 `.env.example` 建立 `.env`，並設定必要值：

```env
DISCORD_BOT_TOKEN=your_discord_bot_token
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
```

`OPENAI_BASE_URL` 可以直接指向 OpenAI，也可以指向 LiteLLM 這類 OpenAI-compatible gateway。

本機測試央行批准流程時，可以設定 `ECONOMY_ALLOW_CENTRAL_BANK_SELF_APPROVAL=true`。正式環境請保持未設定或 `false`。

個人長期記憶永遠開啟；使用者可以用 `/memory show`、`/memory clear` 與 `/memory regenerate` 管理自己的記憶。

## 資料與隱私

這個 bot 會把 runtime data 存在本機 `data/` 底下；SQLite 資料庫集中在 `data/database/`。

- `database/messages.db`：human messages 與 bot 自己的回覆，用於聊天歷史與摘要。
- `database/economy.db`：`user_wallet` 存每位使用者的可用餘額與 gross totals，`user_account` 存 cached Discord account name / avatar URL、VIP、admin、央行成員、簽到與 leaderboard flags，另存長期信貸申請與契約、賭場每日統計，以及 bot-wide jackpot pool 與 casino ledger。
- `database/stock.db`：DB-managed 模擬 stock profile、float supply、price tick、position、trade operation、ordered trade leg 與 AI-or-fallback stock news。
- `database/games.db`：每位玩家的 Blackjack 對局歷史、釣魚目錄與每位使用者的裝備、魚餌與漁獲紀錄，以及公開 expiring response 的清理追蹤（guild/channel 名稱、user name、channel ID 與 message ID），用於 bot 重啟後的清理。
- 臨時 media 下載使用專案根目錄的 `tmp/` scratch folder（不在 `data/` 底下），傳送完成後即刪除。
- `memories/`：每個 Discord user id 一個資料夾的純文字 markdown 個人長期記憶，由你的對話在背景累積，並在後續 AI 回覆時注入。

當 bot 需要用 AI 回覆時，當前上下文中的相關文字、支援的附件、embedded media 與參與者身份會送到你設定的 LLM endpoint。本專案不會把這些資料送到其他服務。

## 故障排除

- **Slash commands 沒出現**：確認邀請連結包含 `applications.commands`。Global command propagation 可能需要一些時間，尤其是新增指令。
- **AI 回覆失敗**：檢查 `OPENAI_API_KEY`、`OPENAI_BASE_URL`，以及 cogs 中設定的 model routing。
- **影片下載失敗**：更新 `yt-dlp`，並確認已安裝 `ffmpeg`。也可以嘗試較低的 quality。
- **權限錯誤**：mention-based chat 與本機訊息記錄需要 Message Content intent，embed、attachment、reaction 與 slash command 也需要一般 Discord 權限。

## 開發

Contributor setup、code conventions、tests 與 release notes 請見 [CONTRIBUTING.md](./CONTRIBUTING.md)。

[文件](https://mai0313.github.io/discordbot/) | [回報問題](https://github.com/Mai0313/discordbot/issues) | [討論](https://github.com/Mai0313/discordbot/discussions)

[![Contributors](https://contrib.rocks/image?repo=Mai0313/discordbot)](https://github.com/Mai0313/discordbot/graphs/contributors)
