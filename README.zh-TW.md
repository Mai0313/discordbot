<div align="center" markdown="1">

# AI 智能 Discord 機器人

[![PyPI version](https://img.shields.io/pypi/v/swebenchv2.svg)](https://pypi.org/project/swebenchv2/)
[![python](https://img.shields.io/badge/-Python_%7C_3.12%7C_3.13%7C_3.14-blue?logo=python&logoColor=white)](https://www.python.org/downloads/source/)
[![uv](https://img.shields.io/badge/-uv_dependency_management-2C5F2D?logo=python&logoColor=white)](https://docs.astral.sh/uv/)
[![nextcord](https://img.shields.io/badge/-Nextcord-5865F2?logo=discord&logoColor=white)](https://github.com/nextcord/nextcord)
[![openai](https://img.shields.io/badge/-OpenAI-412991?logo=openai&logoColor=white)](https://openai.com)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://docs.pydantic.dev/latest/contributing/#badges)
[![tests](https://github.com/Mai0313/discordbot/actions/workflows/test.yml/badge.svg)](https://github.com/Mai0313/discordbot/actions/workflows/test.yml)
[![code-quality](https://github.com/Mai0313/discordbot/actions/workflows/code-quality-check.yml/badge.svg)](https://github.com/Mai0313/discordbot/actions/workflows/code-quality-check.yml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/Mai0313/discordbot/tree/main?tab=License-1-ov-file)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Mai0313/discordbot/pulls)
[![contributors](https://img.shields.io/github/contributors/Mai0313/discordbot.svg)](https://github.com/Mai0313/discordbot/graphs/contributors)

[**English**](./README.md) | **繁體中文** | [**简体中文**](./README.zh-CN.md)

</div>

一個自架 Discord bot，提供 AI chat、圖片與影片生成、Threads 連結展開、影片下載、虛擬歡樂豆與賭場小遊戲。它基於 nextcord 執行，用本機 SQLite 保存 runtime data，並連接 OpenAI-compatible LLM endpoint，例如 LiteLLM。

## 功能展示

tag bot 並問它會做什麼。這裡沒有 help 指令，它會讀自己的功能說明，並用你發問的語言回答。

![請 bot 自我介紹](assets/showcase-ai-chat.png)

叫它畫一張圖，它會直接生成，並以自己的語氣回覆。

![請 bot 生成圖片](assets/showcase-image-generation.png)

再叫它把同一張圖動起來，它會回傳一小段影片。

![請 bot 把生成的圖片變成影片](assets/showcase-video-generation.png)

## 功能

- **AI chat**：在 server tag bot 或傳送 DM。它可以回答問題、總結近期聊天、檢查支援的附件、觀看貼上的 YouTube 影片、生成或編輯圖片、用提示或附加圖片生成短影片、編輯引用的影片、以接續 reply 訊息延續長回覆，並在可用時使用 model-provided web tools。它還會在背景慢慢累積對你個人偏好的長期記憶（僅自己可見，且依來源做隱私隔離：在某個伺服器說的私事不會出現在別的伺服器，只有語氣偏好與明顯無害的一般事實會跨伺服器沿用），可用 `/memory show`、`/memory regenerate` 與 `/memory clear` 管理。
- **Threads 解析**：貼上 Threads.net 或 Threads.com URL，bot 會展開貼文、media 與 reply chain，引用別人或自己先前的貼文時也會一起帶出被引用的那篇；改成 tag bot 並附上連結，或是回覆別人貼連結的訊息時 tag bot，它會改為連底下的留言一起讀過再回答。
- **抖音解析**：貼上抖音連結，bot 會直接把影片（或圖文貼文的圖片）傳到頻道；改成 tag bot 並附上連結，它會改為看過影片再回答。
- **Bilibili 問答**：tag bot 並附上 B 站影片連結，它會看過影片再回答。單獨貼連結不會自動展開；`/download_video` 仍可下載檔案。
- **影片下載**：`/download_video` 可從 YouTube、TikTok、Instagram、X、Facebook、Bilibili，以及其他 yt-dlp 支援的網站下載影片。抖音也支援，無浮水印且包含圖文貼文。檔案太大無法上傳時會改以連結提供。
- **虛擬歡樂豆與金融系統**：使用者可從訊息獲得虛擬歡樂豆，可轉帳、購買 VIP、使用長期個人信貸或央行借款，並查看排行榜。
- **賭場遊戲**：多人 `/games blackjack` 與 `/games dragon_gate` lobby。Blackjack 莊家改為賭場系統 (deterministic H17)，bot 本身會以玩家身份入桌並由獨立的確定性策略 (fractional-Kelly 下注與 EV 決策) 決策，`/casino` 與 `/pocat` 分別顯示賭場帳本與 bot 玩家錢包。
- **問題回報**：`/feedback` 開啟只有本人看得到的面板，用單號列出自己回報過的問題，並提供表單把新的回報開成設定好的 repository 上的 GitHub issue（背景會由 LLM 整理成好讀的內容）。開發者在該 issue 上的回覆會出現在同一個面板裡，也有按鈕可以再補一句，讓回報是雙向的對話而不是單向信箱。
- **本地化指令**：slash command metadata 支援英文、繁體中文、日文。AI 回覆會跟隨使用者語言。沒有 help 指令：直接問 bot 會做什麼，它會讀一份英文的功能說明並用你發問的語言回答。

## 指令

| 指令                                        | 功能                                                                                   |
| ------------------------------------------- | -------------------------------------------------------------------------------------- |
| `@bot <message>`                            | 和 AI chat。需要 bot 檢查檔案或圖片時，可附上支援的附件。                              |
| _Threads URL_                               | 自動展開 Threads 貼文與 media；被 tag 時改為連留言一起讀過再回答。                     |
| _抖音 URL_                                  | 自動傳回影片或圖片；被 tag 時改為看過影片再回答。                                      |
| _Bilibili URL + tag_                        | 看過連結的影片後回答（單獨貼連結不會自動展開）。                                       |
| `/download_video <url> [quality]`           | 下載影片並傳回 Discord。抖音的圖文貼文會傳回圖片。                                     |
| `/feedback`                                 | 開啟只有你看得到的回報面板：用單號列出你的回報、開發者的回覆，以及開新回報的表單。     |
| `/balance [member]`                         | 私密顯示成員的虛擬歡樂豆餘額、債務、淨資產與 VIP 狀態。                                |
| `/vip`                                      | 購買永久 VIP 權益。                                                                    |
| `/leaderboard`                              | 顯示全域餘額排行榜。                                                                   |
| `/loss_leaderboard`                         | 顯示今日賭場輸局累計排行榜。                                                           |
| `/credit status\|borrow\|call\|repay`       | 處理個人信貸申請、180 秒批准/拒絕/取消按鈕、還款、催收與狀態。                         |
| `/central_bank status\|borrow\|call\|repay` | 處理央行借款申請、180 秒批准/拒絕/取消按鈕、還款、催收與可放貸額度。                   |
| `/give <member> <amount>`                   | 轉帳虛擬歡樂豆給其他成員或 bot。                                                       |
| `/admin refund_tax\|collect_tax`            | 手動調整成員或 bot 餘額；限定 `economy admin` 帳號 flag，不是 Discord 身分組。         |
| `/games blackjack <bet>`                    | 開一個多人 Blackjack lobby；`bet` 可輸入含逗號的數字，`0` 就是 all in。                |
| `/games dragon_gate`                        | 開一個由共享 jackpot pool 支撐的多人射龍門桌。                                         |
| `/casino`                                   | 顯示賭場系統累積 P&L (跨伺服器)。                                                      |
| `/pocat`                                    | 顯示 bot 玩家自己的錢包 (等同 `/balance @bot`)。                                       |
| `/memory show\|regenerate\|clear`           | 私密查看、重建或清除 bot 對你記住的內容（regenerate 在背景執行，clear 會先要求確認）。 |
| `/ping`                                     | 檢查 bot latency。                                                                     |

## 開發

Contributor setup、code conventions、tests 與 release notes 請見 [CONTRIBUTING.md](./.github/CONTRIBUTING.md)。

[文件](https://mai0313.github.io/discordbot/) | [回報問題](https://github.com/Mai0313/discordbot/issues) | [討論](https://github.com/Mai0313/discordbot/discussions)

[![Contributors](https://contrib.rocks/image?repo=Mai0313/discordbot)](https://github.com/Mai0313/discordbot/graphs/contributors)
