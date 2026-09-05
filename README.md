<div align="center" markdown="1">

# AI-Powered Discord Bot

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

**English** | [**繁體中文**](./README.zh-TW.md) | [**简体中文**](./README.zh-CN.md)

</div>

A self-hosted Discord bot for AI chat, image and video generation, Threads link expansion, video downloads, virtual currency, and casino mini-games. It runs on nextcord, stores runtime data in local SQLite files, and talks to an OpenAI-compatible LLM endpoint such as LiteLLM.

## Showcase

Mention the bot and ask what it can do. There is no help command; it answers from its own capability reference, in the language you asked in.

![Asking the bot to introduce itself](assets/showcase-ai-chat.png)

Ask for a picture and it generates one, then replies about it in character.

![Asking the bot to generate an image](assets/showcase-image-generation.png)

Ask it to animate that same picture and it returns a short video.

![Asking the bot to turn the generated image into a video](assets/showcase-video-generation.png)

## How a reply happens

Every mention, DM, and `/ask` runs the same pipeline. Exactly one triage call sits on the critical path and picks the route; everything else is either prepared in parallel before that decision lands or runs after the answer text is already on screen.

```mermaid
flowchart TD
    U(["@mention · DM · /ask"]) --> UP & RT & CX

    UP["Upload attachments"]
    RT["Route + effort"]
    CX["History · memory · tone"]

    UP & RT & CX --> R{"Route"}

    R -->|QA| L{"Linked Threads,<br/>Douyin or Bilibili post"}
    L -->|"the user is asking about it"| LF["Fetch the post, upload its media"]
    L -->|"incidental link"| LS["Nothing fetched"]
    LF & LS --> Y{"Linked YouTube video"}
    Y -->|"the user is asking about it"| YT["Interactions stream:<br/>Gemini watches the video"]
    Y -->|"otherwise"| QS["Responses stream"]
    YT & QS --> Q2["Inline markers:<br/>voice · image · music · video<br/>deep research · memory"]

    R -->|IMAGE| I1["Refine the prompt, render or edit the image"]
    I1 --> I2["Send it, then reply in character"]
    R -->|VIDEO| V1["Render or edit the clip"]
    V1 --> V2["Send it, watch it, then reply"]

    Q2 --> OUT(["Reply on screen"])
    I2 --> OUT
    V2 --> OUT
    OUT -.-> MM["Background: memory review"]

    classDef proxy fill:#12607a,stroke:#12607a,color:#ffffff
    classDef direct fill:#a8481b,stroke:#a8481b,color:#ffffff
    class RT,QS,I1,I2,MM proxy
    class UP,LF,YT,V1 direct
```

Blue steps run on the OpenAI-compatible proxy; orange ones call Google directly, which is what the Gemini Files API, watching a YouTube video, and native video and music generation each require.

The two content branches cost nothing when they do not apply. A linked post is fetched only when the router judges the user is asking about it, so an incidental link downloads nothing at all; and watching a YouTube video is the one thing that moves an answer turn onto the direct path, because the proxy fetches the link as a web page and the model never sees the footage. Everything after the answer text lands is best-effort: a clip that fails to render leaves the reply standing and adds only a small hint.

## Features

- **AI chat**: mention the bot in a server or send a DM. It can answer questions, summarize recent chat, inspect supported attachments, watch a linked YouTube video, generate or edit images, generate short videos from a prompt or attached images, edit a referenced video, continue long replies as follow-up reply messages, and use model-provided web tools when available. It also builds a private per-user long-term memory of your preferences in the background — privacy-scoped by source, so something told in one server never surfaces in another (only your tone preferences and clearly harmless general facts carry over) — manageable with `/memory show`, `/memory regenerate`, and `/memory clear`. You can also just ask it to remember something, or tell it that something it remembers is wrong, and a short note under its reply says what it took down.
- **Threads parser**: paste a Threads.net or Threads.com URL and the bot expands the post, media, and reply chain, plus the post it quotes when it is a quote post. Mention the bot alongside the link instead, or mention it in a reply to a message carrying one, and it reads the post together with the comments under it and answers about it.
- **Douyin parser**: paste a Douyin link and the bot posts the video (or the photo post's images) straight into the channel. Mention the bot alongside the link instead and it watches the clip and answers about it.
- **Bilibili Q&A**: mention the bot with a Bilibili video link and it watches the video and answers about it. A bare link is not auto-expanded; `/download_video` still downloads the file.
- **Video downloader**: `/download_video` downloads videos from YouTube, TikTok, Instagram, X, Facebook, Bilibili, and other yt-dlp supported sites. Douyin is supported too, watermark free and including photo posts. Files too large to upload are served as a link instead.
- **Virtual currency and finance**: users earn 虛擬歡樂豆 from messages, can transfer balances, buy VIP, use long-term personal credit or central-bank loans, and view leaderboards.
- **Casino games**: multiplayer `/games blackjack` and `/games dragon_gate` lobbies. Blackjack is dealt by the casino system (deterministic H17), the bot itself joins each round as a player driven by its own deterministic strategy (fractional-Kelly betting and EV-based play), and `/casino` / `/pocat` surface the casino ledger and the bot's wallet.
- **Localized commands**: slash command metadata is localized for English, Traditional Chinese, and Japanese. AI replies follow the user's language. There is no help command: ask the bot what it can do and it answers from a single English capability reference, translated into whatever language you asked in.

## Commands

| Command                                     | What it does                                                                                                                                     |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `@bot <message>`                            | Chat with the AI. Attach supported files or images when you want the bot to inspect them.                                                        |
| _Threads URL_                               | Automatically expands Threads posts and media, unless the bot is mentioned (then it answers about the comments too).                             |
| `/clean_threads_url <url>`                  | Privately turns a Threads share link into the post's own URL, so passing it on no longer names whoever shared it.                                |
| _Douyin URL_                                | Automatically posts the video or photos, unless the bot is mentioned (then it answers about it).                                                 |
| _Bilibili URL + mention_                    | Watches the linked video and answers about it (a bare link is not auto-expanded).                                                                |
| `/download_video <url> [quality]`           | Downloads a video and sends it back to Discord. A Douyin photo post comes back as images.                                                        |
| `/balance [member]`                         | Privately shows a member's 虛擬歡樂豆 balance, debt, net worth, and VIP status.                                                                  |
| `/vip`                                      | Buys permanent VIP perks.                                                                                                                        |
| `/leaderboard`                              | Shows the global top balances.                                                                                                                   |
| `/loss_leaderboard`                         | Shows today's accumulated casino losses.                                                                                                         |
| `/credit status\|borrow\|call\|repay`       | Handles personal credit requests, 180-second approval/rejection/cancel buttons, repayment, collection, and status.                               |
| `/central_bank status\|borrow\|call\|repay` | Handles central-bank loan requests, 180-second approval/rejection/cancel buttons, repayment, collection, and capacity.                           |
| `/give <member> <amount>`                   | Transfers 虛擬歡樂豆 to another member or bot.                                                                                                   |
| `/admin refund_tax\|collect_tax`            | Manual balance adjustments for members or bots; gated on the `economy admin` account flag, not on a Discord role.                                |
| `/games blackjack <bet>`                    | Opens a multiplayer Blackjack lobby; `bet` accepts comma-formatted numbers, and `0` means all in.                                                |
| `/games dragon_gate`                        | Opens a multiplayer 射龍門 table backed by the shared jackpot pool.                                                                              |
| `/casino`                                   | Shows the casino system's cumulative profit and loss.                                                                                            |
| `/pocat`                                    | Shows the bot player's own wallet (shortcut for `/balance @bot`).                                                                                |
| `/memory show\|regenerate\|clear`           | Privately shows, rebuilds, or erases what the bot remembers about you; regenerate runs in the background, and clear asks for confirmation first. |
| `/ping`                                     | Checks bot latency.                                                                                                                              |

## Development

Contributor setup, code conventions, tests, and release notes live in [CONTRIBUTING.md](./.github/CONTRIBUTING.md).

[Documentation](https://mai0313.github.io/discordbot/) | [Report a Bug](https://github.com/Mai0313/discordbot/issues) | [Discussions](https://github.com/Mai0313/discordbot/discussions)

[![Contributors](https://contrib.rocks/image?repo=Mai0313/discordbot)](https://github.com/Mai0313/discordbot/graphs/contributors)
