<div align="center" markdown="1">

# AI 智能 Discord 机器人

[![PyPI version](https://img.shields.io/pypi/v/swebenchv2.svg)](https://pypi.org/project/swebenchv2/) [![python](https://img.shields.io/badge/-Python_%7C_3.12%7C_3.13%7C_3.14-blue?logo=python&logoColor=white)](https://www.python.org/downloads/source/) [![uv](https://img.shields.io/badge/-uv_dependency_management-2C5F2D?logo=python&logoColor=white)](https://docs.astral.sh/uv/) [![nextcord](https://img.shields.io/badge/-Nextcord-5865F2?logo=discord&logoColor=white)](https://github.com/nextcord/nextcord) [![openai](https://img.shields.io/badge/-OpenAI-412991?logo=openai&logoColor=white)](https://openai.com) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty) [![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://docs.pydantic.dev/latest/contributing/#badges) [![tests](https://github.com/Mai0313/discordbot/actions/workflows/test.yml/badge.svg)](https://github.com/Mai0313/discordbot/actions/workflows/test.yml) [![code-quality](https://github.com/Mai0313/discordbot/actions/workflows/code-quality-check.yml/badge.svg)](https://github.com/Mai0313/discordbot/actions/workflows/code-quality-check.yml) [![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/Mai0313/discordbot/tree/main?tab=License-1-ov-file) [![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Mai0313/discordbot/pulls) [![contributors](https://img.shields.io/github/contributors/Mai0313/discordbot.svg)](https://github.com/Mai0313/discordbot/graphs/contributors)

[**English**](./README.md) | [**繁體中文**](./README.zh-TW.md) | **简体中文**

</div>

一个自托管 Discord 机器人，提供 AI 聊天、图片与视频生成、Threads、Facebook、Instagram、Twitter 与抖音链接展开、视频下载、虚拟欢乐豆与赌场小游戏。

## 功能展示

标记机器人并问它会做什么。这里没有 help 指令，它会用你提问的语言回答。

![请机器人自我介绍](assets/showcase-ai-chat.png)

叫它画一张图，它会直接生成，并以自己的语气回复。

![请机器人生成图片](assets/showcase-image-generation.png)

再叫它把同一张图动起来，它会返回一小段视频。

![请机器人把生成的图片变成视频](assets/showcase-video-generation.png)

## 一条回复是怎么跑的

```mermaid
flowchart TD
    U(["标记机器人 · DM · /ask"]) --> UP & RT & CX

    UP["上传附件"]
    RT["路由 + 思考强度 + 记忆挑选"]
    CX["频道历史 · 语气"]

    RT --> R{"分派路线"}

    R -->|QA| LF["抓用户在问的链接帖子"]
    LF --> Y{"在问贴上的 YouTube 视频？"}
    Y -->|是| YT["Interactions 流式：<br/>Gemini 自己看视频"]
    Y -->|否| QS["Responses 流式"]
    YT & QS --> Q2["内嵌标记：<br/>语音 · 图片 · 音乐<br/>视频 · 研究 · 记忆"]

    R -->|IMAGE| I1["扩写提示词，<br/>生成或编辑图片"]
    I1 --> I2["先发出，<br/>再以自己的语气回复"]
    R -->|VIDEO| V1["扩写提示词，<br/>编辑视频时跳过"]
    V1 --> V2["生成或编辑视频"]
    V2 --> V3["先发出，看过再回复"]

    Q2 & I2 & V3 --> OUT(["回复上屏"])
    Q2 -.->|"记忆标记"| MM["后台：整理记忆"]
    Q2 -.->|"研究标记"| DR["深度研究讨论串"]

    classDef proxy fill:#12607a,stroke:#12607a,color:#ffffff
    classDef direct fill:#a8481b,stroke:#a8481b,color:#ffffff
    class RT,QS,I1,I2,V1,V3,MM proxy
    class UP,LF,YT,V2,DR direct
```

蓝色的步骤走 OpenAI-compatible proxy，橙色的直接调用 Google。附件上传和频道历史与路由同时开跑，等到写回复时才等它们。

## 功能

- **AI 聊天**：在服务器标记机器人或发送 DM。它可以回答问题、总结近期聊天、检查支持的附件、观看贴上的 YouTube 视频、开口念出一段话、创作并录制短歌、生成或编辑图片、用提示或附加图片生成短视频、编辑引用的视频、以接续回复消息延续长回复，并搜索网络。它还会在后台慢慢积累对你个人偏好与服务器社群的长期记忆（按来源做隐私隔离：在某个服务器说的私事不会出现在别的服务器，只有语气偏好与明显无害的一般事实会跨服务器沿用）。你也可以直接叫它记住某件事，或告诉它记错了，回复下方会有一行小字说明它记下了什么。
- **Threads 解析**：贴上 Threads.net 或 Threads.com 链接，机器人会展开贴文、媒体与回复串，引用别人或自己先前的贴文时也会一起带出被引用的那篇；改成 tag 机器人并附上链接，或是回复别人贴链接的消息时 tag 机器人，它会改为连底下的留言一起读过再回答。
- **Facebook 解析**：贴上公开的 Facebook 帖子链接，机器人会把帖子、图片与互动数字展开到频道；链接若带 `comment_id`，底下会再显示那一则评论。改成 tag 机器人并附上链接，或在带链接的消息下 tag 它回复，它会读帖子加上页面预先加载的那几则评论再回答。只读得到公开帖子，视频帖子会以链接呈现而不是文件。
- **Instagram 解析**：贴上公开的 Instagram 帖子链接，机器人会把帖子文字、轮播图片与互动数字展开到频道；链接若指向某一则评论，底下会再显示那一则。改成 tag 机器人并附上链接，或在带链接的消息下 tag 它回复，它会读帖子加上底下的评论再回答。只读得到公开账号，Reel 会以链接呈现而不是看过视频。
- **Twitter 解析**：贴上 x.com 链接，机器人会把帖子、图片、它上面的整串帖子和它引用的那一条一起展开到频道；视频只放缩略图和链接。改成 tag 机器人并附上链接，或在回复含链接的消息时 tag 它，它会连同底下最热门的回复一起读过再回答。长帖子的后半段无法获取。
- **抖音解析**：贴上抖音链接，机器人会直接把视频（或图文贴文的图片）传到频道；改成 tag 机器人并附上链接，它会改为看过视频再回答。
- **不展开链接**：把链接包在剧透标记（`||链接||`）或尖括号（`<链接>`）里，上面这些解析都不会展开它。
- **Bilibili 问答**：tag 机器人并附上 B 站视频链接，它会看过视频再回答。单独贴链接不会自动展开；`/download_video` 仍可下载文件。
- **视频下载**：`/download_video` 可从 YouTube、TikTok、Instagram、X、Facebook、Bilibili，以及其他 yt-dlp 支持的网站下载视频。抖音也支持，无水印且包含图文贴文。文件太大无法上传时会改以链接提供。
- **虚拟欢乐豆与金融系统**：用户可从消息获得虚拟欢乐豆，可转账、购买 VIP、使用长期个人信贷或央行借款，并查看排行榜。
- **赌场游戏**：多人 `/games blackjack` 与 `/games dragon_gate` 牌桌。Blackjack 由赌场系统发牌 (H17)，bot 本身只要钱包大于零就会以玩家身份入桌，`/casino` 与 `/pocat` 分别显示赌场账本与 bot 玩家钱包。
- **本地化指令**：斜线指令支持英文、繁体中文、日文，AI 回复也会跟随用户语言。

## 指令

| 指令                                        | 功能                                                                                          |
| ------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `@bot <message>`                            | 和 AI 聊天。需要机器人检查文件或图片时，可附上支持的附件。                                    |
| `/ask <question> [attachment]`              | 在机器人看不到对话的地方和 AI 聊天：机器人没加入的服务器、群组 DM，或你和别人的 DM。          |
| `/deep_research <topic>`                    | 在自己开的讨论串里产出一份附引用的长篇研究报告；只能在机器人所在服务器的普通文字频道使用。    |
| _Threads URL_                               | 自动展开 Threads 贴文与媒体；被 tag 时改为连留言一起读过再回答。                              |
| `/clean_threads_url <url>`                  | 私密地把 Threads 分享链接还原成贴文本身的网址，转贴时不会带出分享者。                         |
| _Facebook URL_                              | 自动展开公开帖子与图片；被 tag 时改为读过帖子再回答。带 `comment_id` 的链接会多显示那则评论。 |
| _Instagram URL_                             | 自动展开公开帖子与轮播图片；被 tag 时改为读过帖子再回答。指向评论的链接会多显示那则评论。     |
| _Twitter URL_                               | 自动展开帖子、图片与上面的整串；被 tag 时改为连同热门回复读过再回答。                         |
| _抖音 URL_                                  | 自动传回视频或图片；被 tag 时改为看过视频再回答。                                             |
| _Bilibili URL + tag_                        | 看过链接的视频后回答（单独贴链接不会自动展开）。                                              |
| `/download_video <url> [quality]`           | 下载视频并传回 Discord。抖音的图文贴文会传回图片。                                            |
| `/balance [member]`                         | 私密显示成员的虚拟欢乐豆余额、债务、净资产与 VIP 状态。                                       |
| `/vip`                                      | 购买永久 VIP 特权；重复购买只会扣款一次，并显示已有的 VIP 状态。                              |
| `/leaderboard`                              | 显示全域余额排行榜。                                                                          |
| `/loss_leaderboard`                         | 显示今日赌场输钱累计排行榜。                                                                  |
| `/credit status\|borrow\|call\|repay`       | 处理个人信贷申请、180 秒批准/拒绝/取消按钮、还款、催收与状态。                                |
| `/central_bank status\|borrow\|call\|repay` | 处理央行借款申请、180 秒批准/拒绝/取消按钮、还款、催收与可放贷额度。                          |
| `/give <member> <amount>`                   | 转账虚拟欢乐豆给其他成员或 bot。                                                              |
| `/admin refund_tax\|collect_tax`            | 手动调整成员或 bot 余额；限定 `economy admin` 账号权限，不是 Discord 身份组。                 |
| `/games blackjack <bet>`                    | 开一个多人 Blackjack 牌桌；`bet` 可输入含逗号的数字，`0` 代表全下。                           |
| `/games dragon_gate`                        | 开一个由共享奖池支撑的多人射龙门桌。                                                          |
| `/games blackjack_history [member] [count]` | 显示某位玩家最近的 Blackjack 牌局：手牌、下注与结果（默认 10 局，最多 50 局）。               |
| `/casino`                                   | 显示赌场系统的累积损益。                                                                      |
| `/pocat`                                    | 显示 bot 玩家自己的钱包：余额与累计流水。                                                     |
| `/memory show\|regenerate\|clear`           | 私密查看、重建或清除 bot 对你记住的内容（regenerate 在后台执行，clear 会先要求确认）。        |
| `/memory server show`                       | 私密查看机器人对这个服务器社群的记忆；只能在机器人所在的服务器使用。                          |
| `/memory server catchup`                    | 把这个频道上次整理后的聊天整理进服务器社群记忆，结果公开贴出；每个服务器 10 分钟一次。        |
| `/ping`                                     | 检查机器人响应延迟。                                                                          |

## 开发

开发环境设置、代码规范、测试与发布流程请见 [CONTRIBUTING.md](./.github/CONTRIBUTING.md)。

[文档](https://mai0313.github.io/discordbot/) | [报告问题](https://github.com/Mai0313/discordbot/issues) | [讨论](https://github.com/Mai0313/discordbot/discussions)

[![Contributors](https://contrib.rocks/image?repo=Mai0313/discordbot)](https://github.com/Mai0313/discordbot/graphs/contributors)
