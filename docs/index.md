# AI-Powered Discord Bot 🤖

**English** | [**繁體中文**](./README_cn.md)

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

A comprehensive Discord Bot built with **nextcord** that provides AI-powered interactions, content processing, and utility features. Features multi-language support and real-time streaming responses. 🚀⚡🔥

_Suggestions and contributions are always welcome!_

## ✨ Features

### 🤖 AI-Powered Interactions

- **Text Generation**: Support for multiple OpenAI models (GPT-4o, o1, o3-mini) with real-time streaming
- **Image Processing**: Vision model support with automatic image conversion
- **Web Search**: Integrated Perplexity API for real-time web search and summarization

### 📊 Content Processing

- **Message Summarization**: Smart channel conversation summaries with user filtering
- **Video Downloading**: Multi-platform support (YouTube, TikTok, Instagram, X, Facebook)

### 🌍 Multi-Language Support

- Traditional Chinese (繁體中文)
- Simplified Chinese (简体中文)
- Japanese (日本語)
- English

### 🔧 Technical Features

- Modular Cog-based architecture
- Async/await patterns with nextcord
- Pydantic-based configuration management
- Comprehensive error handling and logging
- Docker support with development containers

## 🎯 Core Commands

| Command           | Description                       | Features                           |
| ----------------- | --------------------------------- | ---------------------------------- |
| `/oai`            | Generate AI text response         | Multi-model support, image input   |
| `/oais`           | Real-time streaming AI response   | Live response updates              |
| `/search`         | Web search with AI summary        | Perplexity API integration         |
| `/sum`            | Interactive message summarization | User filtering, configurable count |
| `/download_video` | Multi-platform video downloader   | Quality options, size validation   |
| `/ping`           | Bot performance testing           | Latency measurement                |

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Discord Bot Token
- OpenAI API Key
- Perplexity API Key (for search)

### Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/Mai0313/discordbot.git
    cd discordbot
    ```

2. **Install dependencies using uv**

    ```bash
    # Install uv if you haven't already
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Install dependencies
    uv sync
    ```

3. **Set up environment variables**

    ```bash
    cp .env.example .env
    # Edit .env with your API keys and configuration
    ```

4. **Run the bot**

    ```bash
    uv run python main.py
    ```

### Docker Setup

```bash
# Using Docker Compose
docker-compose up -d

# Or build manually
docker build -t discordbot .
docker run -d discordbot
```

## ⚙️ Configuration

### Required Environment Variables

```env
# Discord Configuration
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_TEST_SERVER_ID=your_test_server_id  # Optional

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_TYPE=openai  # or "azure"

# Azure OpenAI (if using Azure)
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com

# Perplexity API (for search)
PERPLEXITY_API_KEY=your_perplexity_api_key
```

## 📁 Project Structure

```
src/
├── bot.py              # Main bot entry point
├── cogs/               # Command modules
│   ├── gen_reply.py    # AI text generation
│   ├── gen_search.py   # Web search integration
│   ├── summary.py      # Message summarization
│   ├── video.py        # Video downloading
│   ├── gen_image.py    # Image generation (placeholder)
│   └── template.py     # System utilities
├── sdk/                # Core business logic
│   ├── llm.py          # LLM integration
│   └── asst.py         # Assistant API wrapper
├── types/              # Configuration models
└── utils/              # Utility functions
```

## 🔍 Key Features Deep Dive

### Multi-Modal AI Support

- Text and image input processing
- Automatic image-to-base64 conversion
- Model-specific constraint handling
- Streaming response capabilities

### Video Download Engine

- Support for 10+ platforms
- Quality selection (4K to audio-only)
- File size validation for Discord limits
- Progress tracking and error handling

## 🔒 Privacy Policy

This Discord bot is committed to protecting user privacy and complies with Discord's Terms of Service and Developer Policy.

### Data Collection and Usage

- **No Message Storage**: This bot does NOT store, log, or retain any user messages, conversation content, or chat history
- **No Personal Data Collection**: We do not collect, store, or process any personal information from users
- **Temporary Processing Only**: User inputs are processed temporarily to generate responses and are immediately discarded
- **No Third-Party Sharing**: No user data is shared with third parties beyond the necessary API calls to provide bot functionality

### Bot Permissions and Intents

This bot requests the following permissions solely for functionality purposes:

- **Message Content Intent**: Required to read and respond to user commands and mentions
- **Slash Commands**: For interactive command processing
- **File Attachments**: For processing images in AI vision features and downloading requested content
- **Embed Links**: For formatting rich responses and search results

### Data Security

- All API communications use encrypted HTTPS connections
- Temporary data processing occurs in secure, ephemeral environments
- No user data persists beyond the immediate request-response cycle

### Contact and Compliance

If you have privacy concerns or questions about data handling:

- Report issues via [GitHub Issues](https://github.com/Mai0313/discordbot/issues)
- Contact the development team through the repository

This bot is designed with privacy-by-design principles and minimal data processing to ensure user privacy protection.

## Contributors

[![Contributors](https://contrib.rocks/image?repo=Mai0313/discordbot)](https://github.com/Mai0313/discordbot/graphs/contributors)

Made with [contrib.rocks](https://contrib.rocks)

## For More info, check the [Docs](https://mai0313.github.io/discordbot/)
