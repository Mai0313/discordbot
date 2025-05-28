<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Python Best Practices

## Code Style

- Follow ruff for linting
- Follow PEP 8 naming conventions:
    - snake_case for functions and variables
    - PascalCase for classes
    - UPPER_CASE for constants
- Use pydantic model, and all pydantic models should include `Field`, and `description` should be included.
- Maximum line length of 99 characters
- Use absolute imports over relative imports
    Example:

```python
from pydantic import BaseModel, Field


class User(BaseModel):
    """Example User model.

    Attributes:
        name (str): The name of the user
    """

    name: str = Field(..., description="The name of the user")


def foo(self, extra_input: str) -> str:
    """Example function.

    Args:
        extra_input (str): Extra input for the function

    Returns:
        str: Result of the function
    """
    return f"Hello, {self.name} and {extra_input}"
```

## Type Hints

- Use type hints for all function parameters and returns
- Use `TypeVar` for generic types
- Use `Protocol` for duck typing

## Discord Bot Project Overview

This is a comprehensive Discord Bot built with **nextcord** (Discord.py fork) that provides AI-powered interactions, content processing, and utility features. The bot follows a modular Cog-based architecture with all commands implemented as slash commands supporting multiple languages (Traditional Chinese, Simplified Chinese, Japanese, and English).

### Core Architecture

- **Framework**: Nextcord (Discord.py fork) with async/await patterns
- **Structure**: Modular Cog system under `src/cogs/` with implementation details in `src/sdk/`
- **Configuration**: Pydantic-based config management with environment variable support
- **Database**: SQLite for message logging with optional Redis/PostgreSQL support
- **Logging**: Comprehensive logging with Logfire integration

### Main Features

#### 1. AI Text Generation (`src/cogs/gen_reply.py`)

**Commands:**

- `/oai` - Generate single AI response
- `/oais` - Generate AI response with real-time streaming

**Implementation Details:**

- **Model Support**: Multiple OpenAI models (GPT-4o, GPT-4o-mini, GPT-4-Turbo, o1, o1-mini, o3-mini)
- **Multi-API Support**: Both OpenAI and Azure OpenAI APIs via `src/sdk/llm.py`
- **Image Processing**: Supports image uploads with vision models using `autogen.agentchat.contrib.img_utils`
- **Streaming Response**: Real-time message editing for streaming responses
- **Content Preparation**: Automatic conversion of images to base64 data URIs
- **Error Handling**: Model-specific constraints (e.g., o1 models don't support images)
- **Response Format**: Automatically mentions the user in responses

**Technical Features:**

- Async OpenAI client with proper streaming support
- Pydantic configuration with model mapping for Azure deployments
- Content type detection and preparation for multi-modal inputs
- Proper error handling for API rate limits and model constraints

#### 2. Web Search Integration (`src/cogs/gen_search.py`)

**Commands:**

- `/search` - Perform web search with AI summarization

**Implementation Details:**

- **Search Engine**: Perplexity API with `llama-3.1-sonar-large-128k-online` model
- **Response Processing**: Direct integration with `LLMSDK.get_search_result()`
- **Real-time Updates**: Deferred response with follow-up editing
- **Multi-language Support**: Localized command descriptions and prompts

#### 3. Message Summarization (`src/cogs/summary.py`)

**Commands:**

- `/sum` - Interactive message summarization with menu selection

**Implementation Details:**

- **Interactive UI**: `SummarizeMenuView` with dropdown menus for configuration
- **Flexible Filtering**: Support for specific user filtering or general channel history
- **Message Processing**: `MessageFetcher.do_summarize()` handles:
    - Channel history fetching with configurable limits
    - Bot message filtering
    - Attachment and embed content extraction
    - Chronological message ordering
- **AI Processing**: Custom prompt template (`SUMMARY_PROMPT`) for user-categorized summaries
- **Content Handling**: Supports embedded content and file attachments in summaries

**Advanced Features:**

- Message count selection (10, 25, 50, 100, 200 messages)
- User-specific message filtering
- Attachment URL extraction and processing
- Embed content integration

#### 4. Video Downloading (`src/cogs/video.py`)

**Commands:**

- `/download_video` - Download videos from multiple platforms

**Implementation Details:**

- **Platform Support**: YouTube, Facebook Reels, Instagram, X (Twitter), TikTok, and more
- **Quality Options**: Best, High (1080p), Medium (720p), Low (480p), Audio Only
- **Backend**: `yt-dlp` library via `VideoDownloader` class in `src/utils/downloader.py`
- **File Management**:
    - Downloads to `./data/downloads/` with timestamp-based naming
    - Automatic file size checking against Discord's 25MB limit
    - Error handling with fallback embed responses
- **Progress Tracking**: Real-time status updates during download process

**Technical Features:**

- Dynamic filename generation with timestamp and URL parsing
- Quality format mapping for optimal file sizes
- Exception handling with user-friendly error messages
- File size validation before Discord upload

#### 5. Image Generation (`src/cogs/gen_image.py`)

**Commands:**

- `/graph` - Image generation placeholder (framework ready)

**Implementation Details:**

- **Current Status**: Stub implementation with async deferral pattern
- **Architecture**: Ready for integration with image generation APIs
- **Response Pattern**: Placeholder response with proper interaction handling

#### 6. MapleStory Database Query (`src/cogs/maplestory.py`)

**Commands:**

- `/maple_monster` - Search for monster drop information
- `/maple_item` - Search for item drop sources
- `/maple_stats` - Display database statistics

**Implementation Details:**

- **Data Source**: Comprehensive JSON database (`data/monsters.json`) with 192+ monsters
- **Search Engine**: Fuzzy string matching with case-insensitive search
- **Interactive UI**: `MapleDropSearchView` with dropdown selection for multiple results
- **Multi-language Support**: Commands and responses localized for Traditional Chinese, Simplified Chinese, Japanese, and English
- **Performance Optimization**: LRU cache for frequent queries and item popularity tracking
- **Rich Information Display**:
    - Monster attributes (level, HP, MP, EXP, defense stats)
    - Drop item categorization (equipment vs consumables/materials)
    - Location mapping with up to 5 display locations
    - Item source tracking with visual thumbnails and external links

**Advanced Features:**

- **Statistics Generation**: Popular item tracking based on drop frequency
- **Visual Enhancement**: Embedded images from external Artale database
- **Error Handling**: Graceful handling of missing data files and malformed JSON
- **Result Pagination**: Discord's 25-option limit handling with "and X more" indicators

**Technical Architecture:**

- **Data Models**: JSON-based monster/item relationships with comprehensive attribute mapping
- **Search Algorithms**: String containment matching with result ranking
- **UI Components**: Custom View classes with Select menus for user interaction
- **External Integration**: Links to MapleStory library for detailed item information

### Critical Core Functionality

#### Message Logging System (`src/sdk/log_message.py`)

This is a **critical and unique feature** that automatically logs ALL user messages across ALL channels and DMs into an SQLite database.

**Implementation Details:**

- **Automatic Triggering**: Registered in `src/bot.py` via `on_message` event handler
- **Data Storage**: SQLite database at `data/messages.db` with table per channel/DM
- **Comprehensive Logging**:
    - User information (name, ID)
    - Message content and timestamps
    - Channel information (name, ID)
    - File attachments (saved to `data/attachments/YYYY-MM-DD/channel_name/`)
    - Discord stickers and embeds
- **File Management**:
    - Automatic directory structure creation by date
    - Duplicate filename handling with incremental numbering
    - Async file downloading and storage
- **Database Schema**: Pandas DataFrame to SQLite with proper data type conversion
- **Privacy Considerations**: Bot messages are excluded from logging

**Technical Architecture:**

- **Pydantic Models**: `MessageLogger` class with computed fields and caching
- **Async Processing**: Non-blocking message processing with `asyncio.create_task()`
- **Path Management**: Dynamic path generation based on message type (DM vs channel)
- **Error Handling**: Comprehensive exception handling for missing resources

#### LLM Integration SDK (`src/sdk/llm.py`)

**Core Features:**

- **Multi-Provider Support**: OpenAI and Azure OpenAI with automatic client selection
- **Model Mapping**: Azure deployment name mapping for seamless switching
- **Streaming Support**: Full async streaming implementation for real-time responses
- **Image Processing**: Automatic image conversion to base64 data URIs
- **Configuration Management**: Pydantic-based configuration with environment variable support

**API Implementations:**

- `get_oai_reply()` - Single response generation
- `get_oai_reply_stream()` - Streaming response generation
- `get_search_result()` - Perplexity API integration
- `_prepare_content()` - Multi-modal content preparation

### Configuration and Environment

#### Environment Variables Required:

- `DISCORD_BOT_TOKEN` - Discord bot token
- `OPENAI_API_KEY` - OpenAI API key
- `AZURE_OPENAI_API_KEY` - Azure OpenAI API key (if using Azure)
- `OPENAI_BASE_URL` - API base URL
- `PERPLEXITY_API_KEY` - Perplexity API key for search
- `SQLITE_FILE_PATH` - SQLite database file path (defaults to `data/messages.db`)

#### Key Configuration Classes:

- `DiscordConfig` - Bot token and server configuration
- `OpenAIConfig` - OpenAI/Azure API configuration with type detection
- `PerplexityConfig` - Perplexity API configuration
- `DatabaseConfig` - Multi-database support (SQLite, PostgreSQL, Redis)

### Development and Deployment

#### Project Structure:

- **Cogs**: Modular command implementations in `src/cogs/`
    - `gen_reply.py` - AI text generation with OpenAI models
    - `gen_search.py` - Web search via Perplexity API
    - `summary.py` - Message summarization with interactive UI
    - `video.py` - Multi-platform video downloading
    - `maplestory.py` - MapleStory database queries and drop searches
    - `gen_image.py` - Image generation placeholder
    - `template.py` - System utilities and ping testing
- **SDK**: Core business logic in `src/sdk/`
- **Types**: Configuration and data models in `src/types/`
- **Utils**: Utility functions in `src/utils/`
- **Tests**: Comprehensive test suite in `tests/`
- **Data**: Game databases and user data in `data/`
    - `monsters.json` - MapleStory monster and drop database (192+ monsters)

#### Key Dependencies:

- `nextcord` - Discord API wrapper
- `openai` - OpenAI API client
- `pydantic` - Data validation and configuration
- `yt-dlp` - Video downloading
- `pandas` - Data processing for logging
- `sqlalchemy` - Database operations
- `logfire` - Advanced logging and monitoring

#### Deployment Features:

- Docker support with `docker-compose.yaml`
- Development container configuration
- Comprehensive CI/CD pipeline with testing and code quality checks
- Documentation generation with MkDocs

This Discord Bot represents a comprehensive AI-powered Discord enhancement that provides intelligent conversation assistance, content processing, and automated archival capabilities with enterprise-grade logging and monitoring.
