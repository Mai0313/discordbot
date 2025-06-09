<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Python Best Practices

## Coding Style

- Follow `ruff-check` and `ruff-format` for code style and formatting using `pre-commit` hooks.
- Follow PEP 8 naming conventions:
    - snake_case for functions and variables
    - PascalCase for classes
    - UPPER_CASE for constants
- Follow the Python version specified in the `pyproject.toml` or `.python-version` file.
- Use pydantic model, and all pydantic models should include `Field`, and `description` should be included.
- Maximum line length of 99 characters
- Use absolute imports over relative imports
- For tests, it should be placed in the `tests/` directory, and the test file should start with `test_`.
    - Use `assert` statements for testing conditions

### Example

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

- **Main Bot Implementation**: The primary bot class `DiscordBot` is implemented in `src/bot.py`, which extends `nextcord.ext.commands.Bot` and handles bot initialization, cog loading, logging configuration, and event management
- **Framework**: Nextcord (Discord.py fork) with async/await patterns
- **Structure**: Modular Cog system under `src/cogs/` with implementation details in `src/sdk/`
- **Configuration**: Pydantic-based config management with environment variable support
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

#### 5. Voice Channel Connection (`src/cogs/voice_recording.py`)

**Commands:**

- `/voice_join` - Join voice channel and establish connection
- `/voice_stop` - Leave voice channel and disconnect
- `/voice_status` - Check current voice connection status

**Implementation Details:**

- **Voice Connection**: Full voice channel connection management via `VoiceRecorder` utility (`src/utils/voice_recorder.py`)
- **Connection Management**: Per-guild voice connection tracking with automatic cleanup
- **Duration Control**: Configurable maximum connection duration (1-60 minutes, default 5 minutes)
- **Auto-Disconnect**: Automatic disconnection after maximum duration to prevent resource waste
- **Permission Validation**: Comprehensive permission checking (connect/speak permissions)
- **Channel Detection**: Smart channel detection (user's current channel or specified channel)
- **Connection Status**: Real-time connection status monitoring and reporting
- **Multi-language Support**: Commands and responses localized for Traditional Chinese, Simplified Chinese, Japanese, and English

**Technical Features:**

- Voice client management with proper connection lifecycle
- Automatic cleanup on bot disconnection events
- Connection duration tracking and status reporting
- Error handling for permission issues and connection failures
- Guild-specific voice recorder instances with isolated state management

**Current Limitations:**

- **⚠️ Audio Recording Not Supported**: nextcord framework does not include the `sinks` module required for audio recording
- **Connection Only**: This implementation provides voice channel connection capabilities but cannot record audio
- **Alternative Solution**: For audio recording, consider migrating to **pycord** which includes full `discord.sinks` support
- **Functional Scope**: Current implementation focuses on voice channel presence and connection management

**Audio Recording Migration Path:**

- **Recommended**: Migrate to `pycord` library which includes `discord.sinks.WaveSink`, `MP3Sink`, `MP4Sink`, etc.
- **Required Changes**: Update imports from `nextcord` to `discord` and install `py-cord[voice]`
- **Recording Features Available in Pycord**: Multi-format recording, per-user audio separation, automatic file generation

#### 6. Image Generation (`src/cogs/gen_image.py`)

**Commands:**

- `/graph` - Image generation placeholder (framework ready)

**Implementation Details:**

- **Current Status**: Stub implementation with async deferral pattern
- **Architecture**: Ready for integration with image generation APIs
- **Response Pattern**: Placeholder response with proper interaction handling

#### 7. MapleStory Database Query (`src/cogs/maplestory.py`)

**Database Query Commands:**

- `/maple_monster` - Search for monster drop information
- `/maple_item` - Search for item drop sources
- `/maple_stats` - Display database statistics

#### 8. Auction System (`src/cogs/auction.py`)

**Auction System Commands:**

- `/auction_create` - Create a new item auction with starting price and bid increment
- `/auction_list` - Browse all active auctions with interactive selection
- `/auction_info` - View detailed information about a specific auction
- `/auction_my` - View your created auctions and their current status

**Auction System Usage Guide:**

The comprehensive auction system allows users to create item auctions and participate in bidding with complete interactive features:

**Core Features:**

- **Server Isolation**: Each Discord server has completely independent auction data with guild-specific filtering
- **Auction Creation**: Two-step interactive process with currency selection dropdown (楓幣/雪花/台幣) followed by modal form for item details (name max 100 chars), starting price (float), bid increment (float), and duration (1-168 hours, default 24)
- **Currency Type Support**: Users can choose between "楓幣" (Mesos), "雪花" (Snowflake), and "台幣" (Taiwan Dollar) via dropdown selection with emoji indicators, with "楓幣" as default
- **Float Price Support**: All price fields (starting price, increment, bid amounts) support decimal values with proper `.2f` formatting throughout the UI
- **Auction Browsing**: Display of top 5 active auctions with dropdown selection for detailed viewing (server-specific)
- **Real-time Updates**: Live remaining time and current price displays with proper float currency formatting
- **Personal Auction Management**: View created auctions and current leading bids with currency type indication and float formatting (server-specific)

**Interactive Components:**

- **Auction Panel Buttons**: 💰 Bid (opens bid form), 📊 View Records (shows top 10 bid history), 🔄 Refresh (updates auction info)
- **Auto-Claim System**: Any button interaction on unclaimed auctions (guild_id=0) automatically assigns them to the current server
- **Bidding Rules**: Minimum bid = current price + increment, creators cannot bid on own auctions, current leaders cannot rebid, expired auctions reject bids
- **Security Features**: Self-bidding prevention, duplicate bid validation, price range validation, expiration time checks, server-specific validation
- **Guild Validation**: All commands restricted to server use only (dm_permission=False), automatic guild_id validation

**Database Architecture:**

- **auctions table**: id, guild_id (INTEGER NOT NULL), item_name, starting_price (REAL), increment (REAL), duration_hours, creator_id/name, created_at, end_time, current_price (REAL), current_bidder_id/name, is_active, currency_type
- **bids table**: id, auction_id, guild_id (INTEGER NOT NULL), bidder_id/name, amount (REAL), timestamp
- **Data Storage**: SQLite database at `data/auctions.db` with ACID compliance, automatic schema migration from INTEGER to REAL for price fields, and guild_id isolation
- **Server Isolation**: All database queries filtered by guild_id to ensure complete separation between servers
- **Auto-Claim Feature**: Unclaimed auctions (guild_id=0) are automatically assigned to the server where users interact with them
- **Currency Support**: Flexible currency type system supporting "楓幣" (Mesos), "雪花" (Snowflake), and "台幣" (Taiwan Dollar) with backward compatibility
- **Migration Support**: Robust database migration logic that handles different schema versions, missing columns with proper default values, and automatic guild_id column addition

**Implementation Details:**

**MapleStory Database Query System:**

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

**Auction System Implementation:**

- **Data Storage**: SQLite database (`data/auctions.db`) with ACID compliance and automatic migration from INTEGER to REAL for float price support

- **Data Models**: Pydantic models (`Auction`, `Bid`) with comprehensive field validation and float-type price fields

- **Interactive UI**: Two-step auction creation with currency selection dropdown followed by modal form, and bidding modals with float price validation

- **Real-time Updates**: Dynamic auction displays with refresh, bid, and history buttons showing proper float formatting

- **Security Features**:

    - Prevent self-bidding on own auctions
    - Duplicate bid validation and proper increment enforcement with float precision
    - Automatic auction expiration handling (customizable 1-168 hour duration)

- **Bid Management**: Complete bid history tracking with timestamps, user information, and float amount formatting

- **Advanced Features:**

- **Statistics Generation**: Popular item tracking based on drop frequency

- **Visual Enhancement**: Embedded images from external Artale database

- **Error Handling**: Graceful handling of missing data files and malformed JSON

- **Result Pagination**: Discord's 25-option limit handling with "and X more" indicators

- **Auction Persistence**: Reliable SQLite storage with proper database schema management

- **Multi-language Auction Support**: All auction interfaces localized for 4 languages

- **Multi-currency Display**: Dynamic currency formatting in all auction displays and interactions with float precision

**Technical Architecture:**

- **MapleStory Data Models**:
    - JSON-based monster/item relationships with comprehensive attribute mapping
- **MapleStory Database Operations**: Search algorithms with string containment matching and result ranking
- **MapleStory UI Components**: Custom View classes with Select menus for user interaction
- **Auction Data Models**: Pydantic-based auction and bid models with field validation, descriptions, currency type support, guild_id isolation, and float price fields
- **Auction Database Operations**: `AuctionDatabase` class with full CRUD operations, guild_id filtering, currency type handling, auto-claim functionality for unclaimed auctions, and robust migration support for float conversion and server isolation
- **Auction UI Components**:
    - Currency selection dropdown (`AuctionCurrencySelectionView`) for two-step auction creation
    - Modal classes for form-based data input with currency pre-selection and guild validation (`AuctionCreateModal`, `AuctionBidModal`)
    - Interactive button views for auction participation with server-specific data (`AuctionView`, `AuctionListView`)
    - Interactive button views for auction participation (`AuctionView`, `AuctionListView`)
- **External Integration**: Links to MapleStory library for detailed item information
- **Auction Logic**: Comprehensive bid validation, auction state management, currency type handling, float price support, auto-claim system for server assignment, and automatic expiration

### Critical Core Functionality

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

#### Key Configuration Classes:

- `DiscordConfig` - Bot token and server configuration
- `OpenAIConfig` - OpenAI/Azure API configuration with type detection
- `PerplexityConfig` - Perplexity API configuration

### Development and Deployment

#### Project Structure:

- **Cogs**: Modular command implementations in `src/cogs/`
    - `gen_reply.py` - AI text generation with OpenAI models
    - `gen_search.py` - Web search via Perplexity API
    - `summary.py` - Message summarization with interactive UI
    - `video.py` - Multi-platform video downloading
    - `voice_recording.py` - Voice channel connection management (recording not supported by nextcord)
    - `maplestory.py` - MapleStory database queries and drop searches
    - `auction.py` - Auction system with bidding functionality
    - `gen_image.py` - Image generation placeholder
    - `template.py` - System utilities and ping testing
- **SDK**: Core business logic in `src/sdk/`
- **Types**: Configuration and data models in `src/types/`
- **Utils**: Utility functions in `src/utils/`
- **Tests**: Comprehensive test suite in `tests/`
- **Data**: Game databases and user data in `data/`
    - `monsters.json` - MapleStory monster and drop database (192+ monsters)
    - `auctions.db` - SQLite database for auction system with bid tracking

#### Key Dependencies:

- `nextcord` - Discord API wrapper
- `openai` - OpenAI API client
- `pydantic` - Data validation and configuration
- `yt-dlp` - Video downloading
- `logfire` - Advanced logging and monitoring

#### Deployment Features:

- Docker support with `docker-compose.yaml`
- Development container configuration
- Comprehensive CI/CD pipeline with testing and code quality checks
- Documentation generation with MkDocs

This Discord Bot represents a comprehensive AI-powered Discord enhancement that provides intelligent conversation assistance, content processing, and utility capabilities with enterprise-grade logging and monitoring.

**Important Note**: Voice recording functionality is currently **not supported** due to nextcord framework limitations (missing `sinks` module). The bot provides voice channel connection capabilities only. For full audio recording features, migration to **pycord** is recommended, which includes complete `discord.sinks` support for multi-format audio recording.
