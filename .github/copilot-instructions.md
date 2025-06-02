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

#### 5. Image Generation (`src/cogs/gen_image.py`)

**Commands:**

- `/graph` - Image generation placeholder (framework ready)

**Implementation Details:**

- **Current Status**: Stub implementation with async deferral pattern
- **Architecture**: Ready for integration with image generation APIs
- **Response Pattern**: Placeholder response with proper interaction handling

#### 6. MapleStory Database Query & Auction System (`src/cogs/maplestory.py`)

**Database Query Commands:**

- `/maple_monster` - Search for monster drop information
- `/maple_item` - Search for item drop sources
- `/maple_stats` - Display database statistics

**Auction System Commands:**

- `/auction_create` - Create a new item auction with starting price and bid increment
- `/auction_list` - Browse all active auctions with interactive selection
- `/auction_info` - View detailed information about a specific auction
- `/auction_my` - View your created auctions and their current status

**Auction System Usage Guide:**

The comprehensive auction system allows users to create item auctions and participate in bidding with complete interactive features:

**Core Features:**

- **Auction Creation**: Interactive modal form for creating auctions with item name (max 100 chars), starting price, bid increment, currency type selection (楓幣/雪花), and duration (1-168 hours, default 24)
- **Currency Type Support**: Users can choose between "楓幣" (Mesos) and "雪花" (Snowflake) as auction currency with "楓幣" as default
- **Auction Browsing**: Display of top 5 active auctions with dropdown selection for detailed viewing
- **Real-time Updates**: Live remaining time and current price displays with proper currency formatting
- **Personal Auction Management**: View created auctions and current leading bids with currency type indication

**Interactive Components:**

- **Auction Panel Buttons**: 💰 Bid (opens bid form), 📊 View Records (shows top 10 bid history), 🔄 Refresh (updates auction info)
- **Bidding Rules**: Minimum bid = current price + increment, creators cannot bid on own auctions, current leaders cannot rebid, expired auctions reject bids
- **Security Features**: Self-bidding prevention, duplicate bid validation, price range validation, expiration time checks

**Database Architecture:**

- **auctions table**: id, item_name, starting_price, increment, duration_hours, creator_id/name, created_at, end_time, current_price, current_bidder_id/name, is_active, currency_type
- **bids table**: id, auction_id, bidder_id/name, amount, timestamp
- **Data Storage**: SQLite database at `data/auctions.db` with ACID compliance and automatic schema management
- **Currency Support**: Flexible currency type system supporting "楓幣" (Mesos) and "雪花" (Snowflake) with backward compatibility

**Implementation Details:**

**Database Query System:**

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

**Auction System:**

- **Data Storage**: SQLite database (`data/auctions.db`) with ACID compliance

- **Data Models**: Pydantic models (`Auction`, `Bid`) with comprehensive field validation

- **Interactive UI**: Modal dialogs for auction creation and bidding with form validation

- **Real-time Updates**: Dynamic auction displays with refresh, bid, and history buttons

- **Security Features**:

    - Prevent self-bidding on own auctions
    - Duplicate bid validation and proper increment enforcement
    - Automatic auction expiration handling (24-hour duration)

- **Bid Management**: Complete bid history tracking with timestamps and user information

- **Advanced Features:**

- **Statistics Generation**: Popular item tracking based on drop frequency

- **Visual Enhancement**: Embedded images from external Artale database

- **Error Handling**: Graceful handling of missing data files and malformed JSON

- **Result Pagination**: Discord's 25-option limit handling with "and X more" indicators

- **Auction Persistence**: Reliable SQLite storage with proper database schema management

- **Multi-language Auction Support**: All auction interfaces localized for 4 languages

- **Multi-currency Display**: Dynamic currency formatting in all auction displays and interactions

**Technical Architecture:**

- **Data Models**:
    - JSON-based monster/item relationships with comprehensive attribute mapping
    - Pydantic-based auction and bid models with field validation, descriptions, and currency type support
- **Database Operations**: `AuctionDatabase` class with full CRUD operations and currency type handling
- **Search Algorithms**: String containment matching with result ranking
- **UI Components**:
    - Custom View classes with Select menus for user interaction
    - Modal classes for form-based data input with currency selection (`AuctionCreateModal`, `AuctionBidModal`)
    - Interactive button views for auction participation (`AuctionView`, `AuctionListView`)
- **External Integration**: Links to MapleStory library for detailed item information
- **Auction Logic**: Comprehensive bid validation, auction state management, currency type handling, and automatic expiration

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
    - `maplestory.py` - MapleStory database queries, drop searches, and auction system
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
