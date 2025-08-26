## Developer Instructions

### Video Downloader Retry Policy

- The retry mechanism for `VideoDownloader.download` (in `src/discordbot/utils/downloader.py`) is implemented using Tenacity as a decorator.
- Settings are intentionally hard-coded:
    - Attempts: 5
    - Wait between attempts: 1 second
    - Retry condition: any `Exception`
- The manual retry loop was removed in favor of Tenacity's `@retry` with `reraise=True`, `stop_after_attempt`, and `wait_fixed`.
- A `before_sleep` hook logs a warning with the current attempt number via `logfire`.

Code reference:

```python
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

_TENACITY_MAX_ATTEMPTS = 5
_TENACITY_WAIT_SECONDS = 1


@retry(
    reraise=True,
    stop=stop_after_attempt(_TENACITY_MAX_ATTEMPTS),
    wait=wait_fixed(_TENACITY_WAIT_SECONDS),
    retry=retry_if_exception_type(Exception),
    before_sleep=_before_sleep_log,
)
def download(self, url: str, quality: str = "best", dry_run: bool = False) -> tuple[str, Path]: ...
```

Notes:

- Keep Tenacity configuration colocated with the downloader for now; do not externalize into settings unless requirements change.
- `pyproject.toml` already lists `tenacity>=9.1.2` as a dependency.
- `max_retries` field remains on the model for backward compatibility but is not used by Tenacity.

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Discord Bot Project Overview

This is a comprehensive Discord Bot built with **nextcord** (Discord.py fork) that provides AI-powered interactions, content processing, and utility features. The bot follows a modular Cog-based architecture with all commands implemented as slash commands supporting multiple languages (Traditional Chinese, Japanese, and English).

### Core Architecture

- **Main Bot Implementation**: The primary bot class `DiscordBot` is implemented in `src/discordbot/cli.py`, which extends `nextcord.ext.commands.Bot` and handles bot initialization, cog loading, logging configuration, and event management
- **Framework**: Nextcord (Discord.py fork) with async/await patterns
- **Structure**: Modular Cog system under `src/discordbot/cogs/` with implementation details in `src/discordbot/sdk/`
- **Configuration**: Pydantic-based config management with environment variable support
- **Logging**: Comprehensive logging with Logfire integration

### Main Features

#### 1. AI Text Generation (`src/discordbot/cogs/gen_reply.py`)

**Commands:**

- `/oai` - Generate AI response with **default streaming** (updates about every 10 characters) and integrated web search capabilities
- `/clear_memory` - Clear per-user conversation memory (resets `previous_response_id`)

**Implementation Details:**

- **Streaming Response**: Default streaming mode that updates message content about every 10 characters for real-time user feedback
- **Progressive Image Rendering**: Image generation shows partial updates using `response.image_generation_call.partial_image` events for smooth visual progression
- **Model Support**: Multiple AI models (openai/gpt-4o [default], openai/gpt-5-mini, openai/gpt-5-nano, claude-3-5-haiku-20241022)
- **Multi-API Support**: Both OpenAI and Azure OpenAI APIs via `src/discordbot/sdk/llm.py`
- **Image Processing**: Supports image uploads via Discord attachments; `LLMSDK.prepare_response_content()` handles conversion/preparation
- **Integrated Web Search & Image Generation**: Uses the new Responses API tools with streaming support, e.g. `{"type": "web_search_preview"}` and `{"type": "image_generation"}`. The cog receives `image_generation_call` partial results (base64 via `partial_image_b64`) and posts images as Discord attachments with embeds.
- **Content Preparation**: Automatic conversion of images to base64 data URIs via `prepare_response_content()` method
- **Error Handling**: Model-specific constraints (e.g., o1 models don't support images)
- **Response Format**: Automatically mentions the user in responses with streaming updates
- **Architecture**: Uses the new OpenAI responses API with streaming support instead of chat completions, enabling tool use for web search
- **Memory Handling**: Conversation memory is tracked per user; `/clear_memory` clears the requesting user's memory

**Image Handling Details:**

- **Streaming Image Generation**: Handles `response.image_generation_call.partial_image` events for progressive image display during generation
- **Final Image Display**: Displays completed images as Discord attachments with green-colored embed indicating completion
- After receiving streaming `responses`, partial images are received via `partial_image_b64`; the accumulated final image is posted after completion
- Decoded content is sent as Discord attachments with corresponding embeds for visual feedback
- The regenerate flow also includes the image tool to allow alternate images with streaming support

**Technical Features:**

- **Real-time Streaming**: Async OpenAI client with streaming responses API for immediate user feedback
- **Progressive Image Display**: Handles `response.image_generation_call.partial_image` events to show image generation progress
- **Character-based Updates**: Text streaming updates message content about every 10 characters (hardcoded for simplicity)
- **Unified Display Method**: Uses a single display method for both partial and final images; final embed uses a green color theme
- Pydantic configuration with model mapping for Azure deployments
- Content type detection and preparation for multi-modal inputs
- Proper error handling for API rate limits and model constraints with streaming context
- Automatic web search integration through tool usage with streaming support

#### 2. Web Search Integration

**Implementation:**

- **Removed**: Standalone `/search` command has been removed
- **Integration**: Web search functionality is now integrated directly into the `/oai` command
- **Technical Details**:
    - Uses OpenAI's new responses API with `web_search_preview` tool
    - LLM can automatically determine when web search is needed
    - No separate search endpoint or UI required
    - Search results are processed contextually within the conversation
    - Eliminates need for separate search command and reduces complexity

#### 3. Message Summarization (`src/discordbot/cogs/summary.py`)

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

- Message count selection (5, 10, 20, 50 messages)
- User-specific message filtering
- Attachment URL extraction and processing
- Embed content integration

#### 4. Video Downloading (`src/discordbot/cogs/video.py`)

**Commands:**

- `/download_video` - Download videos from multiple platforms

**Implementation Details:**

- **Platform Support**: YouTube, Facebook Reels, Instagram, X (Twitter), TikTok, and more
- **Quality Options**: Best, High (1080p), Medium (720p), Low (480p)
- **Backend**: `yt-dlp` library via `VideoDownloader` class in `src/discordbot/utils/downloader.py`
    - Bilibili specifics: We set `Referer: https://www.bilibili.com`, enable `geo_bypass`, prefer `bestvideo*+bestaudio` with safe fallbacks, and force `merge_output_format=mp4`.
    - Headers are site-specific: Referer is applied only for Bilibili; other sites use minimal headers to avoid issues (e.g., Facebook).
    - API note: `VideoDownloader.get_params(quality, dry_run, url=None)` conditionally sets headers based on the URL; `download(url, quality, dry_run)` passes the URL through.
    - If a format error occurs, try a lower `quality` or pass cookies when needed (not wired by default).
- **File Management**:
    - Downloads to `./data/downloads/` under daily YYYYMMDD folders
    - Automatic file size checking against Discord's 25MB limit; auto low-quality re-download if exceeded
    - Error handling with user-facing messages
- **Progress Tracking**: Informational status updates during download process

**Reliability**:

- Wrapper-level configurable retry with linear backoff (defaults: `max_retries=3`, `retry_backoff_seconds=1.0`).

**Technical Features:**

- Dynamic filename generation with timestamp and URL parsing
- Quality format mapping for optimal file sizes
- Exception handling with user-friendly error messages
- File size validation before Discord upload

#### 5. Image Generation (`src/discordbot/cogs/gen_image.py`)

**Commands:**

- `/graph` - Image generation placeholder (framework ready)

**Implementation Details:**

- **Current Status**: Placeholder implementation with async deferral pattern. Primary image generation is integrated in `/oai` via Responses API tools.
- **Architecture**: Framework ready for integration with image generation APIs (DALL-E, Stable Diffusion)
- **Response Pattern**: Currently displays "功能沒寫完..." (Feature not implemented) message
- **Technical Foundation**: Command structure and localization already implemented

#### 6. MapleStory Database Query (`src/discordbot/cogs/maplestory.py`)

**Database Query Commands:**

- `/maple_monster` - Search for monster drop information
- `/maple_item` - Search for item drop sources
- `/maple_stats` - Display database statistics

#### 7. Auction System (`src/discordbot/cogs/auction.py`)

**Auction System Commands:**

- `/auction_create` - Create a new item auction with starting price and bid increment
- `/auction_list` - Browse all active auctions with interactive selection
- `/auction_info` - View detailed information about a specific auction
- `/auction_my` - View your created auctions and their current status

#### 8. Lottery System (`src/discordbot/cogs/lottery.py`)

**Lottery System Command:**

- `/lottery` - Single entry. Shows a dropdown to choose registration method (Discord button join / YouTube keyword), then opens a modal to fill in title/description and method-specific fields. The creation message renders a button control panel; reactions are not used:
    - `🎉` Join (Discord mode)
    - `🚫` Cancel join (Discord mode)
    - `✅` Start drawing (host-only)
    - `📊` Status (ephemeral to the requester)
    - `🔄` Recreate lottery (host-only)

**Lottery System Features:**

- **Dual-Platform Registration**: Discord button-based join OR YouTube chat keyword participation. Reactions are not used. Each lottery uses only one method (single-platform per lottery).
- **Button Controls**: `🎉` Join, `🚫` Cancel, `✅` Start (host-only), `📊` Status (ephemeral), `🔄` Recreate (host-only)
- **Button Controls**: `🎉` Join, `🚫` Cancel, `✅` Start (host-only), `📊` Status (ephemeral), `🔄` Recreate (host-only), `🔁` Update Participants (YouTube/host-only)
- **Winners Per Draw**: Creation modal supports configuring `draw_count` (default 1). On `✅`, the bot draws up to `min(draw_count, len(participants))` winners in a single go
- **Recreate Flow (`🔄`)**: Host can recreate a fresh lottery with identical settings. The bot restores all previous participants (including prior winners) and closes the old lottery
- **Comprehensive Status Monitoring**: Press `📊` to get an ephemeral status embed only visible to the requester
- **Auto-Updating Creation Message**: The creation message is edited in place to include participant name lists as users join or cancel
- **Winner Exclusion**: Winners are prevented from re-joining the same lottery. Use `🔄` Recreate to reset eligibility.
- **Memory Optimization**: defaultdict-based storage for automatic list initialization and efficient data handling
- **Interactive UI Components**: Modal forms, button views, and detailed status displays; includes recreate functionality
- **Security Features**: Creator-only controls, winner exclusion (cannot re-join same lottery), and UI-gated single-platform behavior

**Internal Design (Implementation Notes):**

- `lotteries_by_id: dict[int, LotteryData]` — direct lookup by `lottery_id` to avoid scanning global state
- `lottery_participants: defaultdict[int, list[LotteryParticipant]]` — auto-initialized participant lists
- `lottery_winners: defaultdict[int, list[LotteryParticipant]]` — winner history tracking
- Removed legacy `reaction_messages` mapping; `control_message_id` now lives inside `LotteryData` and is updated via `update_control_message_id()`
- Extracted helpers to remove duplication:
    - `add_participants_field(embed, participants)` shows a unified participant list (single field)
    - `build_creation_embed(lottery)` centralizes creation message embed with live participant name lists
    - Reaction-based helpers were removed in favor of button-based interactions
    - UI Button classes: `JoinLotteryButton` and `CancelJoinLotteryButton` now subclass `nextcord.ui.Button` and encapsulate their own `callback` logic. This replaces inline closures for better readability, reuse, testing, and persistent-view readiness (easy to assign stable `custom_id` if needed).
    - Winner re-join prevention is implemented centrally in `add_participant()`; UI flows rely on it to avoid duplicate checks

**Data Model Notes:**

- `LotteryData` includes `draw_count: int = 1` for winners-per-draw configuration

- **Button Handling:**

- `🎉` Join: Adds the Discord user to participants and edits the creation message to show updated participant name lists

- `🚫` Cancel: Removes the Discord user from participants and edits the creation message accordingly (Discord mode only)

- `✅` Draw: Draws up to `draw_count` winners. Each winner is removed from `lottery_participants` and appended to `lottery_winners`

- `🔄` Recreate: Gathers previous participants and winners, deduplicates by `(id, source)`, creates a new lottery with the same settings (including `draw_count` and YouTube fields), restores participants to the new lottery, sends a fresh embed with the control view, updates `control_message_id`, and calls `close_lottery()` on the old one

These changes are internal-only and preserve all user-visible behaviors.

Additional behavior:

- In YouTube mode, the host can press `🔁` Update Participants at any time to fetch participants from live chat using the configured keyword; the bot also performs a fetch right before drawing on `✅`.

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
- **Multi-language Support**: Commands and responses localized for Traditional Chinese, Japanese, and English
- **Performance Optimization**: LRU cache for frequent queries and item popularity tracking
- **Rich Information Display**:
    - Monster attributes (level, HP, MP, EXP, defense stats)
    - Drop item categorization (equipment vs consumables/materials)
    - Location mapping with up to 5 display locations
    - Item source tracking with visual thumbnails and external links

**Lottery System Implementation:**

- **Data Storage**: Optimized in-memory global variables with defaultdict for automatic initialization (lightweight, resets on restart)
- **Single-Platform Support**: Either Discord button-based joins OR YouTube chat per lottery (prevents cross-platform confusion)
- **Data Models**: Pydantic models (`LotteryData`, `LotteryParticipant`) with type validation
- **Interactive UI Components**:
    - `LotteryMethodSelectionView` for pre-selecting the registration method via dropdown
    - `LotteryCreateModal` for activity creation with platform-specific form fields
    - Detailed status display with participant name lists and platform breakdown
- **Selection**: Cryptographically secure random selection using the `secrets` module
- **Security Features**:
    - Creator-only access controls for lottery operations
    - UI-gated single-platform flow; no cross-platform duplicate validation at code level
    - Automatic participant removal upon winner selection
    - Permission validation for interactive elements
- **YouTube Integration**: Uses `YoutubeStream.get_registered_accounts()` to fetch participants by keyword at draw-time
- **Memory Architecture**: defaultdict-optimized storage with automatic list creation
    - `lottery_participants: defaultdict[int, list[LotteryParticipant]]`: Auto-initializing participant lists
    - `lottery_winners: defaultdict[int, list[LotteryParticipant]]`: Winner history tracking
- `control_message_id` (field on `LotteryData`): Message ID of the creation/control panel message. Used to map button interactions back to the correct lottery via `get_lottery_by_message_id()`
- **Display Optimization**:
    - Comma-separated participant formatting to fit within Discord field limits
    - Unified participant list field; platform is single per lottery so grouping is unnecessary
    - Real-time participant counting and validation

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

#### LLM Integration SDK (`src/discordbot/sdk/llm.py`)

**Core Features:**

- **Multi-Provider Support**: OpenAI and Azure OpenAI with automatic client selection
- **Model Mapping**: Azure deployment name mapping for seamless switching
- **Tool Usage Support**: Supports OpenAI responses API with tool usage for web search
- **Image Processing**: Automatic image conversion to base64 data URIs
- **Configuration Management**: Pydantic-based configuration with environment variable support

**API Methods:**

- `prepare_response_content()` - Multi-modal content preparation for responses API (supports input_text and attachments types)
- `prepare_completion_content()` - Multi-modal content preparation for completions API (supports text and attachments types)

**Note:** Previous methods `get_oai_reply()`, `get_oai_reply_stream()`, and `get_search_result()` have been removed. The new architecture uses OpenAI's responses API directly with integrated web search tools.

### Configuration and Environment

#### Environment Variables Required:

- `DISCORD_BOT_TOKEN` - Discord bot token
- `OPENAI_API_KEY` - OpenAI API key
- `OPENAI_BASE_URL` - API base URL (or `AZURE_OPENAI_ENDPOINT` when using Azure)
- Optional: `AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION`, `SQLITE_FILE_PATH`, `POSTGRES_URL`, `REDIS_URL`

#### Key Configuration Classes:

- `DiscordConfig` - Bot token and test server configuration
- `LLMSDK` - OpenAI/Azure client configuration (model, base URL, API key, optional version)
- `DatabaseConfig` - Aggregates `SQLiteConfig`, `PostgreSQLConfig`, and `RedisConfig`

### Development and Deployment

#### Project Structure:

- **Cogs**: Modular command implementations in `src/discordbot/cogs/`
    - `gen_reply.py` - AI text generation with multiple AI models (GPT-4o default, GPT-5 mini/nano, Claude) and integrated web search
    - `summary.py` - Message summarization with interactive UI (5/10/20/50 message options)
    - `video.py` - Multi-platform video downloading with quality options
    - `maplestory.py` - MapleStory database queries and drop searches
    - `auction.py` - Auction system with bidding functionality and multi-currency support
    - `lottery.py` - Multi-platform lottery system with button-based controls
    - `gen_image.py` - Image generation (placeholder implementation)
    - `template.py` - System utilities and ping testing
- **SDK**: Core business logic in `src/discordbot/sdk/`
- **Typings**: Configuration and data models in `src/discordbot/typings/`
- **Utils**: Utility functions in `src/discordbot/utils/`
- **Tests**: Comprehensive test suite in `tests/`
- **Data**: Game databases and user data in `data/`
    - `monsters.json` - MapleStory monster and drop database (192+ monsters)
    - `auctions.db` - SQLite database for auction system with bid tracking
    - `downloads/` - Video download storage directory

#### Running Locally

```bash
uv sync
uv run discordbot
# or: uv run python -m discordbot.cli
```

#### Testing Guidance

- We use `pytest` with `pytest-xdist`, `pytest-asyncio`, and `pytest-cov`. Config lives under `[tool.pytest.ini_options]` in `pyproject.toml`.
- To install and run tests locally:

```bash
uv sync --group test
uv run pytest -q
```

- Reports are written to:

    - `./.github/reports/pytest.xml` (JUnit)
    - `./.github/reports/coverage.xml` (coverage)
    - `./.github/coverage_html_report/index.html` (HTML coverage)

- Recently added cog unit tests cover:

    - `TemplateCogs` (`src/discordbot/cogs/template.py`): message listener reaction and `/ping` embed building
    - `MessageFetcher` (`src/discordbot/cogs/summary.py`): `_format_messages()` and `do_summarize()`; mocks `LLMSDK` and Discord channel history
    - `ReplyGeneratorCogs` (`src/discordbot/cogs/gen_reply.py`): `_get_attachment_list()` and `/clear_memory`
    - `ImageGeneratorCogs` (`src/discordbot/cogs/gen_image.py`): `/graph` flow (placeholder)
    - `VideoCogs` (`src/discordbot/cogs/video.py`): `/download_video` happy path; mocks `VideoDownloader`

- Test style tips:

    - Prefer `AsyncMock` for awaited Discord APIs (e.g., `interaction.response.defer`, `interaction.followup.send`).
    - For attributes like `bot.latency`, set a float (seconds) instead of a `Mock()` to avoid type errors.
    - Keep tests hermetic; avoid network access by mocking SDK and I/O.

#### Docker

```bash
docker-compose up -d
# or
docker build -t discordbot . && docker run -d discordbot
```

Note: The Docker image installs `ffmpeg` so `yt-dlp` can merge separate audio/video streams.

#### Key Dependencies:

- `nextcord` - Discord API wrapper
- `openai` - OpenAI API client
- `pydantic` - Data validation and configuration
- `yt-dlp` - Video downloading (with configured headers and retries)
- `logfire` - Advanced logging and monitoring
- `playwright` - Used by `scripts/artale_data_v*.py` to scrape MapleStory data (requires `playwright install chromium`)

#### Deployment Features:

- Docker support with `docker-compose.yaml`
- Development container configuration
- Comprehensive CI/CD pipeline with testing and code quality checks
- Documentation generation with MkDocs

### Maintenance Scripts

- `uv run update`: installs Playwright Chromium (first run) and refreshes `./data/monsters.json` via `scripts/artale_data_v2.py`.

### Environment

- Optional (YouTube lottery): set `YOUTUBE_DATA_API_KEY` and provide `./data/client_secret.json` (Desktop app). First use opens a localhost:8080 OAuth flow and saves `./data/token.pickle`.

This Discord Bot represents a comprehensive AI-powered Discord enhancement that provides intelligent conversation assistance, content processing, and utility capabilities with enterprise-grade logging and monitoring.
