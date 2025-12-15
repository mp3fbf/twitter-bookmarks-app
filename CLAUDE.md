# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Application Overview

Twitter Bookmarks Processing App - fetches bookmarks via **Twillot extension** (not Twitter API) and processes them with LLMs (OpenAI, Anthropic, Google Gemini) to generate summaries, study guides, and insights.

### Why Twillot Instead of Twitter API?

Twitter API was removed because:
1. **Complex setup** - Required Developer Portal, OAuth 2.0, credential management
2. **Rate limits** - Free tier: 1 request per 15 minutes
3. **Data quality** - API returns truncated text, no media
4. **Twillot advantages** - No limits, complete data, simpler setup

## Development Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium

# Run the application
python main.py

# Environment setup (only LLM keys needed)
cp .env.example .env
# Edit .env with optional LLM API keys
```

## Architecture & Key Components

### Data Source: Twillot
- **twillot_scraper.py**: Browser automation with Playwright + Twillot extension
- Features:
  - `TwillotScraper`: Automated browser scraping
  - `TwillotImporter`: Import from JSON/CSV exports
  - Complete tweet text, media URLs, thread content
  - No rate limits

### Bookmark Management
- **bookmarks_fetcher.py**: Local bookmark operations only (no API calls)
- Methods:
  - `load_bookmarks()`: Load from JSON file
  - `save_bookmarks()`: Save to JSON file (with deduplication)
  - `export_to_markdown()`: Export for reading
  - `get_stats()`: Display statistics

### Intelligent Bookmark Analysis
- **bookmark_analyzer.py**: Topic extraction and knowledge graphs
- Features:
  - Keyword extraction with stop word filtering
  - Automatic categorization (AI/ML, programming, tools, etc.)
  - Knowledge graph creation
  - Mermaid diagram export
- Key methods:
  - `analyze_bookmarks()`: Main analysis entry point
  - `extract_keywords()`: NLP-based extraction
  - `create_topic_clusters()`: Group by topic
  - `export_knowledge_graph_mermaid()`: Visualization

### LLM Provider Factory
- **llm_providers.py**: Factory pattern for multiple providers
- Supports OpenAI, Anthropic, Google Gemini
- Special handling for o1/o3/GPT-5.x models (`max_completion_tokens` instead of `max_tokens`)
- **Vision support**: `generate_with_vision()` method for image analysis
- **Video support**: `generate_with_video()` method for video analysis

## Official Model Names (DO NOT CHANGE)

These are the correct model identifiers for API calls:

### OpenAI
- `gpt-5.2` - Latest GPT model with vision + video support

### Anthropic
- `claude-opus-4-5-20251101` - Claude Opus 4.5 with vision (max 20 images, 5MB each)

### Google Gemini
- `gemini-2.5-flash` - Fast, cost-effective vision + video
- `gemini-3-pro-preview` - Best quality for complex analysis

### Media Linker
- **media_linker.py**: Links Twillot JSON with local media files
- Features:
  - Scans `twillot-media-files-by-date/` folder recursively
  - Parses filename pattern: `{screen_name}-{tweet_id}-{media_id}.{ext}`
  - Enriches bookmarks with `local_media_paths` field
- Key methods:
  - `build_index()`: Scan and index media files
  - `get_media_for_tweet()`: Get media paths for a tweet ID
  - `enrich_bookmarks()`: Add local media paths to bookmarks

### Smart Processing System
- Topic-based processing in `main.py`
- Five modes:
  - Process by specific topic
  - Generate topic overview
  - Create learning paths
  - Compare similar tools
  - Deep dive analysis

### Content Fetcher
- **content_fetcher.py**: Extracts full content from URLs in tweets
- Features:
  - URL expansion (t.co, bit.ly, etc.)
  - Full page content extraction (not just metadata)
  - Paywall bypass attempts (archive.org, 12ft.io)
  - Content caching (1 week)
  - Special handling for GitHub, YouTube
- Key methods:
  - `fetch_content()`: Fetch and extract content from URL
  - `fetch_urls_from_tweet()`: Extract and fetch all URLs from tweet text
  - `get_content_summary()`: Get formatted summary of fetched content

### Smart Prompts
- **smart_prompts.py**: Intelligent prompt selection based on content type
- Detects tweet type and selects appropriate prompt:
  - `ARTICLE_LINK`: Extract key points from linked articles
  - `TOP_LIST`: Extract complete lists/rankings
  - `TUTORIAL_GUIDE`: Extract actionable steps
  - `TOOL_ANNOUNCEMENT`: Extract installation and use cases
  - `CODE_SNIPPET`: Extract complete code/prompts
  - `VIDEO_CONTENT`: Summarize video content
  - `SCREENSHOT_INFO`: OCR and extract visible text
- Key class: `SmartPromptSelector` with `detect_content_type()` and `build_prompt()`

## Data Flow

```
Twillot Extension/Export
        ↓
  twillot_scraper.py (import/scrape)
        ↓
  media_linker.py (link local media files)
        ↓
  bookmarks.json (local storage)
        ↓
  bookmark_analyzer.py (topics, keywords)
        ↓
  smart_prompts.py (detect content type, select prompt)
        ↓
  content_fetcher.py (fetch linked content)
        ↓
  llm_providers.py (AI processing + vision + video)
        ↓
  Markdown exports / Console output
```

## Menu Structure (11 options)

1. Fetch via Twillot (browser automation)
2. Import Twillot export file (JSON/CSV + optional media linking)
3. Load saved bookmarks from file
4. View bookmark statistics
5. Export bookmarks to Markdown
6. Expand URLs in loaded bookmarks
7. Analyze bookmark topics
8. Smart processing (by topic)
9. **Multimodal analysis (text + images)** - NEW
10. Configure LLM provider
11. Exit

## Files Removed (Twitter API cleanup)

- `twitter_auth.py` - OAuth 2.0 no longer needed
- Flask dependency - No callback server needed
- Tweepy dependency - No API client needed
- `tokens.json` - No tokens to store
- `pagination_state.json` - No API pagination

## Common Issues & Solutions

1. **Twillot automation fails**:
   - Run `playwright install chromium`
   - Ensure extension in `./extensions/twillot/`

2. **Import fails**:
   - Check JSON/CSV format matches Twillot export
   - Try both import options in TwillotImporter

3. **LLM errors**:
   - Verify API key in `.env`
   - Check provider rate limits

## Current Development Plan

See [PLAN.md](PLAN.md) for the full implementation plan including:
- Phase 1: Twitter API removal (DONE)
- Phase 2: Multimodal analysis with vision models (DONE)
- Phase 3: Value extraction improvements (DONE)
  - Smart prompts by content type
  - Content fetching from links
  - Video analysis support
  - Local media integration

### Multimodal Analysis Modes
Option 9 provides four analysis modes:
1. **Analyze single bookmark** - Deep analysis of one tweet with all its images
2. **Batch visual categorization** - Classify images by type (chart, screenshot, diagram, meme, photo, infographic)
3. **Extract insights from images** - OCR + visual understanding to extract information
4. **Compare visual content** - Group similar images and find patterns

### Smart Analysis Features
- **Content Type Detection**: Automatically detects if tweet is a list, tutorial, tool announcement, etc.
- **Link Content Fetching**: Fetches and extracts content from linked articles
- **Local Media Analysis**: Uses locally downloaded media files (12GB+) for real vision analysis
- **Video Analysis**: Full video processing with GPT-5.2 and Gemini models
