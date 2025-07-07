# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Application Overview

This is a Twitter Bookmarks Processing App that fetches bookmarks from Twitter using OAuth 2.0 and processes them with LLMs (OpenAI, Anthropic, or Google Gemini) to generate summaries, study guides, and insights.

## Development Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the application
python main.py

# Environment setup
cp .env.example .env
# Edit .env with TWITTER_CLIENT_ID and optional LLM API keys
```

## Architecture & Key Components

### Authentication Flow (OAuth 2.0 with PKCE)
- **twitter_auth.py**: Implements OAuth 2.0 authentication without using Tweepy's OAuth handlers
- Uses direct API calls to Twitter API v2 because Tweepy forces OAuth 1.0a for user endpoints
- Key methods:
  - `generate_oauth_url()`: Creates OAuth URL with PKCE challenge
  - `exchange_code_for_token()`: Exchanges auth code for access/refresh tokens
  - `make_api_request()`: Direct API calls with Bearer token (bypasses Tweepy limitations)

### Intelligent Bookmark Analysis (NEW)
- **bookmark_analyzer.py**: Analyzes bookmarks to extract topics and create knowledge graphs
- Key features:
  - Keyword extraction with stop word filtering
  - Automatic categorization into topics (AI/ML, programming, tools, etc.)
  - Knowledge graph creation with bookmark relationships
  - Mermaid diagram export for visualization
- Key methods:
  - `analyze_bookmarks()`: Main analysis entry point
  - `extract_keywords()`: NLP-based keyword extraction
  - `create_topic_clusters()`: Groups bookmarks by topic
  - `export_knowledge_graph_mermaid()`: Visualization export

### Critical Implementation Details

1. **Tweepy Limitation Workaround**: 
   - Tweepy hardcodes OAuth 1.0a for user endpoints (get_me, get_bookmarks)
   - Solution: Use direct HTTP requests with Bearer token instead of Tweepy client
   - See `make_api_request()`, `get_me()`, and `get_bookmarks()` in twitter_auth.py

2. **Authentication Server**:
   - Uses Flask with auto-shutdown after successful auth
   - `AuthServer` class handles the OAuth callback and stops automatically
   - Tokens saved to `tokens.json` for persistence

3. **Rate Limiting**:
   - Free tier: 1 bookmark request per 15 minutes
   - App enforces this limit in `bookmarks_fetcher.py`

4. **LLM Provider Factory**:
   - `llm_providers.py` implements factory pattern for multiple LLM providers
   - Each provider inherits from `LLMProvider` abstract base class
   - Supports OpenAI (including o3!), Anthropic, and Google Gemini
   - Special handling for o1/o3 models (use `max_completion_tokens` instead of `max_tokens`)

5. **Smart Processing System**:
   - Topic-based processing instead of random bookmark selection
   - Five processing modes:
     - Process by specific topic
     - Generate topic overview
     - Create learning paths
     - Compare similar tools
     - Deep dive analysis
   - Topic-specific prompts for better LLM results

## Common Issues & Solutions

1. **"Consumer key must be string or bytes" error**:
   - Caused by Tweepy expecting OAuth 1.0a credentials
   - Fixed by using direct API calls instead of Tweepy client

2. **401 Unauthorized errors**:
   - OAuth 2.0 tokens must be passed as Bearer tokens
   - Check token format in API request headers

3. **Authentication loop**:
   - Flask server must auto-shutdown after auth
   - Check `AuthServer.shutdown_server()` implementation

4. **Reaching 800 bookmark limit**:
   - Use option 8 to unbookmark saved tweets
   - This frees up space for new bookmarks
   - Rate limit: 50 unbookmarks per 15 minutes

## Twitter App Configuration

Required OAuth 2.0 settings in Twitter Developer Portal:
- Callback URL: `http://127.0.0.1:3000/callback`
- Scopes: `tweet.read`, `users.read`, `bookmark.read`, `bookmark.write`, `offline.access`
- App can use any valid HTTPS URL for Website URL (Twitter requirement)

## Data Flow

1. User authenticates via OAuth 2.0 → tokens saved to `tokens.json`
2. Direct API calls fetch user ID and bookmarks (max 100 per request)
3. Bookmarks saved to `bookmarks.json` for offline processing
4. LLM providers process bookmarks to generate insights
5. Results exported as Markdown or displayed in console