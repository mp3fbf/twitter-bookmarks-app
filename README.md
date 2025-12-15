# Twitter Bookmarks Processing App

Transform your Twitter bookmarks into actionable knowledge using LLMs.

## Why Twillot Instead of Twitter API?

This app uses **Twillot** (a browser extension) instead of Twitter's official API because:

| Aspect | Twitter API | Twillot |
|--------|-------------|---------|
| **Setup** | Complex (Developer Portal, OAuth 2.0, credentials) | Simple (browser extension) |
| **Rate Limits** | 1 request/15 min (free tier) | None |
| **Bookmark Limit** | 100 per request, max 800 total | Unlimited |
| **Data Quality** | Truncated text, no media | Complete text, media, threads |
| **Cost** | May have future costs | Free |
| **Maintenance** | API changes, token expiration | Stable |

**Bottom line:** Twillot provides better data with zero API complexity.

## Features

- **Twillot Integration**
  - Automated browser scraping with Playwright
  - Import from Twillot export files (JSON/CSV)
  - Complete tweet text, media URLs, thread content
  - No rate limits!

- **Intelligent Topic Analysis**
  - Automatic categorization (AI/ML, Programming, Tools, etc.)
  - Keyword extraction and frequency analysis
  - Knowledge graph visualization (Mermaid diagrams)

- **URL Expansion**
  - Expands t.co shortened links
  - Extracts metadata (title, description)
  - Special handling for GitHub, YouTube

- **Smart Processing** with LLMs (OpenAI, Anthropic, Gemini)
  - Topic-specific analysis
  - Learning path generation
  - Tool comparison
  - Deep dive analysis

## Quick Start

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### 2. Get Your Bookmarks

**Option A: Import Twillot Export (Easiest)**
1. Install [Twillot extension](https://chromewebstore.google.com/detail/twillot/cedokfdbikcoefpkofjncipjjmffnknf)
2. Export your bookmarks as JSON or CSV
3. Run the app and select "Import Twillot export file"

**Option B: Automated Fetch**
1. Download Twillot extension to `./extensions/twillot/`
2. Run the app and select "Fetch via Twillot"

### 3. Configure LLM (Optional)

```bash
cp .env.example .env
# Add your preferred LLM API key
```

### 4. Run

```bash
python main.py
```

## Menu Options

```
=== Twitter Bookmarks Processing App ===

Fetch Bookmarks:
1. Fetch via Twillot (browser automation)
2. Import Twillot export file (JSON/CSV)
3. Load saved bookmarks from file

View & Export:
4. View bookmark statistics
5. Export bookmarks to Markdown
6. Expand URLs in loaded bookmarks

Analysis:
7. Analyze bookmark topics
8. Smart processing (by topic)

Settings:
9. Configure LLM provider
10. Exit
```

## Smart Processing Modes

1. **Process by Topic** - Analyze bookmarks grouped by subject
2. **Topic Overview** - Insights across all your interests
3. **Learning Path** - Structured curriculum from educational content
4. **Tool Comparison** - Compare tools and services
5. **Deep Dive** - In-depth exploration of specific topics

## LLM Providers

| Provider | Default Model | Notes |
|----------|---------------|-------|
| OpenAI | gpt-3.5-turbo | Supports GPT-4, o3 |
| Anthropic | claude-3-haiku | Supports Claude 3 Opus/Sonnet |
| Gemini | gemini-1.5-flash | Has free tier! |

## File Structure

```
twitter-bookmarks-app/
├── main.py              # Main application
├── bookmarks_fetcher.py # Bookmark management (load, save, export)
├── bookmark_analyzer.py # Topic analysis and knowledge graphs
├── link_expander.py     # URL expansion and metadata
├── twillot_scraper.py   # Twillot browser automation
├── llm_providers.py     # LLM provider factory
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
└── extensions/          # Twillot extension (for automation)
    └── twillot/
```

## Troubleshooting

**Twillot automation not working:**
- Ensure Playwright is installed: `playwright install chromium`
- Check extension is in `./extensions/twillot/`

**LLM errors:**
- Verify API key in `.env`
- Check API usage limits

## Security

- API keys stored in `.env` (git-ignored)
- No Twitter credentials needed
- Bookmarks saved locally in `bookmarks.json`
