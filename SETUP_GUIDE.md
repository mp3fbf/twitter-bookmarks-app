# Twitter Bookmarks App - Setup Guide

This app uses **Twillot** to fetch your Twitter bookmarks. No Twitter API credentials needed!

## Quick Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Install Playwright browser (required for Twillot automation)
playwright install chromium
```

### 2. Get the Twillot Extension

**Option A: Download from Chrome Web Store (Recommended)**
1. Install [Twillot](https://chromewebstore.google.com/detail/twillot/cedokfdbikcoefpkofjncipjjmffnknf) in Chrome
2. Export your bookmarks as JSON or CSV
3. Use the app's "Import Twillot export file" option

**Option B: For Automated Scraping**
1. Download Twillot extension
2. Extract to `./extensions/twillot/`
3. Use the app's "Fetch via Twillot" option

### 3. Configure LLM (Optional)

For AI-powered analysis, add your API key to `.env`:

```bash
cp .env.example .env
```

Edit `.env` with your preferred LLM:
```env
OPENAI_API_KEY=your_key_here
# or
ANTHROPIC_API_KEY=your_key_here
# or
GEMINI_API_KEY=your_key_here
```

### 4. Run the App

```bash
python main.py
```

## Usage

### Fetching Bookmarks

1. **Import Twillot Export** (Easiest)
   - Export bookmarks from Twillot extension (JSON/CSV)
   - Select option 2 in the app menu
   - Point to your export file

2. **Automated Fetch via Twillot**
   - Requires Twillot extension in `./extensions/twillot/`
   - Select option 1 in the app menu
   - Browser will open, log in to Twitter
   - App auto-scrolls and extracts all bookmarks

### Processing Bookmarks

After loading bookmarks:
- **View statistics** - See counts and top authors
- **Analyze topics** - Discover categories and keywords
- **Smart processing** - Use LLM for insights, learning paths, comparisons
- **Export to Markdown** - Save for reading

## Why Twillot Instead of Twitter API?

We removed Twitter API integration because:

| Aspect | Twitter API | Twillot |
|--------|-------------|---------|
| Setup | Complex (Developer Portal, OAuth) | Simple (browser extension) |
| Rate Limits | 1 request/15 min | None |
| Data Quality | Truncated text, no media | Complete data |
| Cost | May have future costs | Free |

Twillot provides better data with simpler setup. See [README.md](README.md) for full details.
