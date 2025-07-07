# Twitter Bookmarks Processing App

Transform your Twitter bookmarks into actionable knowledge using LLMs.

## Features

- 📚 Fetch up to 800 of your most recent Twitter bookmarks
- 💾 Save and load bookmarks locally
- 📊 View bookmark statistics and topic analysis
- 📝 Export bookmarks to Markdown
- 🧠 **NEW: Intelligent Topic Analysis**
  - Automatic categorization into topics (AI/ML, Programming, Tools, etc.)
  - Keyword extraction and frequency analysis
  - Knowledge graph visualization (Mermaid diagrams)
  - Find connections between related bookmarks
- 🔗 **NEW: Automatic URL Expansion**
  - Expands t.co shortened links to real URLs
  - Extracts metadata (title, description, images)
  - Special handling for GitHub repos, YouTube videos
  - Rich bookmark exports with actual content
- 🤖 **Smart Processing by Topic** with your choice of LLM:
  - OpenAI (GPT-4, o3 models)
  - Anthropic (Claude)
  - Google Gemini
- 🎯 Advanced Processing Modes:
  - **Topic-specific analysis**: Process bookmarks grouped by subject
  - **Topic overview**: Get insights across all your interests
  - **Learning paths**: Create structured study plans from educational content
  - **Tool comparison**: Compare similar tools and services
  - **Deep dive analysis**: In-depth exploration of specific topics
- 🔄 Bookmark management:
  - Pagination support for fetching all bookmarks
  - Unbookmark processed tweets to avoid 800 limit

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Configure Twitter App

1. Go to [Twitter Developer Portal](https://developer.twitter.com)
2. Navigate to your project and app
3. Enable OAuth 2.0 in User Authentication Settings:
   - App Type: Web App or Native App
   - Callback URL: `http://127.0.0.1:3000/callback`
   - Website URL: `http://127.0.0.1:3000`
   - Scopes: `tweet.read`, `users.read`, `bookmark.read`, `offline.access`

### 3. Set Up Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
# At minimum, add your TWITTER_CLIENT_ID
```

### 4. Run the App

```bash
python main.py
```

## Usage

1. **First Run**: The app will open a browser for Twitter authentication
2. **Main Menu Options**:
   - **Fetch new bookmarks** - Get latest bookmarks (1 request per 15 minutes on free tier)
   - **Load saved bookmarks** - Load previously fetched bookmarks
   - **View statistics** - See bookmark counts and basic stats
   - **Export to Markdown** - Save bookmarks for reading
   - **Analyze topics** 🆕 - Discover topics, keywords, and connections in your bookmarks
   - **Smart processing** 🆕 - Process bookmarks intelligently by topic with LLM
   - **Expand URLs** 🆕 - Expand t.co links and extract metadata from websites
   - **Configure LLM** - Set up your preferred AI provider
   - **Reset pagination** - Start fetching from the beginning
   - **Unbookmark tweets** - Remove processed bookmarks from Twitter

### Smart Processing Modes

When you select "Smart processing", you can choose from:

1. **Process by Topic** - Select a specific topic (AI/ML, Programming, etc.) to analyze
2. **Topic Overview** - Get a comprehensive analysis of all your bookmark topics
3. **Learning Path** - Generate a structured curriculum from educational bookmarks
4. **Tool Comparison** - Compare similar tools and services mentioned in bookmarks
5. **Deep Dive** - In-depth analysis of a topic with related content

## LLM Configuration

The app supports multiple LLM providers. You can either:
- Set API keys in the `.env` file
- Enter them when prompted in the app

### Supported Providers:
- **OpenAI**: Uses GPT-3.5-turbo by default
- **Anthropic**: Uses Claude 3 Haiku by default  
- **Google Gemini**: Uses Gemini 1.5 Flash by default (has free tier!)

## Rate Limits

- **Free Twitter API**: 1 bookmark request every 15 minutes (up to 800 bookmarks)
- **LLM APIs**: Varies by provider and plan

## File Structure

```
twitter-bookmarks-app/
├── main.py              # Main application with smart processing
├── twitter_auth.py      # OAuth 2.0 authentication
├── bookmarks_fetcher.py # Bookmark fetching and management
├── bookmark_analyzer.py # Topic analysis and knowledge graphs 🆕
├── link_expander.py     # URL expansion and metadata extraction 🆕
├── llm_providers.py     # Modular LLM providers (supports o3!)
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment variables
├── CLAUDE.md           # Developer documentation
└── README.md           # This file
```

## Troubleshooting

1. **Authentication Issues**: 
   - Ensure your Twitter app has OAuth 2.0 enabled
   - Check that the redirect URI matches exactly

2. **Rate Limit Errors**:
   - Free tier allows only 1 request per 15 minutes
   - Wait before making another request

3. **LLM Errors**:
   - Verify your API keys are correct
   - Check your API usage limits

## Security Notes

- Never commit your `.env` file
- Tokens are saved locally in `tokens.json` (git-ignored)
- API keys are masked when entered in the app

## Future Enhancements

- [ ] Batch processing for more than 800 bookmarks
- [ ] Advanced categorization and tagging
- [ ] Export to Notion, Obsidian, etc.
- [ ] Scheduled bookmark fetching
- [ ] Web UI interface