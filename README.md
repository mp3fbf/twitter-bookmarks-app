# Twitter Bookmarks Processing App

Transform your Twitter bookmarks into actionable knowledge using LLMs.

## Features

- 📚 Fetch up to 800 of your most recent Twitter bookmarks
- 💾 Save and load bookmarks locally
- 📊 View bookmark statistics
- 📝 Export bookmarks to Markdown
- 🤖 Process bookmarks with your choice of LLM:
  - OpenAI (GPT-3.5/GPT-4)
  - Anthropic (Claude)
  - Google Gemini
- 🎯 Generate:
  - Summaries
  - Study guides
  - Key insights
  - Action items

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
   - Fetch new bookmarks (limited to 1 request per 15 minutes on free tier)
   - Load previously saved bookmarks
   - View statistics about your bookmarks
   - Export to Markdown for reading
   - Process with LLM for insights

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
├── main.py              # Main application
├── twitter_auth.py      # OAuth 2.0 authentication
├── bookmarks_fetcher.py # Bookmark fetching and management
├── llm_providers.py     # Modular LLM providers
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment variables
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