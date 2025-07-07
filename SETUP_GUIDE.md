# Twitter Bookmarks App Setup Guide

## Step 1: Configure Twitter Developer App

1. Go to [Twitter Developer Portal](https://developer.twitter.com/en)
2. Navigate to your project: **Default project-194157371925353676768**
3. Go to your app settings: **194157371925353676768mp3fbf**

### Enable OAuth 2.0:
1. Click on the app settings (gear icon)
2. Go to **User Authentication Settings**
3. Enable **OAuth 2.0**
4. Set the following:
   - **App Type**: Web App, Single Page App, or Native App
   - **Callback URL**: `http://127.0.0.1:3000/callback` (for local development)
   - **Website URL**: `http://127.0.0.1:3000`

### Required Scopes:
- `tweet.read` - Read tweets
- `users.read` - Read user info
- `bookmark.read` - Read bookmarks
- `offline.access` - Keep access with refresh tokens

### Save Your Credentials:
After setting up OAuth 2.0, you'll need:
- **Client ID** (for OAuth 2.0)
- **Client Secret** (if using confidential client)
- **API Key** (Consumer Key)
- **API Secret** (Consumer Secret)

## Step 2: Environment Setup

Create a `.env` file in the project root with your credentials:

```env
# Twitter API Credentials
TWITTER_CLIENT_ID=your_client_id_here
TWITTER_CLIENT_SECRET=your_client_secret_here
TWITTER_API_KEY=your_api_key_here
TWITTER_API_SECRET=your_api_secret_here

# OAuth Settings
REDIRECT_URI=http://127.0.0.1:3000/callback
```

## Important Notes:
- Free tier allows 1 bookmark request every 15 minutes
- Maximum 800 bookmarks can be fetched
- Rate limits reset every 15 minutes