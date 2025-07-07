"""
Twitter OAuth 2.0 Authentication Module
"""

import os
import base64
import hashlib
import secrets
import urllib.parse
import threading
import time
from flask import Flask, request, redirect, session
import tweepy
from dotenv import load_dotenv
from werkzeug.serving import make_server

load_dotenv()


class TwitterAuth:
    """Handle Twitter OAuth 2.0 authentication"""
    
    def __init__(self):
        self.client_id = os.getenv("TWITTER_CLIENT_ID")
        self.client_secret = os.getenv("TWITTER_CLIENT_SECRET")
        self.redirect_uri = os.getenv("REDIRECT_URI", "http://127.0.0.1:3000/callback")
        
        if not self.client_id:
            raise ValueError("TWITTER_CLIENT_ID not found in environment variables")
        
        # OAuth 2.0 scopes
        self.scopes = ["tweet.read", "users.read", "bookmark.read", "bookmark.write", "offline.access"]
        
        # Store tokens
        self.access_token = None
        self.refresh_token = None
    
    def generate_oauth_url(self):
        """Generate OAuth 2.0 authorization URL with PKCE"""
        # Generate code verifier and challenge for PKCE
        code_verifier = base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8').rstrip('=')
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')
        
        # Generate state for security
        state = secrets.token_urlsafe(32)
        
        # Build authorization URL
        auth_url = "https://twitter.com/i/oauth2/authorize"
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.scopes),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256"
        }
        
        full_url = f"{auth_url}?{urllib.parse.urlencode(params)}"
        
        return full_url, state, code_verifier
    
    def exchange_code_for_token(self, code, code_verifier):
        """Exchange authorization code for access token"""
        import requests
        
        token_url = "https://api.twitter.com/2/oauth2/token"
        
        data = {
            "code": code,
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "code_verifier": code_verifier
        }
        
        # Add client secret if confidential client
        if self.client_secret:
            auth_header = base64.b64encode(
                f"{self.client_id}:{self.client_secret}".encode()
            ).decode()
            headers = {"Authorization": f"Basic {auth_header}"}
        else:
            headers = {}
        
        
        response = requests.post(token_url, data=data, headers=headers)
        
        if response.status_code == 200:
            token_data = response.json()
            
            self.access_token = token_data.get("access_token")
            self.refresh_token = token_data.get("refresh_token")
            return token_data
        else:
            raise Exception(f"Token exchange failed: {response.text}")
    
    def refresh_access_token(self):
        """Refresh the access token using refresh token"""
        import requests
        
        if not self.refresh_token:
            raise ValueError("No refresh token available")
        
        token_url = "https://api.twitter.com/2/oauth2/token"
        
        data = {
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token",
            "client_id": self.client_id
        }
        
        response = requests.post(token_url, data=data)
        
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data.get("access_token")
            if token_data.get("refresh_token"):
                self.refresh_token = token_data.get("refresh_token")
            return token_data
        else:
            raise Exception(f"Token refresh failed: {response.text}")
    
    def make_api_request(self, method, endpoint, params=None, json_data=None):
        """Make direct API request with OAuth 2.0 bearer token"""
        import requests
        
        if not self.access_token:
            raise ValueError("No access token available. Please authenticate first.")
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "User-Agent": "TwitterBookmarksApp/1.0",
            "Content-Type": "application/json"
        }
        
        url = f"https://api.twitter.com/2{endpoint}"
        
        
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_data
        )
        
        
        if response.status_code == 401:
            raise Exception(f"Unauthorized: {response.text}")
        elif response.status_code >= 400:
            raise Exception(f"API Error {response.status_code}: {response.text}")
        
        return response.json()
    
    def get_me(self):
        """Get authenticated user info using direct API call"""
        return self.make_api_request("GET", "/users/me")
    
    def get_bookmarks(self, user_id, max_results=100, pagination_token=None):
        """Get bookmarks using direct API call"""
        params = {
            "max_results": max_results,
            "tweet.fields": "created_at,author_id,public_metrics,entities",
            "expansions": "author_id",
            "user.fields": "name,username"
        }
        if pagination_token:
            params["pagination_token"] = pagination_token
        
        return self.make_api_request("GET", f"/users/{user_id}/bookmarks", params=params)
    
    def remove_bookmark(self, user_id, tweet_id):
        """Remove a bookmark using direct API call"""
        return self.make_api_request("DELETE", f"/users/{user_id}/bookmarks/{tweet_id}")
    
    def save_tokens(self, filename="tokens.json"):
        """Save tokens to file"""
        import json
        
        tokens = {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token
        }
        
        with open(filename, 'w') as f:
            json.dump(tokens, f)
    
    def load_tokens(self, filename="tokens.json"):
        """Load tokens from file"""
        import json
        
        try:
            with open(filename, 'r') as f:
                tokens = json.load(f)
                self.access_token = tokens.get("access_token")
                self.refresh_token = tokens.get("refresh_token")
                return True
        except FileNotFoundError:
            return False


# Flask app for OAuth callback
class AuthServer:
    """Flask server that can be stopped programmatically"""
    def __init__(self, auth_handler):
        self.auth_handler = auth_handler
        self.server = None
        self.thread = None
        self.auth_success = False
        
    def create_app(self):
        """Create Flask app for OAuth callback"""
        app = Flask(__name__)
        app.secret_key = secrets.token_urlsafe(32)
        
        @app.route('/')
        def index():
            # Generate OAuth URL
            auth_url, state, code_verifier = self.auth_handler.generate_oauth_url()
            
            # Store in session
            session['state'] = state
            session['code_verifier'] = code_verifier
            
            
            return f'''
            <h1>Twitter Bookmarks App</h1>
            <p>Click the link below to authenticate with Twitter:</p>
            <a href="{auth_url}">Authenticate with Twitter</a>
            <br><br>
            <small>Redirect URI: {self.auth_handler.redirect_uri}</small>
            <br>
            <small>Client ID: {self.auth_handler.client_id[:10]}...</small>
            '''
        
        @app.route('/callback')
        def callback():
            
            # Get code and state from query params
            code = request.args.get('code')
            state = request.args.get('state')
            error = request.args.get('error')
            
            if error:
                self.shutdown_server()
                return f"Authentication error: {error} - {request.args.get('error_description', '')}", 400
            
            # Verify state
            if state != session.get('state'):
                self.shutdown_server()
                return "State mismatch. Possible CSRF attack.", 400
            
            # Exchange code for token
            try:
                code_verifier = session.get('code_verifier')
                token_data = self.auth_handler.exchange_code_for_token(code, code_verifier)
                self.auth_handler.save_tokens()
                self.auth_success = True
                
                # Schedule server shutdown
                threading.Timer(1.0, self.shutdown_server).start()
                
                return f'''
                <h1>Authentication Successful!</h1>
                <p>Access token obtained. You can now close this window.</p>
                <p>Tokens have been saved to tokens.json</p>
                <p><strong>Returning to the app...</strong></p>
                <script>
                    setTimeout(function() {{
                        window.close();
                    }}, 2000);
                </script>
                '''
            except Exception as e:
                self.shutdown_server()
                return f"Authentication failed: {str(e)}", 400
        
        return app
    
    def shutdown_server(self):
        """Shutdown the Flask server"""
        if self.server:
            self.server.shutdown()
    
    def run(self, host='127.0.0.1', port=3000):
        """Run the server"""
        app = self.create_app()
        self.server = make_server(host, port, app)
        self.server.serve_forever()
        
        
def create_auth_app(auth_handler):
    """Create Flask app for OAuth callback (legacy compatibility)"""
    server = AuthServer(auth_handler)
    return server.create_app()