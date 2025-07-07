"""
Twitter Bookmarks Fetcher Module
"""

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from link_expander import LinkExpander

console = Console()


class BookmarksFetcher:
    """Fetch and manage Twitter bookmarks"""
    
    def __init__(self, twitter_auth):
        self.auth = twitter_auth
        self.bookmarks = []
        self.pagination_file = "pagination_state.json"
        self.link_expander = LinkExpander(timeout=10, delay=0.3)  # 0.3s delay between expansions
    
    def fetch_bookmarks(self, max_results: int = 100, expand_links: bool = True) -> List[Dict]:
        """
        Fetch bookmarks from Twitter API
        
        Args:
            max_results: Maximum results per page (max 100)
            expand_links: Whether to expand URLs in real-time
        
        Returns:
            List of bookmark dictionaries
        """
        console.print("[bold blue]Fetching bookmarks from Twitter...[/bold blue]")
        
        all_bookmarks = []
        pagination_token = self._load_pagination_token()
        page_count = 0
        
        if pagination_token:
            console.print("[yellow]Continuing from previous pagination token...[/yellow]")
            console.print("[dim]Tip: Use option 8 to reset pagination if you want to start from the beginning[/dim]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Fetching bookmarks...", total=None)
            
            while True:
                try:
                    # Get current user ID
                    user_data = self.auth.get_me()
                    if not user_data or "data" not in user_data:
                        console.print("[red]Error: Could not get user information[/red]")
                        break
                    
                    user_id = user_data["data"]["id"]
                    
                    # Fetch bookmarks
                    response = self.auth.get_bookmarks(
                        user_id=user_id,
                        max_results=max_results,
                        pagination_token=pagination_token
                    )
                    
                    if not response or "data" not in response or not response["data"]:
                        console.print("[yellow]No more bookmarks found[/yellow]")
                        break
                    
                    # Process tweets
                    tweets = response["data"]
                    users = {}
                    if "includes" in response and "users" in response["includes"]:
                        users = {u["id"]: u for u in response["includes"]["users"]}
                    
                    for tweet in tweets:
                        author_id = tweet.get("author_id")
                        author = users.get(author_id, {})
                        
                        bookmark = {
                            'id': tweet.get('id'),
                            'text': tweet.get('text'),
                            'created_at': tweet.get('created_at'),
                            'author_id': author_id,
                            'author_username': author.get('username'),
                            'author_name': author.get('name'),
                            'metrics': tweet.get('public_metrics'),
                            'entities': tweet.get('entities'),
                            'url': f"https://twitter.com/{author.get('username')}/status/{tweet.get('id')}" if author.get('username') else None
                        }
                        all_bookmarks.append(bookmark)
                    
                    page_count += 1
                    progress.update(task, description=f"Fetched {len(all_bookmarks)} bookmarks ({page_count} pages)...")
                    
                    # Check for next page
                    pagination_token = response.get("meta", {}).get('next_token')
                    if not pagination_token:
                        break
                    
                    # Rate limit: 1 request per 15 minutes for free tier
                    # Since we can't make multiple requests quickly, we'll stop after one page
                    if page_count >= 1:
                        console.print("[yellow]Rate limit reached. Free tier allows 1 request per 15 minutes.[/yellow]")
                        if pagination_token:
                            self._save_pagination_token(pagination_token)
                            console.print("[green]Pagination token saved. Next run will fetch the next 100 bookmarks.[/green]")
                        else:
                            console.print("[dim]No more bookmarks to fetch.[/dim]")
                        console.print("[yellow]To fetch more bookmarks, run the script again after 15 minutes.[/yellow]")
                        break
                    
                except Exception as e:
                    console.print(f"[red]Error fetching bookmarks: {str(e)}[/red]")
                    if "429" in str(e):
                        console.print("[yellow]Rate limit exceeded. Please wait 15 minutes before trying again.[/yellow]")
                    break
        
        # Expand links if requested and bookmarks were fetched
        if expand_links and all_bookmarks:
            console.print(f"\n[cyan]Expanding URLs in {len(all_bookmarks)} bookmarks...[/cyan]")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Expanding URLs...", total=len(all_bookmarks))
                
                for i, bookmark in enumerate(all_bookmarks):
                    try:
                        # Process URLs in the bookmark
                        self.link_expander.process_bookmark_urls(bookmark)
                        progress.update(task, advance=1, description=f"Expanded URLs in {i+1}/{len(all_bookmarks)} bookmarks")
                    except KeyboardInterrupt:
                        console.print("\n[yellow]URL expansion interrupted by user[/yellow]")
                        break
                    except Exception as e:
                        console.print(f"[yellow]Error expanding URLs for bookmark {i+1}: {str(e)}[/yellow]")
                        continue
            
            console.print("[green]URL expansion complete![/green]")
        
        self.bookmarks = all_bookmarks
        if len(all_bookmarks) > 0:
            console.print(f"[green]Successfully fetched {len(all_bookmarks)} bookmarks![/green]")
        
        return all_bookmarks
    
    def save_bookmarks(self, filename: str = "bookmarks.json", append: bool = True):
        """Save bookmarks to JSON file
        
        Args:
            filename: File to save bookmarks to
            append: If True, append to existing bookmarks. If False, overwrite.
        """
        existing_bookmarks = []
        if append and os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    existing_bookmarks = json.load(f)
            except:
                pass
        
        # Create a set of existing bookmark IDs to avoid duplicates
        existing_ids = {b.get('id') for b in existing_bookmarks if b.get('id')}
        
        # Add only new bookmarks
        new_bookmarks = [b for b in self.bookmarks if b.get('id') not in existing_ids]
        all_bookmarks = existing_bookmarks + new_bookmarks
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_bookmarks, f, ensure_ascii=False, indent=2)
        
        console.print(f"[green]Added {len(new_bookmarks)} new bookmarks to {filename}[/green]")
    
    def load_bookmarks(self, filename: str = "bookmarks.json") -> List[Dict]:
        """Load bookmarks from JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.bookmarks = json.load(f)
            console.print(f"[green]Loaded {len(self.bookmarks)} bookmarks from {filename}[/green]")
            return self.bookmarks
        except FileNotFoundError:
            console.print(f"[yellow]No saved bookmarks found at {filename}[/yellow]")
            return []
    
    def get_stats(self):
        """Display bookmark statistics"""
        if not self.bookmarks:
            console.print("[yellow]No bookmarks to analyze[/yellow]")
            return
        
        df = pd.DataFrame(self.bookmarks)
        
        # Create stats table
        table = Table(title="Bookmark Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Bookmarks", str(len(self.bookmarks)))
        table.add_row("Unique Authors", str(df['author_username'].nunique()))
        
        # Top authors
        top_authors = df['author_username'].value_counts().head(5)
        for author, count in top_authors.items():
            table.add_row(f"Top Author: @{author}", str(count))
        
        console.print(table)
    
    def export_to_markdown(self, filename: str = "bookmarks.md"):
        """Export bookmarks to markdown file"""
        if not self.bookmarks:
            console.print("[yellow]No bookmarks to export[/yellow]")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Twitter Bookmarks\n\n")
            f.write(f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            for i, bookmark in enumerate(self.bookmarks, 1):
                f.write(f"## {i}. @{bookmark.get('author_username', 'Unknown')}\n")
                f.write(f"*{bookmark.get('created_at', 'Unknown date')}*\n\n")
                f.write(f"{bookmark.get('text', '')}\n\n")
                
                # Add expanded URL information
                if bookmark.get('url_metadata'):
                    f.write("### 📎 Links in this bookmark:\n\n")
                    for link_data in bookmark['url_metadata']:
                        if link_data.get('title'):
                            f.write(f"**[{link_data['title']}]({link_data['expanded_url']})**")
                        else:
                            f.write(f"**[Link]({link_data['expanded_url']})**")
                        
                        if link_data.get('description'):
                            f.write(f"\n> {link_data['description'][:200]}...")
                        
                        content_type = link_data.get('content_type', '')
                        if content_type == 'github' and link_data.get('extra_data'):
                            extra = link_data['extra_data']
                            if extra.get('stars'):
                                f.write(f"\n> ⭐ {extra['stars']} stars")
                            if extra.get('language'):
                                f.write(f" • 💻 {extra['language']}")
                        elif content_type == 'youtube' and link_data.get('extra_data'):
                            extra = link_data['extra_data']
                            if extra.get('video_id'):
                                f.write(f"\n> 📺 YouTube Video ID: {extra['video_id']}")
                        
                        f.write("\n\n")
                
                if bookmark.get('url'):
                    f.write(f"[View on Twitter]({bookmark['url']})\n\n")
                f.write("---\n\n")
        
        console.print(f"[green]Bookmarks exported to {filename}[/green]")
    
    def _save_pagination_token(self, token: str):
        """Save pagination token for next request"""
        with open(self.pagination_file, 'w') as f:
            json.dump({"next_token": token}, f)
    
    def _load_pagination_token(self) -> Optional[str]:
        """Load saved pagination token"""
        try:
            with open(self.pagination_file, 'r') as f:
                data = json.load(f)
                return data.get("next_token")
        except FileNotFoundError:
            return None
    
    def reset_pagination(self):
        """Reset pagination to start from beginning"""
        if os.path.exists(self.pagination_file):
            os.remove(self.pagination_file)
            console.print("[green]Pagination reset. Next fetch will start from the beginning.[/green]")
    
    def unbookmark_saved(self, max_unbookmarks: int = 50):
        """Remove bookmarks that have been saved to file
        
        Args:
            max_unbookmarks: Maximum number to unbookmark (rate limit is 50 per 15 minutes)
        """
        # Load saved bookmarks
        saved_bookmarks = []
        try:
            with open("bookmarks.json", 'r', encoding='utf-8') as f:
                saved_bookmarks = json.load(f)
        except FileNotFoundError:
            console.print("[yellow]No saved bookmarks found to unbookmark[/yellow]")
            return
        
        if not saved_bookmarks:
            console.print("[yellow]No saved bookmarks to unbookmark[/yellow]")
            return
        
        # Get user ID
        try:
            user_data = self.auth.get_me()
            if not user_data or "data" not in user_data:
                console.print("[red]Error: Could not get user information[/red]")
                return
            user_id = user_data["data"]["id"]
        except Exception as e:
            console.print(f"[red]Error getting user ID: {str(e)}[/red]")
            return
        
        # Unbookmark saved tweets
        unbookmarked_count = 0
        failed_count = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Unbookmarking saved tweets (max {max_unbookmarks})...", total=min(len(saved_bookmarks), max_unbookmarks))
            
            for i, bookmark in enumerate(saved_bookmarks[:max_unbookmarks]):
                tweet_id = bookmark.get('id')
                if not tweet_id:
                    continue
                
                try:
                    response = self.auth.remove_bookmark(user_id, tweet_id)
                    if response.get("data", {}).get("bookmarked") == False:
                        unbookmarked_count += 1
                        progress.update(task, advance=1, description=f"Unbookmarked {unbookmarked_count} tweets...")
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1
                
                # Small delay to be respectful of API
                time.sleep(0.5)
        
        console.print(f"[green]Successfully unbookmarked {unbookmarked_count} tweets![/green]")
        if failed_count > 0:
            console.print(f"[yellow]Failed to unbookmark {failed_count} tweets[/yellow]")
        
