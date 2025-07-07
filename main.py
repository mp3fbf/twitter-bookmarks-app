"""
Twitter Bookmarks Processing App
Main entry point
"""

import os
import sys
import argparse
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from dotenv import load_dotenv

from twitter_auth import TwitterAuth, AuthServer
from bookmarks_fetcher import BookmarksFetcher
from llm_providers import LLMFactory

load_dotenv()
console = Console()


class BookmarksApp:
    """Main application class"""
    
    def __init__(self):
        self.auth = TwitterAuth()
        self.fetcher = None
        self.llm = None
    
    def setup_auth(self):
        """Setup Twitter authentication"""
        # Try to load existing tokens
        if self.auth.load_tokens():
            console.print("[green]Loaded existing authentication tokens[/green]")
            try:
                # Test if tokens are still valid
                user_data = self.auth.get_me()
                if user_data and "data" in user_data:
                    username = user_data["data"].get("username", "Unknown")
                    console.print(f"[green]Authenticated as @{username}[/green]")
                    return True
            except Exception as e:
                console.print(f"[yellow]Existing tokens expired or invalid[/yellow]")
                # Try to refresh
                try:
                    self.auth.refresh_access_token()
                    self.auth.save_tokens()
                    console.print("[green]Successfully refreshed tokens[/green]")
                    return True
                except:
                    console.print("[red]Failed to refresh tokens. Need to re-authenticate.[/red]")
        
        # Need new authentication
        console.print("[yellow]Need to authenticate with Twitter[/yellow]")
        console.print("Starting OAuth server on http://127.0.0.1:3000")
        console.print("Please visit the URL and authenticate with Twitter")
        console.print("[dim]The server will automatically shut down after authentication[/dim]")
        
        # Create and run auth server
        auth_server = AuthServer(self.auth)
        
        try:
            auth_server.run(host='127.0.0.1', port=3000)
        except KeyboardInterrupt:
            console.print("\n[yellow]Authentication cancelled by user[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Authentication server error: {str(e)}[/red]")
        
        # Check if authentication was successful
        if auth_server.auth_success:
            console.print("\n[green]Authentication completed successfully![/green]")
            # Verify tokens were saved
            if self.auth.load_tokens():
                try:
                    user_data = self.auth.get_me()
                    if user_data and "data" in user_data:
                        username = user_data["data"].get("username", "Unknown")
                        console.print(f"[green]Authenticated as @{username}[/green]")
                        return True
                except Exception as e:
                    console.print(f"[red]Error verifying authentication: {str(e)}[/red]")
            return True
        else:
            # Even if server was interrupted, check if tokens exist
            if os.path.exists("tokens.json") and self.auth.load_tokens():
                console.print("[yellow]Checking for saved tokens...[/yellow]")
                try:
                    user_data = self.auth.get_me()
                    if user_data and "data" in user_data:
                        username = user_data["data"].get("username", "Unknown")
                        console.print(f"[green]Found valid authentication for @{username}[/green]")
                        return True
                except:
                    pass
            
            console.print("[red]Authentication was not completed[/red]")
            return False
    
    def setup_llm(self):
        """Setup LLM provider"""
        providers = ['openai', 'anthropic', 'gemini', 'none']
        
        console.print("\n[bold]Select LLM Provider:[/bold]")
        for i, provider in enumerate(providers, 1):
            console.print(f"{i}. {provider.capitalize()}")
        
        choice = Prompt.ask("Enter your choice", choices=[str(i) for i in range(1, len(providers) + 1)])
        provider = providers[int(choice) - 1]
        
        if provider == 'none':
            console.print("[yellow]Skipping LLM setup[/yellow]")
            return
        
        # Check for API key
        env_keys = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'gemini': 'GEMINI_API_KEY'
        }
        
        env_key = env_keys.get(provider)
        if not os.getenv(env_key):
            api_key = Prompt.ask(f"Enter your {provider.upper()} API key", password=True)
            if not api_key:
                console.print("[yellow]No API key provided. Skipping LLM setup.[/yellow]")
                return
        else:
            api_key = None  # Will use from environment
        
        # Ask for custom model if desired
        use_custom = Confirm.ask("Use a custom model?", default=False)
        custom_model = None
        if use_custom:
            if provider == 'openai':
                console.print("[dim]Available models: gpt-4-turbo-preview, gpt-4o, gpt-3.5-turbo, o3-mini, o3, o1-mini, o1[/dim]")
            elif provider == 'anthropic':
                console.print("[dim]Available models: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307[/dim]")
            elif provider == 'gemini':
                console.print("[dim]Available models: gemini-1.5-pro, gemini-1.5-flash[/dim]")
            custom_model = Prompt.ask("Enter model name")
        
        try:
            self.llm = LLMFactory.create(provider, api_key=api_key, model=custom_model)
            model_info = f" (model: {custom_model})" if custom_model else ""
            console.print(f"[green]Successfully configured {provider.capitalize()} LLM{model_info}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to setup LLM: {str(e)}[/red]")
    
    def process_bookmarks(self):
        """Process bookmarks with LLM"""
        if not self.llm:
            console.print("[yellow]No LLM configured. Skipping processing.[/yellow]")
            return
        
        if not self.fetcher or not self.fetcher.bookmarks:
            console.print("[yellow]No bookmarks to process[/yellow]")
            return
        
        console.print("\n[bold]Processing Options:[/bold]")
        console.print("1. Summarize all bookmarks")
        console.print("2. Generate study guide")
        console.print("3. Extract key insights")
        console.print("4. Create action items")
        
        choice = Prompt.ask("Enter your choice", choices=['1', '2', '3', '4'])
        
        # Group bookmarks by theme (simple grouping by first 10)
        bookmarks = self.fetcher.bookmarks[:10]  # Process first 10 for demo
        
        with console.status("[bold green]Processing with LLM..."):
            try:
                if choice == '1':
                    summary = self.llm.summarize_tweets(bookmarks)
                    console.print(Panel(summary, title="Summary", border_style="green"))
                    
                elif choice == '2':
                    prompt = self._create_study_guide_prompt(bookmarks)
                    response = self.llm.generate(prompt)
                    console.print(Panel(response.content, title="Study Guide", border_style="blue"))
                    
                elif choice == '3':
                    prompt = self._create_insights_prompt(bookmarks)
                    response = self.llm.generate(prompt)
                    console.print(Panel(response.content, title="Key Insights", border_style="cyan"))
                    
                elif choice == '4':
                    prompt = self._create_action_items_prompt(bookmarks)
                    response = self.llm.generate(prompt)
                    console.print(Panel(response.content, title="Action Items", border_style="yellow"))
                
                # Save processed content
                if Confirm.ask("Save processed content to file?"):
                    filename = Prompt.ask("Enter filename", default="processed_bookmarks.md")
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"# Processed Bookmarks\n\n")
                        f.write(response.content if choice != '1' else summary)
                    console.print(f"[green]Saved to {filename}[/green]")
                    
            except Exception as e:
                console.print(f"[red]Error processing bookmarks: {str(e)}[/red]")
    
    def _create_study_guide_prompt(self, bookmarks):
        """Create prompt for study guide generation"""
        tweets = "\n\n".join([f"Tweet: {b.get('text', '')}" for b in bookmarks])
        return f"""Create a comprehensive study guide from these Twitter bookmarks. 
        Organize the content into clear sections with headings, key concepts, and practical examples:

{tweets}

Format as a structured study guide with:
1. Overview
2. Key Concepts (with explanations)
3. Practical Tips
4. Action Steps
5. Further Learning Resources"""
    
    def _create_insights_prompt(self, bookmarks):
        """Create prompt for insights extraction"""
        tweets = "\n\n".join([f"Tweet: {b.get('text', '')}" for b in bookmarks])
        return f"""Extract the most valuable insights from these Twitter bookmarks.
        Focus on unique, actionable, and thought-provoking ideas:

{tweets}

Provide:
1. Top 5 Key Insights (with brief explanations)
2. Common Themes
3. Surprising or Counter-intuitive Points
4. Most Actionable Advice"""
    
    def _create_action_items_prompt(self, bookmarks):
        """Create prompt for action items generation"""
        tweets = "\n\n".join([f"Tweet: {b.get('text', '')}" for b in bookmarks])
        return f"""Extract concrete action items from these Twitter bookmarks.
        Focus on specific, implementable tasks:

{tweets}

Create a prioritized list of action items with:
1. Task description
2. Expected outcome
3. Time/effort required
4. Priority level (High/Medium/Low)"""
    
    def run(self):
        """Main application loop"""
        console.print(Panel.fit(
            "[bold cyan]Twitter Bookmarks Processing App[/bold cyan]\n"
            "Transform your bookmarks into actionable knowledge",
            border_style="cyan"
        ))
        
        # Setup authentication
        if not self.setup_auth():
            console.print("[red]Authentication failed. Exiting.[/red]")
            return
        
        # Create fetcher
        self.fetcher = BookmarksFetcher(self.auth)
        
        while True:
            console.print("\n[bold]Main Menu:[/bold]")
            
            # Show current status
            if self.fetcher and self.fetcher.bookmarks:
                console.print(f"[dim]Bookmarks loaded: {len(self.fetcher.bookmarks)}[/dim]")
            else:
                console.print("[dim]No bookmarks loaded - use option 1 or 2 to load bookmarks[/dim]")
            
            console.print("\n1. Fetch new bookmarks from Twitter")
            console.print("2. Load saved bookmarks from file")
            console.print("3. View bookmark statistics")
            console.print("4. Export bookmarks to Markdown")
            console.print("5. Process bookmarks with LLM")
            console.print("6. Configure LLM provider")
            console.print("7. Reset pagination (start from first bookmarks)")
            console.print("8. Unbookmark saved tweets (free up space)")
            console.print("9. Exit")
            
            choice = Prompt.ask("\nEnter your choice", choices=['1', '2', '3', '4', '5', '6', '7', '8', '9'])
            
            if choice == '1':
                console.print("\n[yellow]Note: Free tier allows 1 request per 15 minutes[/yellow]")
                bookmarks = self.fetcher.fetch_bookmarks()
                if bookmarks and Confirm.ask("Save bookmarks to file?"):
                    self.fetcher.save_bookmarks()
                    
            elif choice == '2':
                self.fetcher.load_bookmarks()
                
            elif choice == '3':
                if not self.fetcher.bookmarks:
                    console.print("[yellow]No bookmarks loaded. Please fetch or load bookmarks first.[/yellow]")
                else:
                    self.fetcher.get_stats()
                
            elif choice == '4':
                if not self.fetcher.bookmarks:
                    console.print("[yellow]No bookmarks loaded. Please fetch or load bookmarks first.[/yellow]")
                else:
                    self.fetcher.export_to_markdown()
                
            elif choice == '5':
                if not self.fetcher.bookmarks:
                    console.print("[yellow]No bookmarks loaded. Please fetch or load bookmarks first.[/yellow]")
                elif not self.llm:
                    console.print("[yellow]No LLM configured. Please configure first.[/yellow]")
                    self.setup_llm()
                if self.llm and self.fetcher.bookmarks:
                    self.process_bookmarks()
                    
            elif choice == '6':
                self.setup_llm()
                
            elif choice == '7':
                self.fetcher.reset_pagination()
                
            elif choice == '8':
                console.print("\n[bold yellow]Unbookmark Saved Tweets[/bold yellow]")
                console.print("This will remove saved tweets from your Twitter bookmarks")
                console.print("Rate limit: 50 unbookmarks per 15 minutes")
                
                if Confirm.ask("\nProceed with unbookmarking?"):
                    max_count = Prompt.ask("How many to unbookmark?", default="50")
                    try:
                        self.fetcher.unbookmark_saved(max_unbookmarks=int(max_count))
                    except ValueError:
                        console.print("[red]Invalid number[/red]")
                
            elif choice == '9':
                console.print("[green]Goodbye![/green]")
                break


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(description="Twitter Bookmarks Processing App")
    parser.add_argument('--auth-only', action='store_true', help='Only run authentication')
    args = parser.parse_args()
    
    app = BookmarksApp()
    
    if args.auth_only:
        app.setup_auth()
    else:
        app.run()


if __name__ == "__main__":
    main()