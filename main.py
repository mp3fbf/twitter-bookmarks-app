"""
Twitter Bookmarks Processing App
Main entry point
"""

import os
import sys
import argparse
from typing import List, Dict
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv

from twitter_auth import TwitterAuth, AuthServer
from bookmarks_fetcher import BookmarksFetcher
from llm_providers import LLMFactory
from bookmark_analyzer import BookmarkAnalyzer

load_dotenv()
console = Console()


class BookmarksApp:
    """Main application class"""
    
    def __init__(self):
        self.auth = TwitterAuth()
        self.fetcher = None
        self.llm = None
        self.analyzer = BookmarkAnalyzer()
    
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
        
        with console.status("[bold green]Processing with LLM...") as status:
            try:
                if choice == '1':
                    status.update("[yellow]Sending bookmarks to LLM...")
                    summary = self.llm.summarize_tweets(bookmarks)
                    status.stop()
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
    
    def analyze_topics(self):
        """Analyze and display bookmark topics"""
        console.print("\n[bold cyan]Analyzing bookmark topics...[/bold cyan]")
        
        # Analyze bookmarks
        analysis = self.analyzer.analyze_bookmarks(self.fetcher.bookmarks)
        
        # Display results
        console.print(Panel.fit(
            f"[bold]Analysis Complete![/bold]\n"
            f"Total bookmarks: {analysis['total_bookmarks']}",
            border_style="cyan"
        ))
        
        # Show categories
        console.print("\n[bold]📊 Categories Distribution:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")
        
        total = analysis['total_bookmarks']
        for category, count in sorted(analysis['categories'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            table.add_row(
                category.replace('_', ' ').title(),
                str(count),
                f"{percentage:.1f}%"
            )
        console.print(table)
        
        # Show top keywords
        console.print("\n[bold]🔍 Top Keywords:[/bold]")
        keywords = analysis['keyword_frequency'][:15]
        keyword_str = ", ".join([f"[cyan]{kw}[/cyan] ({count})" for kw, count in keywords])
        console.print(Panel(keyword_str, title="Most Frequent Terms"))
        
        # Show top authors
        console.print("\n[bold]👤 Top Authors:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Author", style="cyan")
        table.add_column("Bookmarks", justify="right")
        
        for author, count in analysis['top_authors'][:10]:
            table.add_row(f"@{author}", str(count))
        console.print(table)
        
        # Show topic summary
        console.print("\n[bold]📝 Topic Summary:[/bold]")
        console.print(self.analyzer.get_topic_summary())
        
        # Export options
        if Confirm.ask("\n[yellow]Export knowledge graph as Mermaid diagram?[/yellow]"):
            mermaid = self.analyzer.export_knowledge_graph_mermaid()
            filename = "knowledge_graph.mmd"
            with open(filename, 'w') as f:
                f.write(mermaid)
            console.print(f"[green]Knowledge graph exported to {filename}[/green]")
    
    def smart_process_bookmarks(self):
        """Process bookmarks intelligently by topic"""
        if not self.analyzer.topics:
            console.print("[yellow]Analyzing topics first...[/yellow]")
            self.analyzer.analyze_bookmarks(self.fetcher.bookmarks)
        
        console.print("\n[bold]Select Processing Mode:[/bold]")
        console.print("1. Process by specific topic")
        console.print("2. Generate topic overview")
        console.print("3. Create learning path")
        console.print("4. Compare similar tools")
        console.print("5. Deep dive analysis")
        
        mode = Prompt.ask("Enter your choice", choices=['1', '2', '3', '4', '5'])
        
        if mode == '1':
            self._process_by_topic()
        elif mode == '2':
            self._generate_topic_overview()
        elif mode == '3':
            self._create_learning_path()
        elif mode == '4':
            self._compare_tools()
        elif mode == '5':
            self._deep_dive_analysis()
    
    def _process_by_topic(self):
        """Process bookmarks for a specific topic"""
        # Show available topics
        console.print("\n[bold]Available Topics:[/bold]")
        topics = sorted(self.analyzer.topics.items(), key=lambda x: len(x[1].bookmark_ids), reverse=True)
        
        for i, (topic_name, topic) in enumerate(topics, 1):
            console.print(f"{i}. {topic_name.replace('_', ' ').title()} ({len(topic.bookmark_ids)} bookmarks)")
        
        # Let user choose topic
        choice = Prompt.ask("\nSelect topic number", 
                           choices=[str(i) for i in range(1, len(topics) + 1)])
        
        selected_topic_name, selected_topic = topics[int(choice) - 1]
        
        # Get bookmarks for this topic
        topic_bookmarks = self.analyzer.get_bookmarks_by_topic(selected_topic_name)
        
        console.print(f"\n[bold]Processing {len(topic_bookmarks)} bookmarks about {selected_topic_name}...[/bold]")
        
        # Create specialized prompt for this topic
        prompt = self._create_topic_specific_prompt(selected_topic_name, topic_bookmarks)
        
        with console.status(f"[bold green]Processing {selected_topic_name} bookmarks...") as status:
            try:
                response = self.llm.generate(prompt)
                status.stop()
                console.print(Panel(response.content, 
                                  title=f"{selected_topic_name.replace('_', ' ').title()} Analysis", 
                                  border_style="green"))
                
                # Save if requested
                if Confirm.ask("Save analysis to file?"):
                    filename = f"{selected_topic_name}_analysis.md"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"# {selected_topic_name.replace('_', ' ').title()} Analysis\n\n")
                        f.write(response.content)
                    console.print(f"[green]Saved to {filename}[/green]")
                    
            except Exception as e:
                console.print(f"[red]Error processing topic: {str(e)}[/red]")
    
    def _generate_topic_overview(self):
        """Generate an overview of all topics"""
        console.print("\n[bold]Generating comprehensive topic overview...[/bold]")
        
        # Create prompt with topic information
        topic_info = []
        for topic_name, topic in self.analyzer.topics.items():
            topic_info.append(f"- {topic_name}: {len(topic.bookmark_ids)} bookmarks, keywords: {', '.join(topic.keywords[:5])}")
        
        prompt = f"""Create a comprehensive overview of the following bookmark topics from a Twitter user's saved content:

{chr(10).join(topic_info)}

Please provide:
1. An executive summary of the user's interests
2. Key themes and patterns across topics
3. Recommendations for learning paths
4. Suggested areas for deeper exploration
5. Connections between different topics"""

        with console.status("[bold green]Generating topic overview...") as status:
            try:
                response = self.llm.generate(prompt)
                status.stop()
                console.print(Panel(response.content, title="Topic Overview", border_style="cyan"))
                
                if Confirm.ask("Save overview to file?"):
                    with open("topic_overview.md", 'w', encoding='utf-8') as f:
                        f.write("# Bookmark Topics Overview\n\n")
                        f.write(response.content)
                    console.print("[green]Saved to topic_overview.md[/green]")
                    
            except Exception as e:
                console.print(f"[red]Error generating overview: {str(e)}[/red]")
    
    def _create_learning_path(self):
        """Create a structured learning path from educational bookmarks"""
        console.print("\n[bold]Creating learning path from educational content...[/bold]")
        
        # Filter educational bookmarks
        edu_bookmarks = []
        if 'education' in self.analyzer.topics:
            edu_bookmarks.extend(self.analyzer.get_bookmarks_by_topic('education'))
        if 'ai_ml' in self.analyzer.topics:
            edu_bookmarks.extend(self.analyzer.get_bookmarks_by_topic('ai_ml'))
        if 'programming' in self.analyzer.topics:
            edu_bookmarks.extend(self.analyzer.get_bookmarks_by_topic('programming')[:10])
        
        if not edu_bookmarks:
            console.print("[yellow]No educational content found in bookmarks[/yellow]")
            return
        
        prompt = self._create_learning_path_prompt(edu_bookmarks)
        
        with console.status("[bold green]Creating learning path...") as status:
            try:
                response = self.llm.generate(prompt)
                status.stop()
                console.print(Panel(response.content, title="Personalized Learning Path", border_style="blue"))
                
                if Confirm.ask("Save learning path to file?"):
                    with open("learning_path.md", 'w', encoding='utf-8') as f:
                        f.write("# Personalized Learning Path\n\n")
                        f.write(response.content)
                    console.print("[green]Saved to learning_path.md[/green]")
                    
            except Exception as e:
                console.print(f"[red]Error creating learning path: {str(e)}[/red]")
    
    def _compare_tools(self):
        """Compare similar tools from bookmarks"""
        console.print("\n[bold]Comparing tools and services...[/bold]")
        
        # Get tool-related bookmarks
        tool_bookmarks = []
        if 'tools' in self.analyzer.topics:
            tool_bookmarks.extend(self.analyzer.get_bookmarks_by_topic('tools'))
        if 'ai_ml' in self.analyzer.topics:
            # Get AI tools specifically
            ai_bookmarks = self.analyzer.get_bookmarks_by_topic('ai_ml')
            tool_bookmarks.extend([b for b in ai_bookmarks if any(
                tool in b.get('text', '').lower() 
                for tool in ['claude', 'cursor', 'gpt', 'copilot', 'agent']
            )])
        
        if not tool_bookmarks:
            console.print("[yellow]No tools found to compare[/yellow]")
            return
        
        prompt = f"""Compare and contrast the following tools/services mentioned in these bookmarks:

{chr(10).join([f"- {b.get('text', '')[:200]}..." for b in tool_bookmarks[:15]])}

Please provide:
1. Feature comparison table
2. Use case recommendations
3. Pros and cons of each tool
4. Price/accessibility information if mentioned
5. Overall recommendations based on different needs"""

        with console.status("[bold green]Comparing tools...") as status:
            try:
                response = self.llm.generate(prompt)
                status.stop()
                console.print(Panel(response.content, title="Tool Comparison", border_style="magenta"))
                
                if Confirm.ask("Save comparison to file?"):
                    with open("tool_comparison.md", 'w', encoding='utf-8') as f:
                        f.write("# Tool Comparison\n\n")
                        f.write(response.content)
                    console.print("[green]Saved to tool_comparison.md[/green]")
                    
            except Exception as e:
                console.print(f"[red]Error comparing tools: {str(e)}[/red]")
    
    def _deep_dive_analysis(self):
        """Perform deep analysis on a specific topic"""
        console.print("\n[bold]Select topic for deep dive analysis:[/bold]")
        
        # Show topics with substantial content
        topics = [(name, topic) for name, topic in self.analyzer.topics.items() 
                  if len(topic.bookmark_ids) >= 5]
        
        if not topics:
            console.print("[yellow]No topics with enough content for deep analysis[/yellow]")
            return
        
        for i, (topic_name, topic) in enumerate(topics, 1):
            console.print(f"{i}. {topic_name.replace('_', ' ').title()} ({len(topic.bookmark_ids)} bookmarks)")
        
        choice = Prompt.ask("\nSelect topic number", 
                           choices=[str(i) for i in range(1, len(topics) + 1)])
        
        selected_topic_name, selected_topic = topics[int(choice) - 1]
        topic_bookmarks = self.analyzer.get_bookmarks_by_topic(selected_topic_name)
        
        # Get related bookmarks from other topics
        related_bookmarks = []
        for bookmark_id in selected_topic.bookmark_ids[:5]:
            related = self.analyzer.get_related_bookmarks(bookmark_id, limit=3)
            related_bookmarks.extend(related)
        
        prompt = f"""Perform a deep dive analysis on the topic: {selected_topic_name.replace('_', ' ').title()}

Main bookmarks on this topic:
{chr(10).join([f"- {b.get('text', '')}" for b in topic_bookmarks])}

Related bookmarks from other topics:
{chr(10).join([f"- {b.get('text', '')}" for b in related_bookmarks[:10]])}

Please provide:
1. Comprehensive topic overview
2. Key insights and patterns
3. Expert opinions and viewpoints mentioned
4. Practical applications and use cases
5. Future trends and predictions
6. Resources for further exploration
7. Action items and next steps"""

        with console.status(f"[bold green]Deep diving into {selected_topic_name}...") as status:
            try:
                response = self.llm.generate(prompt)
                status.stop()
                console.print(Panel(response.content, 
                                  title=f"Deep Dive: {selected_topic_name.replace('_', ' ').title()}", 
                                  border_style="cyan"))
                
                if Confirm.ask("Save analysis to file?"):
                    filename = f"deep_dive_{selected_topic_name}.md"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"# Deep Dive: {selected_topic_name.replace('_', ' ').title()}\n\n")
                        f.write(response.content)
                    console.print(f"[green]Saved to {filename}[/green]")
                    
            except Exception as e:
                console.print(f"[red]Error in deep dive: {str(e)}[/red]")
    
    def _create_topic_specific_prompt(self, topic_name: str, bookmarks: List[Dict]) -> str:
        """Create a specialized prompt based on topic"""
        base_prompt = f"Analyze these {topic_name.replace('_', ' ')} related bookmarks:\n\n"
        
        for i, bookmark in enumerate(bookmarks, 1):
            base_prompt += f"{i}. {bookmark.get('text', '')}\n\n"
        
        # Add topic-specific instructions
        if topic_name == 'ai_ml':
            base_prompt += """Focus on:
1. Latest AI/ML trends and developments
2. Practical applications and use cases
3. Tools and frameworks mentioned
4. Learning resources and tutorials
5. Expert insights and predictions"""
        elif topic_name == 'programming':
            base_prompt += """Focus on:
1. Programming best practices
2. New tools and frameworks
3. Code optimization techniques
4. Development workflows
5. Common problems and solutions"""
        elif topic_name == 'education':
            base_prompt += """Focus on:
1. Learning strategies and techniques
2. Course recommendations
3. Skill development paths
4. Resource quality assessment
5. Practical application tips"""
        else:
            base_prompt += """Provide:
1. Key themes and insights
2. Practical takeaways
3. Recommended actions
4. Related resources
5. Summary of main points"""
        
        return base_prompt
    
    def expand_bookmark_urls(self):
        """Expand URLs in already loaded bookmarks"""
        console.print("\n[bold cyan]Expanding URLs in loaded bookmarks...[/bold cyan]")
        
        # Check if bookmarks already have expanded URLs
        already_expanded = sum(1 for b in self.fetcher.bookmarks if b.get('url_metadata'))
        if already_expanded > 0:
            console.print(f"[yellow]{already_expanded} bookmarks already have expanded URLs[/yellow]")
            if not Confirm.ask("Continue and re-expand all URLs?"):
                return
        
        from rich.progress import Progress, SpinnerColumn, TextColumn
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Expanding URLs...", total=len(self.fetcher.bookmarks))
            
            for i, bookmark in enumerate(self.fetcher.bookmarks):
                try:
                    # Process URLs in the bookmark
                    self.fetcher.link_expander.process_bookmark_urls(bookmark)
                    progress.update(task, advance=1, description=f"Expanded URLs in {i+1}/{len(self.fetcher.bookmarks)} bookmarks")
                except Exception as e:
                    console.print(f"[yellow]Error expanding URLs for bookmark {i+1}: {str(e)}[/yellow]")
                    continue
        
        console.print("[green]URL expansion complete![/green]")
        
        # Ask to save
        if Confirm.ask("Save bookmarks with expanded URLs?"):
            self.fetcher.save_bookmarks()
    
    def _create_learning_path_prompt(self, bookmarks: List[Dict]) -> str:
        """Create prompt for learning path generation"""
        content = "\n\n".join([
            f"Source: @{b.get('author_username', 'unknown')}\n"
            f"Content: {b.get('text', '')}"
            for b in bookmarks
        ])
        
        return f"""Create a structured learning path based on these educational bookmarks:

{content}

Please provide:
1. Learning objectives and goals
2. Prerequisites and starting point
3. Structured curriculum (beginner → intermediate → advanced)
4. Estimated time for each section
5. Hands-on projects and exercises
6. Additional resources and references
7. Skills assessment checkpoints
8. Next steps after completion

Format as a practical, actionable learning plan."""
    
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
            console.print("5. [yellow]Analyze bookmark topics[/yellow]")
            console.print("6. [green]Smart processing (by topic)[/green]")
            console.print("7. [cyan]Expand URLs in loaded bookmarks[/cyan]")
            console.print("8. Configure LLM provider")
            console.print("9. Reset pagination (start from first bookmarks)")
            console.print("10. Unbookmark saved tweets (free up space)")
            console.print("11. Exit")
            
            choice = Prompt.ask("\nEnter your choice", choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
            
            if choice == '1':
                console.print("\n[yellow]Note: Free tier allows 1 request per 15 minutes[/yellow]")
                # Store current bookmarks in case fetch fails
                current_bookmarks = self.fetcher.bookmarks.copy() if self.fetcher.bookmarks else []
                
                # Ask about URL expansion
                expand_urls = Confirm.ask("Expand URLs during fetch?", default=True)
                new_bookmarks = self.fetcher.fetch_bookmarks(expand_links=expand_urls)
                
                if new_bookmarks:
                    if Confirm.ask("Save bookmarks to file?"):
                        self.fetcher.save_bookmarks()
                else:
                    # Restore previous bookmarks if fetch failed
                    if current_bookmarks:
                        self.fetcher.bookmarks = current_bookmarks
                        console.print("[dim]Previous bookmarks restored[/dim]")
                    
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
                # Analyze bookmark topics
                if not self.fetcher.bookmarks:
                    console.print("[yellow]No bookmarks loaded. Please fetch or load bookmarks first.[/yellow]")
                else:
                    self.analyze_topics()
                    
            elif choice == '6':
                # Smart processing by topic
                if not self.fetcher.bookmarks:
                    console.print("[yellow]No bookmarks loaded. Please fetch or load bookmarks first.[/yellow]")
                elif not self.llm:
                    console.print("[yellow]No LLM configured. Please configure first.[/yellow]")
                    self.setup_llm()
                if self.llm and self.fetcher.bookmarks:
                    self.smart_process_bookmarks()
                    
            elif choice == '7':
                # Expand URLs in loaded bookmarks
                if not self.fetcher.bookmarks:
                    console.print("[yellow]No bookmarks loaded. Please fetch or load bookmarks first.[/yellow]")
                else:
                    self.expand_bookmark_urls()
                    
            elif choice == '8':
                self.setup_llm()
                
            elif choice == '9':
                self.fetcher.reset_pagination()
                
            elif choice == '10':
                console.print("\n[bold yellow]Unbookmark Saved Tweets[/bold yellow]")
                console.print("This will remove saved tweets from your Twitter bookmarks")
                console.print("Rate limit: 50 unbookmarks per 15 minutes")
                
                if Confirm.ask("\nProceed with unbookmarking?"):
                    max_count = Prompt.ask("How many to unbookmark?", default="50")
                    try:
                        self.fetcher.unbookmark_saved(max_unbookmarks=int(max_count))
                    except ValueError:
                        console.print("[red]Invalid number[/red]")
                
            elif choice == '11':
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