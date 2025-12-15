"""
Twitter Bookmarks Processing App
Main entry point

Data Source: Twillot extension (Twitter API removed - see README.md for rationale)
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

from bookmarks_fetcher import BookmarksFetcher
from llm_providers import LLMFactory
from bookmark_analyzer import BookmarkAnalyzer
from media_linker import MediaLinker

# Twillot scraper for bookmark data
try:
    from twillot_scraper import TwillotScraper, TwillotImporter, check_playwright_installed
    TWILLOT_AVAILABLE = True
except ImportError:
    TWILLOT_AVAILABLE = False

load_dotenv()
console = Console()


class BookmarksApp:
    """Main application class"""

    def __init__(self):
        self.fetcher = BookmarksFetcher()
        self.llm = None
        self.analyzer = BookmarkAnalyzer()
        self.media_linker = None  # Will be initialized when media folder is provided

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
        console.print("\n[bold]Categories Distribution:[/bold]")
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
        console.print("\n[bold]Top Keywords:[/bold]")
        keywords = analysis['keyword_frequency'][:15]
        keyword_str = ", ".join([f"[cyan]{kw}[/cyan] ({count})" for kw, count in keywords])
        console.print(Panel(keyword_str, title="Most Frequent Terms"))

        # Show top authors
        console.print("\n[bold]Top Authors:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Author", style="cyan")
        table.add_column("Bookmarks", justify="right")

        for author, count in analysis['top_authors'][:10]:
            table.add_row(f"@{author}", str(count))
        console.print(table)

        # Show topic summary
        console.print("\n[bold]Topic Summary:[/bold]")
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

    def fetch_via_twillot(self):
        """Fetch bookmarks using Twillot automation (richer data)"""
        if not TWILLOT_AVAILABLE:
            console.print("[red]Twillot scraper not available![/red]")
            console.print("Install with: [cyan]pip install playwright && playwright install chromium[/cyan]")
            return

        if not check_playwright_installed():
            return

        console.print("\n[bold cyan]Twillot Automated Bookmark Fetcher[/bold cyan]")
        console.print("This will open a browser window with Twillot extension to fetch rich bookmark data.")
        console.print("\n[yellow]Benefits:[/yellow]")
        console.print("  - Complete tweet text (not truncated)")
        console.print("  - Media URLs (images, videos)")
        console.print("  - Thread content")
        console.print("  - Full engagement metrics")
        console.print("  - No rate limits\n")

        if not Confirm.ask("Continue with Twillot scraper?"):
            return

        headless = Confirm.ask("Run in headless mode (no visible browser)?", default=False)

        try:
            scraper = TwillotScraper(headless=headless)
            bookmarks = scraper.scrape(save_to_file=True)

            if bookmarks:
                # Convert to our standard format and merge
                console.print(f"\n[green]Fetched {len(bookmarks)} bookmarks via Twillot![/green]")

                # Update fetcher's bookmarks
                self.fetcher.bookmarks = bookmarks

                if Confirm.ask("Save to bookmarks.json?"):
                    self.fetcher.save_bookmarks()

        except Exception as e:
            console.print(f"[red]Error during Twillot scrape: {e}[/red]")

    def import_twillot_export(self):
        """Import bookmarks from a Twillot export file with optional local media linking"""
        if not TWILLOT_AVAILABLE:
            console.print("[yellow]Twillot module not loaded, using basic import...[/yellow]")

        console.print("\n[bold cyan]Import Twillot Export[/bold cyan]")
        console.print("Supported formats: JSON, CSV")

        filepath = Prompt.ask("Path to Twillot export file")

        if not os.path.exists(filepath):
            console.print(f"[red]File not found: {filepath}[/red]")
            return

        # Ask for optional media folder
        console.print("\n[dim]Twillot can export media files to a local folder.[/dim]")
        console.print("[dim]Pattern: twillot-media-files-by-date/{date}/{screen_name}-{tweet_id}-{media_id}.{ext}[/dim]")

        link_media = Confirm.ask("Link with local media files?", default=False)
        media_folder = None

        if link_media:
            default_media_path = os.path.expanduser("~/Downloads/twillot-media-files-by-date")
            media_folder = Prompt.ask(
                "Path to media folder",
                default=default_media_path if os.path.exists(default_media_path) else ""
            )
            if media_folder and os.path.exists(media_folder):
                try:
                    self.media_linker = MediaLinker(media_folder)
                    index = self.media_linker.build_index()
                    stats = self.media_linker.get_stats()
                    console.print(f"[green]Media index built: {stats['total_files']} files for {stats['tweets_with_media']} tweets[/green]")
                    console.print(f"[dim]  Images: {stats['images']}, Videos: {stats['videos']}, GIFs: {stats['gifs']}[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Could not index media folder: {e}[/yellow]")
                    self.media_linker = None
            else:
                console.print("[yellow]Media folder not found, skipping media linking[/yellow]")

        try:
            if TWILLOT_AVAILABLE:
                if filepath.endswith('.csv'):
                    bookmarks = TwillotImporter.import_csv(filepath)
                else:
                    bookmarks = TwillotImporter.import_json(filepath)
            else:
                # Basic JSON import fallback
                import json
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                bookmarks = data if isinstance(data, list) else data.get('bookmarks', [])
                console.print(f"[green]Imported {len(bookmarks)} bookmarks[/green]")

            # Enrich bookmarks with local media paths if available
            if self.media_linker and bookmarks:
                bookmarks = self.media_linker.enrich_bookmarks(bookmarks)
                with_media = sum(1 for b in bookmarks if b.get('local_media_paths'))
                console.print(f"[green]Linked {with_media} bookmarks with local media files[/green]")

            if bookmarks:
                # Show sample
                console.print("\n[bold]Sample imported bookmarks:[/bold]")
                table = Table(show_header=True)
                table.add_column("Author")
                table.add_column("Text", max_width=50)
                table.add_column("Likes")

                for b in bookmarks[:3]:
                    text = b.get('text', '')[:47] + '...' if len(b.get('text', '')) > 50 else b.get('text', '')
                    likes = b.get('metrics', {}).get('like_count', b.get('likes', 0))
                    table.add_row(
                        f"@{b.get('author_username', 'unknown')}",
                        text,
                        str(likes)
                    )
                console.print(table)

                # Merge with existing or replace
                if self.fetcher.bookmarks:
                    action = Prompt.ask(
                        "Existing bookmarks found. What to do?",
                        choices=["merge", "replace"],
                        default="merge"
                    )
                    if action == "merge":
                        existing_ids = {b.get('id') for b in self.fetcher.bookmarks}
                        new_bookmarks = [b for b in bookmarks if b.get('id') not in existing_ids]
                        self.fetcher.bookmarks.extend(new_bookmarks)
                        console.print(f"[green]Added {len(new_bookmarks)} new bookmarks[/green]")
                    else:
                        self.fetcher.bookmarks = bookmarks
                        console.print(f"[green]Replaced with {len(bookmarks)} bookmarks[/green]")
                else:
                    self.fetcher.bookmarks = bookmarks

                if Confirm.ask("Save to bookmarks.json?"):
                    self.fetcher.save_bookmarks()

        except Exception as e:
            console.print(f"[red]Error importing file: {e}[/red]")

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
3. Structured curriculum (beginner -> intermediate -> advanced)
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

    def multimodal_analysis(self):
        """Analyze bookmarks with text + image understanding"""
        console.print("\n[bold cyan]Multimodal Analysis (Text + Images)[/bold cyan]")

        # Check prerequisites
        if not self.llm:
            console.print("[yellow]No LLM configured. Please configure first.[/yellow]")
            self.setup_llm()
            if not self.llm:
                return

        if not self.fetcher.bookmarks:
            console.print("[yellow]No bookmarks loaded. Please load bookmarks first.[/yellow]")
            return

        # Filter bookmarks with local media
        bookmarks_with_media = [
            b for b in self.fetcher.bookmarks
            if b.get('local_media_paths') and len(b.get('local_media_paths', [])) > 0
        ]

        if not bookmarks_with_media:
            console.print("[yellow]No bookmarks with local media files found.[/yellow]")
            console.print("[dim]Import bookmarks with 'Link with local media files' option enabled.[/dim]")

            # Offer to setup media linker now
            if Confirm.ask("Set up media linker now?"):
                media_folder = Prompt.ask("Path to media folder (e.g., ~/Downloads/twillot-media-files-by-date)")
                if media_folder and os.path.exists(os.path.expanduser(media_folder)):
                    try:
                        self.media_linker = MediaLinker(media_folder)
                        self.media_linker.build_index()
                        self.fetcher.bookmarks = self.media_linker.enrich_bookmarks(self.fetcher.bookmarks)
                        bookmarks_with_media = [
                            b for b in self.fetcher.bookmarks
                            if b.get('local_media_paths')
                        ]
                        console.print(f"[green]Found {len(bookmarks_with_media)} bookmarks with media[/green]")
                    except Exception as e:
                        console.print(f"[red]Error setting up media linker: {e}[/red]")
                        return
                else:
                    console.print("[red]Media folder not found[/red]")
                    return
            else:
                return

        # Filter to only images (vision models work best with images)
        image_bookmarks = []
        for b in bookmarks_with_media:
            image_paths = [
                p for p in b.get('local_media_paths', [])
                if any(p.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif'])
            ]
            if image_paths:
                b_copy = b.copy()
                b_copy['image_paths'] = image_paths
                image_bookmarks.append(b_copy)

        console.print(f"\n[green]Found {len(image_bookmarks)} bookmarks with images[/green]")

        # Show processing modes
        console.print("\n[bold]Select Analysis Mode:[/bold]")
        console.print("1. Analyze single bookmark (deep analysis with images)")
        console.print("2. Batch visual categorization (classify media types)")
        console.print("3. Extract insights from images (OCR + understanding)")
        console.print("4. Compare visual content (group similar images)")

        mode = Prompt.ask("Enter your choice", choices=['1', '2', '3', '4'])

        if mode == '1':
            self._analyze_single_bookmark(image_bookmarks)
        elif mode == '2':
            self._batch_visual_categorization(image_bookmarks)
        elif mode == '3':
            self._extract_image_insights(image_bookmarks)
        elif mode == '4':
            self._compare_visual_content(image_bookmarks)

    def _analyze_single_bookmark(self, bookmarks: List[Dict]):
        """Deep analysis of a single bookmark with its images"""
        console.print("\n[bold]Select a bookmark to analyze:[/bold]")

        # Show bookmarks with images
        table = Table(show_header=True)
        table.add_column("#", style="dim")
        table.add_column("Author")
        table.add_column("Text", max_width=40)
        table.add_column("Images")

        for i, b in enumerate(bookmarks[:20], 1):
            text = b.get('text', '')[:37] + '...' if len(b.get('text', '')) > 40 else b.get('text', '')
            table.add_row(
                str(i),
                f"@{b.get('author_username', 'unknown')}",
                text,
                str(len(b.get('image_paths', [])))
            )
        console.print(table)

        choice = Prompt.ask("Select bookmark number", choices=[str(i) for i in range(1, min(21, len(bookmarks) + 1))])
        selected = bookmarks[int(choice) - 1]

        console.print(f"\n[cyan]Analyzing bookmark with {len(selected['image_paths'])} image(s)...[/cyan]")

        prompt = f"""Analyze this Twitter bookmark and its associated images.

Tweet text: {selected.get('text', '')}
Author: @{selected.get('author_username', 'unknown')}
Likes: {selected.get('metrics', {}).get('like_count', 0)}

Please provide:
1. Summary of what the tweet and images are about
2. Key information visible in the images (text, diagrams, code, etc.)
3. Context and relevance of the visual content
4. Actionable insights from combining text and images
5. Any additional resources or references mentioned"""

        with console.status("[bold green]Analyzing with vision model...") as status:
            try:
                response = self.llm.generate_with_vision(
                    prompt=prompt,
                    images=selected['image_paths'][:10],  # Limit images
                    system_prompt="You are an expert at analyzing social media content with images. Extract all valuable information from both text and visual elements."
                )
                status.stop()

                console.print(Panel(
                    response.content,
                    title=f"Analysis ({response.images_processed} images processed)",
                    border_style="green"
                ))

                if Confirm.ask("Save analysis to file?"):
                    filename = f"multimodal_analysis_{selected.get('id', 'unknown')}.md"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"# Multimodal Analysis\n\n")
                        f.write(f"**Tweet:** {selected.get('text', '')}\n\n")
                        f.write(f"**Author:** @{selected.get('author_username', 'unknown')}\n\n")
                        f.write(f"**Images:** {len(selected['image_paths'])}\n\n")
                        f.write("---\n\n")
                        f.write(response.content)
                    console.print(f"[green]Saved to {filename}[/green]")

            except Exception as e:
                console.print(f"[red]Error analyzing bookmark: {e}[/red]")

    def _batch_visual_categorization(self, bookmarks: List[Dict]):
        """Categorize bookmarks by visual content type"""
        console.print("\n[bold]Batch Visual Categorization[/bold]")
        console.print("[dim]This will classify images by type (charts, screenshots, memes, photos, etc.)[/dim]")

        # Limit for cost/time
        max_bookmarks = min(20, len(bookmarks))
        console.print(f"\nWill analyze {max_bookmarks} bookmarks with images.")

        if not Confirm.ask("Continue?"):
            return

        results = []

        with console.status("[bold green]Categorizing images...") as status:
            for i, bookmark in enumerate(bookmarks[:max_bookmarks]):
                status.update(f"[bold green]Processing {i+1}/{max_bookmarks}...")

                prompt = """Categorize this image into ONE of these types:
- chart (graphs, data visualizations)
- screenshot (code, UI, app screenshots)
- diagram (flowcharts, architecture diagrams)
- meme (humor, reaction images)
- photo (real photos of people/places/things)
- infographic (information graphics, lists)
- other

Return ONLY the category name, nothing else."""

                try:
                    response = self.llm.generate_with_vision(
                        prompt=prompt,
                        images=[bookmark['image_paths'][0]],  # First image only
                    )
                    category = response.content.strip().lower()
                    results.append({
                        'id': bookmark.get('id'),
                        'author': bookmark.get('author_username'),
                        'text': bookmark.get('text', '')[:50],
                        'category': category,
                        'image': bookmark['image_paths'][0]
                    })
                except Exception as e:
                    results.append({
                        'id': bookmark.get('id'),
                        'category': 'error',
                        'error': str(e)
                    })

        # Display results
        console.print("\n[bold]Categorization Results:[/bold]")

        # Group by category
        from collections import Counter
        categories = Counter(r['category'] for r in results)

        table = Table(show_header=True)
        table.add_column("Category")
        table.add_column("Count")
        table.add_column("Percentage")

        for category, count in categories.most_common():
            pct = (count / len(results)) * 100
            table.add_row(category.title(), str(count), f"{pct:.1f}%")

        console.print(table)

        if Confirm.ask("Save detailed results to file?"):
            import json
            with open("visual_categories.json", 'w') as f:
                json.dump(results, f, indent=2)
            console.print("[green]Saved to visual_categories.json[/green]")

    def _extract_image_insights(self, bookmarks: List[Dict]):
        """Extract text and insights from images"""
        console.print("\n[bold]Extract Insights from Images[/bold]")
        console.print("[dim]Uses OCR and visual understanding to extract information[/dim]")

        # Select bookmarks to analyze
        max_bookmarks = min(10, len(bookmarks))
        console.print(f"\nWill analyze {max_bookmarks} bookmarks with images.")

        if not Confirm.ask("Continue?"):
            return

        all_insights = []

        with console.status("[bold green]Extracting insights...") as status:
            for i, bookmark in enumerate(bookmarks[:max_bookmarks]):
                status.update(f"[bold green]Processing {i+1}/{max_bookmarks}...")

                prompt = f"""Analyze this image from a Twitter bookmark.

Tweet context: {bookmark.get('text', '')[:200]}

Extract:
1. Any text visible in the image (OCR)
2. Key data points or numbers
3. Main message or insight
4. Technical details (if code/diagram)

Be concise and focus on actionable information."""

                try:
                    response = self.llm.generate_with_vision(
                        prompt=prompt,
                        images=bookmark['image_paths'][:3],
                    )
                    all_insights.append({
                        'author': bookmark.get('author_username'),
                        'tweet': bookmark.get('text', '')[:100],
                        'insights': response.content
                    })
                except Exception as e:
                    console.print(f"[yellow]Error on bookmark {i+1}: {e}[/yellow]")

        # Display combined insights
        console.print("\n[bold]Extracted Insights:[/bold]")
        for insight in all_insights:
            console.print(Panel(
                f"[dim]@{insight['author']}:[/dim] {insight['tweet']}...\n\n{insight['insights']}",
                border_style="cyan"
            ))

        if Confirm.ask("Save all insights to file?"):
            with open("image_insights.md", 'w', encoding='utf-8') as f:
                f.write("# Extracted Image Insights\n\n")
                for insight in all_insights:
                    f.write(f"## @{insight['author']}\n\n")
                    f.write(f"> {insight['tweet']}...\n\n")
                    f.write(f"{insight['insights']}\n\n---\n\n")
            console.print("[green]Saved to image_insights.md[/green]")

    def _compare_visual_content(self, bookmarks: List[Dict]):
        """Compare and group similar visual content"""
        console.print("\n[bold]Compare Visual Content[/bold]")
        console.print("[dim]Groups bookmarks with similar visual themes[/dim]")

        # This is a simplified version - for production, you'd want embedding-based similarity
        max_bookmarks = min(15, len(bookmarks))

        prompt = f"""I have {max_bookmarks} images from Twitter bookmarks. Analyze them and group them by visual similarity/theme.

For each image, I'll provide the tweet text for context:

"""
        for i, b in enumerate(bookmarks[:max_bookmarks], 1):
            prompt += f"{i}. @{b.get('author_username', '?')}: {b.get('text', '')[:100]}...\n"

        prompt += """

Based on the images and context, create 3-5 groups of related content. For each group:
1. Group name/theme
2. Which bookmark numbers belong to it
3. What they have in common visually
4. Combined insights from this group"""

        with console.status("[bold green]Analyzing and grouping images...") as status:
            try:
                # Get first image from each bookmark
                images = [b['image_paths'][0] for b in bookmarks[:max_bookmarks]]

                response = self.llm.generate_with_vision(
                    prompt=prompt,
                    images=images,
                    system_prompt="You are an expert at visual analysis and content categorization."
                )
                status.stop()

                console.print(Panel(
                    response.content,
                    title="Visual Content Groups",
                    border_style="magenta"
                ))

                if Confirm.ask("Save grouping to file?"):
                    with open("visual_groups.md", 'w', encoding='utf-8') as f:
                        f.write("# Visual Content Groups\n\n")
                        f.write(response.content)
                    console.print("[green]Saved to visual_groups.md[/green]")

            except Exception as e:
                console.print(f"[red]Error comparing content: {e}[/red]")

    def run(self):
        """Main application loop"""
        console.print(Panel.fit(
            "[bold cyan]Twitter Bookmarks Processing App[/bold cyan]\n"
            "Transform your bookmarks into actionable knowledge\n\n"
            "[dim]Data source: Twillot extension (no Twitter API needed)[/dim]",
            border_style="cyan"
        ))

        while True:
            console.print("\n[bold]Main Menu:[/bold]")

            # Show current status
            if self.fetcher and self.fetcher.bookmarks:
                console.print(f"[dim]Bookmarks loaded: {len(self.fetcher.bookmarks)}[/dim]")
            else:
                console.print("[dim]No bookmarks loaded - use option 1, 2, or 3 to load bookmarks[/dim]")

            console.print("\n[bold underline]Fetch Bookmarks:[/bold underline]")
            console.print("1. [magenta]Fetch via Twillot (browser automation)[/magenta]")
            console.print("2. [magenta]Import Twillot export file (JSON/CSV)[/magenta]")
            console.print("3. Load saved bookmarks from file")

            console.print("\n[bold underline]View & Export:[/bold underline]")
            console.print("4. View bookmark statistics")
            console.print("5. Export bookmarks to Markdown")
            console.print("6. [cyan]Expand URLs in loaded bookmarks[/cyan]")

            console.print("\n[bold underline]Analysis:[/bold underline]")
            console.print("7. [yellow]Analyze bookmark topics[/yellow]")
            console.print("8. [green]Smart processing (by topic)[/green]")
            console.print("9. [bold magenta]Multimodal analysis (text + images)[/bold magenta]")

            console.print("\n[bold underline]Settings:[/bold underline]")
            console.print("10. Configure LLM provider")
            console.print("11. Exit")

            choice = Prompt.ask("\nEnter your choice", choices=[str(i) for i in range(1, 12)])

            if choice == '1':
                # Fetch via Twillot (automated)
                self.fetch_via_twillot()

            elif choice == '2':
                # Import Twillot export file
                self.import_twillot_export()

            elif choice == '3':
                # Load saved bookmarks
                self.fetcher.load_bookmarks()

            elif choice == '4':
                # View statistics
                if not self.fetcher.bookmarks:
                    console.print("[yellow]No bookmarks loaded. Please fetch or load bookmarks first.[/yellow]")
                else:
                    self.fetcher.get_stats()

            elif choice == '5':
                # Export to Markdown
                if not self.fetcher.bookmarks:
                    console.print("[yellow]No bookmarks loaded. Please fetch or load bookmarks first.[/yellow]")
                else:
                    self.fetcher.export_to_markdown()

            elif choice == '6':
                # Expand URLs
                if not self.fetcher.bookmarks:
                    console.print("[yellow]No bookmarks loaded. Please fetch or load bookmarks first.[/yellow]")
                else:
                    self.expand_bookmark_urls()

            elif choice == '7':
                # Analyze topics
                if not self.fetcher.bookmarks:
                    console.print("[yellow]No bookmarks loaded. Please fetch or load bookmarks first.[/yellow]")
                else:
                    self.analyze_topics()

            elif choice == '8':
                # Smart processing
                if not self.fetcher.bookmarks:
                    console.print("[yellow]No bookmarks loaded. Please fetch or load bookmarks first.[/yellow]")
                elif not self.llm:
                    console.print("[yellow]No LLM configured. Please configure first.[/yellow]")
                    self.setup_llm()
                if self.llm and self.fetcher.bookmarks:
                    self.smart_process_bookmarks()

            elif choice == '9':
                # Multimodal analysis
                self.multimodal_analysis()

            elif choice == '10':
                # Configure LLM
                self.setup_llm()

            elif choice == '11':
                # Exit
                console.print("[green]Goodbye![/green]")
                break


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(description="Twitter Bookmarks Processing App")
    args = parser.parse_args()

    app = BookmarksApp()
    app.run()


if __name__ == "__main__":
    main()
