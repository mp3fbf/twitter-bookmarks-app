"""
Twitter Bookmarks Fetcher Module

Note: Twitter API integration was removed in favor of Twillot.
See README.md for rationale. This module now handles only local
bookmark management (load, save, export, stats).
"""

import json
import os
from datetime import datetime
from typing import List, Dict
import pandas as pd
from rich.console import Console
from rich.table import Table
from link_expander import LinkExpander

console = Console()


class BookmarksFetcher:
    """Manage Twitter bookmarks (load, save, export, analyze)

    Data source: Twillot extension (via twillot_scraper.py)
    """

    def __init__(self):
        self.bookmarks = []
        self.link_expander = LinkExpander(timeout=10, delay=0.3)

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
                    f.write("### Links in this bookmark:\n\n")
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
                                f.write(f"\n> {extra['stars']} stars")
                            if extra.get('language'):
                                f.write(f" | {extra['language']}")
                        elif content_type == 'youtube' and link_data.get('extra_data'):
                            extra = link_data['extra_data']
                            if extra.get('video_id'):
                                f.write(f"\n> YouTube Video ID: {extra['video_id']}")

                        f.write("\n\n")

                if bookmark.get('url'):
                    f.write(f"[View on Twitter]({bookmark['url']})\n\n")
                f.write("---\n\n")

        console.print(f"[green]Bookmarks exported to {filename}[/green]")
