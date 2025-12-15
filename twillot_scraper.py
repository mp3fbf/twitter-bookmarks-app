"""
Twillot Scraper Module
Automates Twitter bookmark extraction using Playwright + Twillot Chrome extension
"""

import json
import os
import time
import zipfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.prompt import Prompt, Confirm

console = Console()

# Check if playwright is available
try:
    from playwright.sync_api import sync_playwright, BrowserContext, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


@dataclass
class TwillotBookmark:
    """Normalized bookmark structure from Twillot export"""
    id: str
    text: str
    created_at: str
    author_id: Optional[str] = None
    author_username: Optional[str] = None
    author_name: Optional[str] = None
    author_avatar: Optional[str] = None
    url: Optional[str] = None

    # Engagement metrics
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    views: int = 0
    bookmarks: int = 0

    # Media
    media_urls: Optional[List[str]] = None
    media_types: Optional[List[str]] = None  # image, video, gif

    # Thread info
    is_thread: bool = False
    thread_tweets: Optional[List[Dict]] = None

    # Quote tweet
    quoted_tweet: Optional[Dict] = None

    # Links
    urls: Optional[List[Dict]] = None  # expanded URLs with metadata

    # Twillot extras
    tags: Optional[List[str]] = None
    folder: Optional[str] = None
    notes: Optional[str] = None

    # Raw data for debugging
    raw_data: Optional[Dict] = None


class TwillotExtensionManager:
    """Manages Twillot Chrome extension setup"""

    EXTENSION_ID = "cedokfdbikcoefpkofjncipjjmffnknf"  # Twillot Chrome Web Store ID
    EXTENSION_URL = f"https://clients2.google.com/service/update2/crx?response=redirect&prodversion=120.0.0.0&x=id%3D{EXTENSION_ID}%26installsource%3Dondemand%26uc"

    def __init__(self, extensions_dir: str = "./extensions"):
        self.extensions_dir = Path(extensions_dir)
        self.twillot_dir = self.extensions_dir / "twillot"

    def is_installed(self) -> bool:
        """Check if Twillot extension is installed locally"""
        manifest_path = self.twillot_dir / "manifest.json"
        return manifest_path.exists()

    def get_extension_path(self) -> str:
        """Get path to the extracted extension"""
        return str(self.twillot_dir.absolute())

    def setup_extension_manually(self) -> bool:
        """Guide user to manually set up the extension"""
        console.print("\n[bold yellow]Twillot Extension Setup Required[/bold yellow]\n")

        console.print("To use the automated scraper, you need to provide the Twillot extension files.")
        console.print("\n[cyan]Option 1: Export from Chrome[/cyan]")
        console.print("1. Open Chrome and go to: chrome://extensions/")
        console.print("2. Enable 'Developer mode' (top right)")
        console.print("3. Find 'Twillot' extension")
        console.print("4. Note the extension ID (e.g., cedokfdbikcoefpkofjncipjjmffnknf)")
        console.print("5. Go to: ~/.config/google-chrome/Default/Extensions/[EXTENSION_ID]/")
        console.print("6. Copy the version folder contents to: ./extensions/twillot/")

        console.print("\n[cyan]Option 2: Download CRX manually[/cyan]")
        console.print("1. Visit: https://www.crx4chrome.com/ or similar")
        console.print("2. Search for 'Twillot'")
        console.print("3. Download the .crx file")
        console.print("4. Rename .crx to .zip and extract")
        console.print("5. Move contents to: ./extensions/twillot/")

        # Create directory structure
        self.twillot_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[green]Created directory: {self.twillot_dir}[/green]")
        console.print("Please add the extension files and run again.")

        return False

    def setup_from_crx(self, crx_path: str) -> bool:
        """Extract extension from CRX file"""
        try:
            crx_path = Path(crx_path)
            if not crx_path.exists():
                console.print(f"[red]CRX file not found: {crx_path}[/red]")
                return False

            # CRX files are basically ZIP files with a header
            # Try to extract as ZIP first
            self.twillot_dir.mkdir(parents=True, exist_ok=True)

            # Read the CRX and find the ZIP start
            with open(crx_path, 'rb') as f:
                data = f.read()

            # CRX3 format: magic(4) + version(4) + header_size(4) + header + zip
            # Find PK signature (ZIP start)
            zip_start = data.find(b'PK\x03\x04')
            if zip_start == -1:
                console.print("[red]Invalid CRX file format[/red]")
                return False

            # Write ZIP portion to temp file
            temp_zip = self.extensions_dir / "temp_twillot.zip"
            with open(temp_zip, 'wb') as f:
                f.write(data[zip_start:])

            # Extract ZIP
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall(self.twillot_dir)

            # Clean up
            temp_zip.unlink()

            console.print(f"[green]Extension extracted to: {self.twillot_dir}[/green]")
            return True

        except Exception as e:
            console.print(f"[red]Error extracting CRX: {e}[/red]")
            return False


class TwillotScraper:
    """Automates bookmark extraction using Playwright + Twillot"""

    def __init__(
        self,
        user_data_dir: str = "./twitter-browser-profile",
        headless: bool = False,
        slow_mo: int = 100,
        export_dir: str = "./exports"
    ):
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright is not installed. Run: pip install playwright && playwright install chromium"
            )

        self.user_data_dir = Path(user_data_dir)
        self.headless = headless
        self.slow_mo = slow_mo
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

        self.extension_manager = TwillotExtensionManager()
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    def _ensure_extension(self) -> bool:
        """Ensure Twillot extension is available"""
        if not self.extension_manager.is_installed():
            return self.extension_manager.setup_extension_manually()
        return True

    def _launch_browser(self, playwright) -> BrowserContext:
        """Launch browser with Twillot extension loaded"""
        extension_path = self.extension_manager.get_extension_path()

        console.print(f"[cyan]Loading extension from: {extension_path}[/cyan]")

        context = playwright.chromium.launch_persistent_context(
            str(self.user_data_dir),
            headless=self.headless,
            slow_mo=self.slow_mo,
            args=[
                f"--disable-extensions-except={extension_path}",
                f"--load-extension={extension_path}",
                "--disable-blink-features=AutomationControlled",
            ],
            viewport={"width": 1280, "height": 900},
            locale="en-US",
        )

        return context

    def _wait_for_login(self, page: Page, timeout: int = 300) -> bool:
        """Wait for user to complete Twitter login if needed"""
        page.goto("https://twitter.com/home")

        # Check if already logged in
        try:
            page.wait_for_selector('[data-testid="primaryColumn"]', timeout=5000)
            console.print("[green]Already logged in to Twitter![/green]")
            return True
        except:
            pass

        console.print("\n[yellow]Please log in to Twitter in the browser window...[/yellow]")
        console.print(f"[dim]Waiting up to {timeout} seconds for login...[/dim]")

        try:
            page.wait_for_selector('[data-testid="primaryColumn"]', timeout=timeout * 1000)
            console.print("[green]Login successful![/green]")
            time.sleep(2)  # Let the page fully load
            return True
        except:
            console.print("[red]Login timeout. Please try again.[/red]")
            return False

    def _navigate_to_bookmarks(self, page: Page) -> bool:
        """Navigate to Twitter bookmarks page"""
        console.print("[cyan]Navigating to bookmarks...[/cyan]")
        page.goto("https://twitter.com/i/bookmarks")

        try:
            # Wait for bookmarks to load
            page.wait_for_selector('[data-testid="cellInnerDiv"]', timeout=10000)
            console.print("[green]Bookmarks page loaded![/green]")
            return True
        except:
            console.print("[yellow]No bookmarks found or page didn't load[/yellow]")
            return False

    def _open_twillot_popup(self, page: Page) -> Optional[Page]:
        """Open Twillot extension popup"""
        # Get extension ID from the loaded extensions
        extension_id = self.extension_manager.EXTENSION_ID

        # Try to open extension popup
        popup_url = f"chrome-extension://{extension_id}/popup.html"

        try:
            popup_page = self.context.new_page()
            popup_page.goto(popup_url)
            time.sleep(2)
            return popup_page
        except Exception as e:
            console.print(f"[yellow]Could not open extension popup: {e}[/yellow]")
            return None

    def _trigger_twillot_sync(self, page: Page) -> bool:
        """Trigger Twillot to sync bookmarks"""
        console.print("[cyan]Triggering Twillot sync...[/cyan]")

        # Twillot automatically syncs when on bookmarks page
        # We just need to scroll to load more bookmarks

        # Scroll down to trigger loading more bookmarks
        last_height = 0
        scroll_count = 0
        max_scrolls = 50  # Limit scrolling

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading bookmarks...", total=None)

            while scroll_count < max_scrolls:
                # Scroll down
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(1.5)

                # Get new height
                new_height = page.evaluate("document.body.scrollHeight")

                scroll_count += 1
                progress.update(task, description=f"Scrolling... ({scroll_count} pages)")

                if new_height == last_height:
                    # No more content to load
                    break

                last_height = new_height

        console.print(f"[green]Finished loading bookmarks after {scroll_count} scrolls[/green]")
        return True

    def _export_via_twillot(self, popup_page: Page) -> Optional[str]:
        """Export bookmarks via Twillot interface"""
        console.print("[cyan]Exporting via Twillot...[/cyan]")

        try:
            # Navigate to export section
            # This depends on Twillot's UI structure
            # Look for export button or link

            # Try common selectors
            export_selectors = [
                'text="Export"',
                'button:has-text("Export")',
                'a:has-text("Export")',
                '[data-action="export"]',
            ]

            for selector in export_selectors:
                try:
                    popup_page.click(selector, timeout=3000)
                    break
                except:
                    continue

            time.sleep(2)

            # Look for JSON export option
            json_selectors = [
                'text="JSON"',
                'button:has-text("JSON")',
                '[data-format="json"]',
            ]

            for selector in json_selectors:
                try:
                    popup_page.click(selector, timeout=3000)
                    break
                except:
                    continue

            # Wait for download
            time.sleep(3)

            # Return path to downloaded file
            export_path = self.export_dir / f"twillot_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            return str(export_path)

        except Exception as e:
            console.print(f"[red]Error during export: {e}[/red]")
            return None

    def _scrape_bookmarks_directly(self, page: Page) -> List[Dict]:
        """Scrape bookmarks directly from the page DOM"""
        console.print("[cyan]Scraping bookmarks from page...[/cyan]")

        bookmarks = []

        # Get all tweet articles
        tweets = page.query_selector_all('article[data-testid="tweet"]')

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Extracting tweets...", total=len(tweets))

            for tweet in tweets:
                try:
                    bookmark = self._extract_tweet_data(tweet)
                    if bookmark:
                        bookmarks.append(bookmark)
                except Exception as e:
                    console.print(f"[yellow]Error extracting tweet: {e}[/yellow]")

                progress.advance(task)

        console.print(f"[green]Extracted {len(bookmarks)} bookmarks[/green]")
        return bookmarks

    def _extract_tweet_data(self, tweet_element) -> Optional[Dict]:
        """Extract data from a tweet element"""
        try:
            data = {}

            # Get tweet text
            text_elem = tweet_element.query_selector('[data-testid="tweetText"]')
            data['text'] = text_elem.inner_text() if text_elem else ""

            # Get author info
            user_elem = tweet_element.query_selector('[data-testid="User-Name"]')
            if user_elem:
                # Parse username and display name
                links = user_elem.query_selector_all('a')
                if links:
                    href = links[0].get_attribute('href')
                    if href:
                        data['author_username'] = href.strip('/')

                spans = user_elem.query_selector_all('span')
                if spans:
                    data['author_name'] = spans[0].inner_text()

            # Get tweet URL
            time_elem = tweet_element.query_selector('time')
            if time_elem:
                parent_link = time_elem.evaluate('el => el.closest("a")?.href')
                if parent_link:
                    data['url'] = parent_link
                    # Extract tweet ID from URL
                    if '/status/' in parent_link:
                        data['id'] = parent_link.split('/status/')[-1].split('?')[0]

                # Get timestamp
                data['created_at'] = time_elem.get_attribute('datetime')

            # Get engagement metrics
            metrics_group = tweet_element.query_selector('[role="group"]')
            if metrics_group:
                # Reply count
                reply_btn = metrics_group.query_selector('[data-testid="reply"]')
                if reply_btn:
                    count_text = reply_btn.inner_text().strip()
                    data['replies'] = self._parse_count(count_text)

                # Retweet count
                retweet_btn = metrics_group.query_selector('[data-testid="retweet"]')
                if retweet_btn:
                    count_text = retweet_btn.inner_text().strip()
                    data['retweets'] = self._parse_count(count_text)

                # Like count
                like_btn = metrics_group.query_selector('[data-testid="like"]')
                if like_btn:
                    count_text = like_btn.inner_text().strip()
                    data['likes'] = self._parse_count(count_text)

                # View count (if available)
                view_elem = metrics_group.query_selector('a[href*="/analytics"]')
                if view_elem:
                    count_text = view_elem.inner_text().strip()
                    data['views'] = self._parse_count(count_text)

            # Get media
            media_container = tweet_element.query_selector('[data-testid="tweetPhoto"]')
            if media_container:
                images = media_container.query_selector_all('img')
                data['media_urls'] = [img.get_attribute('src') for img in images if img.get_attribute('src')]
                data['media_types'] = ['image'] * len(data.get('media_urls', []))

            # Get video if present
            video_elem = tweet_element.query_selector('video')
            if video_elem:
                video_src = video_elem.get_attribute('src')
                if video_src:
                    data['media_urls'] = data.get('media_urls', []) + [video_src]
                    data['media_types'] = data.get('media_types', []) + ['video']

            # Get URLs in tweet
            links = tweet_element.query_selector_all('[data-testid="tweetText"] a')
            urls = []
            for link in links:
                href = link.get_attribute('href')
                if href and not href.startswith('/') and 'twitter.com' not in href:
                    urls.append({
                        'url': href,
                        'display_text': link.inner_text()
                    })
            if urls:
                data['urls'] = urls

            # Get avatar
            avatar_img = tweet_element.query_selector('[data-testid="Tweet-User-Avatar"] img')
            if avatar_img:
                data['author_avatar'] = avatar_img.get_attribute('src')

            return data if data.get('id') or data.get('text') else None

        except Exception as e:
            return None

    def _parse_count(self, count_str: str) -> int:
        """Parse count strings like '1.2K', '5M' into integers"""
        if not count_str:
            return 0

        count_str = count_str.strip().upper()

        try:
            if 'K' in count_str:
                return int(float(count_str.replace('K', '')) * 1000)
            elif 'M' in count_str:
                return int(float(count_str.replace('M', '')) * 1000000)
            elif count_str.isdigit():
                return int(count_str)
            else:
                # Try to extract just digits
                digits = ''.join(filter(str.isdigit, count_str))
                return int(digits) if digits else 0
        except:
            return 0

    def scrape(
        self,
        max_bookmarks: Optional[int] = None,
        use_twillot_export: bool = True,
        save_to_file: bool = True
    ) -> List[Dict]:
        """
        Main method to scrape bookmarks

        Args:
            max_bookmarks: Maximum number of bookmarks to fetch (None = all)
            use_twillot_export: Try to use Twillot's export feature
            save_to_file: Save results to JSON file

        Returns:
            List of bookmark dictionaries
        """
        if not self._ensure_extension():
            console.print("[yellow]Extension not available. Using direct scraping mode.[/yellow]")
            use_twillot_export = False

        bookmarks = []

        with sync_playwright() as playwright:
            console.print("[bold cyan]Launching browser with Twillot extension...[/bold cyan]")

            try:
                self.context = self._launch_browser(playwright)
                self.page = self.context.pages[0] if self.context.pages else self.context.new_page()

                # Login if needed
                if not self._wait_for_login(self.page):
                    return []

                # Navigate to bookmarks
                if not self._navigate_to_bookmarks(self.page):
                    console.print("[red]Could not load bookmarks page[/red]")
                    return []

                # Scroll to load all bookmarks
                self._trigger_twillot_sync(self.page)

                # Try Twillot export first
                if use_twillot_export:
                    popup = self._open_twillot_popup(self.page)
                    if popup:
                        export_path = self._export_via_twillot(popup)
                        if export_path and os.path.exists(export_path):
                            bookmarks = self._load_twillot_export(export_path)
                            popup.close()

                # Fallback to direct scraping
                if not bookmarks:
                    console.print("[yellow]Falling back to direct DOM scraping...[/yellow]")
                    bookmarks = self._scrape_bookmarks_directly(self.page)

                # Apply limit
                if max_bookmarks and len(bookmarks) > max_bookmarks:
                    bookmarks = bookmarks[:max_bookmarks]

                # Save to file
                if save_to_file and bookmarks:
                    output_file = self.export_dir / f"bookmarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(bookmarks, f, ensure_ascii=False, indent=2)
                    console.print(f"[green]Saved {len(bookmarks)} bookmarks to {output_file}[/green]")

            finally:
                if self.context:
                    # Keep browser open for debugging?
                    if Confirm.ask("Close browser?", default=True):
                        self.context.close()

        return bookmarks

    def _load_twillot_export(self, filepath: str) -> List[Dict]:
        """Load and normalize Twillot export file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Normalize to our format
            bookmarks = []
            items = data if isinstance(data, list) else data.get('bookmarks', data.get('tweets', []))

            for item in items:
                bookmark = self._normalize_twillot_item(item)
                if bookmark:
                    bookmarks.append(bookmark)

            return bookmarks

        except Exception as e:
            console.print(f"[red]Error loading Twillot export: {e}[/red]")
            return []

    def _normalize_twillot_item(self, item: Dict) -> Optional[Dict]:
        """Normalize a Twillot export item to our standard format"""
        # Map common field names
        field_mappings = {
            'id': ['id', 'tweet_id', 'tweetId'],
            'text': ['text', 'content', 'full_text', 'tweet_text'],
            'created_at': ['created_at', 'createdAt', 'date', 'timestamp'],
            'author_username': ['author_username', 'username', 'screen_name', 'user_screen_name'],
            'author_name': ['author_name', 'name', 'user_name', 'displayName'],
            'url': ['url', 'tweet_url', 'link'],
            'likes': ['likes', 'like_count', 'favorite_count', 'favourites'],
            'retweets': ['retweets', 'retweet_count', 'rt_count'],
            'replies': ['replies', 'reply_count'],
            'views': ['views', 'view_count', 'impressions'],
            'media_urls': ['media_urls', 'media', 'images', 'photos'],
        }

        normalized = {}

        for our_field, possible_names in field_mappings.items():
            for name in possible_names:
                if name in item and item[name] is not None:
                    normalized[our_field] = item[name]
                    break

        # Store raw data for reference
        normalized['raw_data'] = item

        return normalized if normalized.get('id') or normalized.get('text') else None


class TwillotImporter:
    """Import bookmarks from Twillot export files"""

    @staticmethod
    def import_json(filepath: str) -> List[Dict]:
        """Import from Twillot JSON export"""
        console.print(f"[cyan]Importing from {filepath}...[/cyan]")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different export formats
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get('bookmarks', data.get('tweets', data.get('data', [])))
        else:
            items = []

        bookmarks = []
        for item in items:
            bookmark = TwillotImporter._normalize_item(item)
            if bookmark:
                bookmarks.append(bookmark)

        console.print(f"[green]Imported {len(bookmarks)} bookmarks[/green]")
        return bookmarks

    @staticmethod
    def import_csv(filepath: str) -> List[Dict]:
        """Import from Twillot CSV export"""
        import csv

        console.print(f"[cyan]Importing from {filepath}...[/cyan]")

        bookmarks = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                bookmark = TwillotImporter._normalize_item(dict(row))
                if bookmark:
                    bookmarks.append(bookmark)

        console.print(f"[green]Imported {len(bookmarks)} bookmarks[/green]")
        return bookmarks

    @staticmethod
    def _normalize_item(item: Dict) -> Optional[Dict]:
        """Normalize item to standard bookmark format

        Handles real Twillot export format:
        - tweet_id: Primary ID
        - full_text: Main text content
        - media_items: URLs to media
        - has_image/has_video/has_gif: Media type flags
        - screen_name: Twitter handle (@username)
        - username: Display name (NOT the handle!)
        - favorite_count: Likes
        - views_count: View count (string format)
        """
        # Get tweet ID
        tweet_id = item.get('id') or item.get('tweet_id') or item.get('tweetId')

        # Get text content
        text = (item.get('text') or item.get('content') or
                item.get('full_text') or item.get('tweet'))

        # Handle author fields - Twillot uses screen_name for handle, username for display name
        # Priority for handle: screen_name > author_username > user
        author_username = (item.get('screen_name') or
                          item.get('author_username') or
                          item.get('user'))

        # Priority for display name: username (if screen_name exists) > name > author_name > display_name
        # Only use 'username' for display name if screen_name is present (Twillot format)
        if item.get('screen_name') and item.get('username'):
            author_name = item.get('username')  # In Twillot, 'username' is display name
        else:
            author_name = (item.get('name') or
                          item.get('author_name') or
                          item.get('display_name') or
                          item.get('username'))

        # Parse metrics - handle both int and string formats
        def parse_int(value):
            """Parse value to int, handling strings like '47143'"""
            if value is None:
                return 0
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                try:
                    return int(value.replace(',', ''))
                except ValueError:
                    return 0
            return 0

        like_count = parse_int(
            item.get('likes') or item.get('like_count') or
            item.get('favorite_count') or item.get('favorites') or 0
        )
        retweet_count = parse_int(
            item.get('retweets') or item.get('retweet_count') or 0
        )
        reply_count = parse_int(
            item.get('replies') or item.get('reply_count') or 0
        )
        view_count = parse_int(
            item.get('views') or item.get('view_count') or
            item.get('views_count') or item.get('impressions') or 0
        )

        normalized = {
            'id': tweet_id,
            'text': text,
            'created_at': item.get('created_at') or item.get('createdAt') or item.get('date'),
            'author_username': author_username,
            'author_name': author_name,
            'url': item.get('url') or item.get('tweet_url') or item.get('link'),
            'metrics': {
                'like_count': like_count,
                'retweet_count': retweet_count,
                'reply_count': reply_count,
                'view_count': view_count,
            },
            'raw_twillot_data': item,  # Keep original for reference
        }

        # Handle media - include media_items for Twillot format
        media = (item.get('media_items') or item.get('media') or
                 item.get('media_urls') or item.get('images') or [])
        if media:
            if isinstance(media, str):
                media = [media]
            normalized['media_urls'] = media

        # Extract media types from Twillot flags
        media_types = []
        has_image = item.get('has_image', False)
        has_video = item.get('has_video', False)
        has_gif = item.get('has_gif', False)

        if has_image:
            media_types.append('image')
        if has_video:
            media_types.append('video')
        if has_gif:
            media_types.append('gif')

        if media_types:
            normalized['media_types'] = media_types
        elif media:
            # Infer types from URLs if flags not present
            inferred_types = []
            for url in media:
                url_lower = url.lower() if isinstance(url, str) else ''
                if any(ext in url_lower for ext in ['.mp4', '.mov', '.webm', 'video']):
                    inferred_types.append('video')
                elif '.gif' in url_lower:
                    inferred_types.append('gif')
                else:
                    inferred_types.append('image')
            normalized['media_types'] = inferred_types

        # Handle URLs in tweet
        urls = item.get('urls') or item.get('expanded_urls') or item.get('links') or []
        if urls:
            normalized['expanded_urls'] = urls if isinstance(urls, list) else [urls]

        # Build tweet URL if not present
        if not normalized['url'] and normalized['author_username'] and normalized['id']:
            normalized['url'] = f"https://twitter.com/{normalized['author_username']}/status/{normalized['id']}"

        return normalized if normalized.get('id') or normalized.get('text') else None


def check_playwright_installed() -> bool:
    """Check if Playwright is installed and browsers are available"""
    if not PLAYWRIGHT_AVAILABLE:
        console.print("[red]Playwright is not installed![/red]")
        console.print("Run: [cyan]pip install playwright && playwright install chromium[/cyan]")
        return False
    return True


# CLI interface when run directly
if __name__ == "__main__":
    console.print("[bold]Twillot Scraper - Twitter Bookmarks Automation[/bold]\n")

    if not check_playwright_installed():
        exit(1)

    action = Prompt.ask(
        "What would you like to do?",
        choices=["scrape", "import", "setup"],
        default="scrape"
    )

    if action == "setup":
        manager = TwillotExtensionManager()
        manager.setup_extension_manually()

    elif action == "import":
        filepath = Prompt.ask("Path to Twillot export file")
        if filepath.endswith('.csv'):
            bookmarks = TwillotImporter.import_csv(filepath)
        else:
            bookmarks = TwillotImporter.import_json(filepath)

        # Show sample
        if bookmarks:
            table = Table(title="Sample Bookmarks")
            table.add_column("Author")
            table.add_column("Text", max_width=50)
            table.add_column("Likes")

            for b in bookmarks[:5]:
                table.add_row(
                    f"@{b.get('author_username', 'unknown')}",
                    (b.get('text', '')[:47] + '...') if len(b.get('text', '')) > 50 else b.get('text', ''),
                    str(b.get('metrics', {}).get('like_count', 0))
                )

            console.print(table)

    elif action == "scrape":
        scraper = TwillotScraper(headless=False)
        bookmarks = scraper.scrape()

        console.print(f"\n[bold green]Done! Scraped {len(bookmarks)} bookmarks.[/bold green]")
