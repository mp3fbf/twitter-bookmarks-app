"""
Content Fetcher - Extracts full content from URLs in tweets

This module goes beyond metadata extraction to fetch the actual content
of linked articles, guides, lists, etc. It handles:
- URL expansion (t.co, bit.ly, etc.)
- Full page content extraction
- Paywall bypass attempts
- Content caching
- Special handling for known sites (GitHub, YouTube, etc.)
"""

import re
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
from rich.console import Console

console = Console()


@dataclass
class FetchedContent:
    """Content fetched from a URL"""
    url: str
    expanded_url: str
    title: Optional[str] = None
    description: Optional[str] = None
    main_content: Optional[str] = None  # The actual article/page content
    content_type: str = "unknown"  # article, github, youtube, image, list, guide, etc.
    site_name: Optional[str] = None
    author: Optional[str] = None
    published_date: Optional[str] = None
    lists_extracted: List[str] = field(default_factory=list)  # For top 10, guides, etc.
    code_blocks: List[str] = field(default_factory=list)  # For code snippets
    extra_data: Dict = field(default_factory=dict)
    fetch_error: Optional[str] = None
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())
    paywall_detected: bool = False
    cached: bool = False


class ContentFetcher:
    """Fetches and extracts full content from URLs"""

    CACHE_DIR = Path("~/.cache/twitter-bookmarks/content").expanduser()
    CACHE_DURATION_HOURS = 24 * 7  # 1 week cache

    # User agents to rotate
    USER_AGENTS = [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ]

    # Sites known to have paywalls
    PAYWALL_SITES = {
        'nytimes.com', 'wsj.com', 'washingtonpost.com', 'ft.com',
        'economist.com', 'bloomberg.com', 'oglobo.globo.com', 'globo.com',
        'folha.uol.com.br', 'estadao.com.br', 'medium.com'
    }

    def __init__(self, timeout: int = 15, delay: float = 0.5, use_cache: bool = True):
        self.timeout = timeout
        self.delay = delay
        self.use_cache = use_cache
        self.session = requests.Session()
        self._ua_index = 0

        # Create cache directory
        if use_cache:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _get_user_agent(self) -> str:
        """Rotate user agents"""
        ua = self.USER_AGENTS[self._ua_index % len(self.USER_AGENTS)]
        self._ua_index += 1
        return ua

    def _get_cache_path(self, url: str) -> Path:
        """Get cache file path for a URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.CACHE_DIR / f"{url_hash}.json"

    def _load_from_cache(self, url: str) -> Optional[FetchedContent]:
        """Load content from cache if valid"""
        if not self.use_cache:
            return None

        cache_path = self._get_cache_path(url)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check cache age
            fetched_at = datetime.fromisoformat(data['fetched_at'])
            if datetime.now() - fetched_at > timedelta(hours=self.CACHE_DURATION_HOURS):
                return None

            content = FetchedContent(**data)
            content.cached = True
            return content
        except Exception:
            return None

    def _save_to_cache(self, content: FetchedContent):
        """Save content to cache"""
        if not self.use_cache:
            return

        cache_path = self._get_cache_path(content.url)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'url': content.url,
                    'expanded_url': content.expanded_url,
                    'title': content.title,
                    'description': content.description,
                    'main_content': content.main_content,
                    'content_type': content.content_type,
                    'site_name': content.site_name,
                    'author': content.author,
                    'published_date': content.published_date,
                    'lists_extracted': content.lists_extracted,
                    'code_blocks': content.code_blocks,
                    'extra_data': content.extra_data,
                    'fetch_error': content.fetch_error,
                    'fetched_at': content.fetched_at,
                    'paywall_detected': content.paywall_detected,
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            console.print(f"[yellow]Cache write error: {e}[/yellow]")

    def extract_urls(self, text: str) -> List[str]:
        """Extract all URLs from text"""
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:[/](?:[-\w._~!$&\'()*+,;=:@]|%[\da-fA-F]{2})*)*(?:\?(?:[-\w._~!$&\'()*+,;=:@/?]|%[\da-fA-F]{2})*)?(?:#(?:[-\w._~!$&\'()*+,;=:@/?]|%[\da-fA-F]{2})*)?'
        urls = re.findall(url_pattern, text)

        # Filter out Twitter media URLs
        filtered = []
        for url in urls:
            if 'twitter.com' in url and ('/photo/' in url or '/video/' in url):
                continue
            if 'pbs.twimg.com' in url or 'video.twimg.com' in url:
                continue
            filtered.append(url)

        return filtered

    def expand_url(self, short_url: str) -> str:
        """Expand a shortened URL"""
        try:
            if 'twitter.com' in short_url or 'x.com' in short_url:
                return short_url

            headers = {'User-Agent': self._get_user_agent()}
            response = self.session.head(
                short_url,
                allow_redirects=True,
                timeout=self.timeout,
                headers=headers
            )
            return response.url
        except Exception as e:
            console.print(f"[yellow]URL expansion failed for {short_url}: {e}[/yellow]")
            return short_url

    def _is_paywall_site(self, url: str) -> bool:
        """Check if URL is from a known paywall site"""
        domain = urlparse(url).netloc.lower()
        return any(pw in domain for pw in self.PAYWALL_SITES)

    def _try_archive_bypass(self, url: str) -> Optional[str]:
        """Try to get content via archive.org or similar"""
        try:
            # Try archive.org Wayback Machine
            archive_url = f"https://web.archive.org/web/2/{url}"
            headers = {'User-Agent': self._get_user_agent()}
            response = self.session.get(archive_url, timeout=self.timeout, headers=headers)
            if response.status_code == 200:
                return response.text
        except Exception:
            pass

        try:
            # Try 12ft.io (paywall remover)
            bypass_url = f"https://12ft.io/{url}"
            headers = {'User-Agent': self._get_user_agent()}
            response = self.session.get(bypass_url, timeout=self.timeout, headers=headers)
            if response.status_code == 200:
                return response.text
        except Exception:
            pass

        return None

    def _extract_article_content(self, soup: BeautifulSoup, url: str) -> Tuple[str, List[str], List[str]]:
        """Extract main article content, lists, and code blocks"""
        main_content = ""
        lists_extracted = []
        code_blocks = []

        # Try to find article content using common selectors
        content_selectors = [
            'article',
            '[role="main"]',
            '.post-content',
            '.article-content',
            '.entry-content',
            '.content',
            'main',
            '#content',
            '.markdown-body',  # GitHub
            '.post',
        ]

        content_element = None
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                break

        if not content_element:
            content_element = soup.body

        if content_element:
            # Remove unwanted elements
            for unwanted in content_element.select('nav, footer, aside, .sidebar, .comments, .advertisement, script, style'):
                unwanted.decompose()

            # Extract text content
            main_content = content_element.get_text(separator='\n', strip=True)

            # Extract lists (ol, ul)
            for list_elem in content_element.select('ol, ul'):
                items = [li.get_text(strip=True) for li in list_elem.select('li')]
                if items and len(items) > 2:  # Only meaningful lists
                    lists_extracted.append('\n'.join(f"- {item}" for item in items))

            # Extract code blocks
            for code in content_element.select('pre code, pre, code'):
                code_text = code.get_text(strip=True)
                if code_text and len(code_text) > 20:  # Only meaningful code
                    code_blocks.append(code_text)

        # Limit content length
        if len(main_content) > 15000:
            main_content = main_content[:15000] + "...[truncated]"

        return main_content, lists_extracted, code_blocks

    def _extract_github_content(self, url: str) -> FetchedContent:
        """Special handling for GitHub URLs"""
        content = FetchedContent(url=url, expanded_url=url, content_type='github')

        parsed = urlparse(url)
        parts = parsed.path.strip('/').split('/')

        if len(parts) >= 2:
            owner, repo = parts[0], parts[1]
            content.extra_data['owner'] = owner
            content.extra_data['repo'] = repo

            # Get repo info from API
            try:
                api_url = f"https://api.github.com/repos/{owner}/{repo}"
                response = self.session.get(api_url, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    content.title = f"{owner}/{repo}"
                    content.description = data.get('description', '')
                    content.extra_data.update({
                        'stars': data.get('stargazers_count', 0),
                        'forks': data.get('forks_count', 0),
                        'language': data.get('language', ''),
                        'topics': data.get('topics', []),
                        'homepage': data.get('homepage', ''),
                    })

                # Try to get README
                readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
                response = self.session.get(readme_url, timeout=self.timeout)
                if response.status_code == 200:
                    import base64
                    readme_data = response.json()
                    readme_content = base64.b64decode(readme_data.get('content', '')).decode('utf-8')
                    content.main_content = readme_content[:10000]  # Limit README size

            except Exception as e:
                content.fetch_error = str(e)

        return content

    def _extract_youtube_content(self, url: str) -> FetchedContent:
        """Special handling for YouTube URLs"""
        content = FetchedContent(url=url, expanded_url=url, content_type='youtube')

        # Extract video ID
        video_id = None
        if 'youtube.com/watch' in url:
            match = re.search(r'v=([a-zA-Z0-9_-]+)', url)
            if match:
                video_id = match.group(1)
        elif 'youtu.be/' in url:
            match = re.search(r'youtu\.be/([a-zA-Z0-9_-]+)', url)
            if match:
                video_id = match.group(1)

        if video_id:
            content.extra_data['video_id'] = video_id

            # Get video info from oEmbed
            try:
                oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
                response = self.session.get(oembed_url, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    content.title = data.get('title', '')
                    content.author = data.get('author_name', '')
            except Exception as e:
                content.fetch_error = str(e)

        return content

    def fetch_content(self, url: str) -> FetchedContent:
        """Fetch and extract content from a URL"""
        # Check cache first
        cached = self._load_from_cache(url)
        if cached:
            return cached

        # Expand URL if shortened
        expanded_url = url
        if 't.co' in url or 'bit.ly' in url or 'tinyurl' in url or 'goo.gl' in url:
            expanded_url = self.expand_url(url)

        content = FetchedContent(url=url, expanded_url=expanded_url)
        domain = urlparse(expanded_url).netloc.lower()

        # Handle special sites
        if 'github.com' in domain:
            content = self._extract_github_content(expanded_url)
            self._save_to_cache(content)
            return content

        if 'youtube.com' in domain or 'youtu.be' in domain:
            content = self._extract_youtube_content(expanded_url)
            self._save_to_cache(content)
            return content

        # Skip Twitter/X URLs (we already have the tweet)
        if 'twitter.com' in domain or 'x.com' in domain:
            content.content_type = 'twitter'
            content.main_content = "Twitter/X link - content already available in tweet"
            return content

        # Fetch the page
        try:
            headers = {'User-Agent': self._get_user_agent()}
            response = self.session.get(expanded_url, timeout=self.timeout, headers=headers)
            response.raise_for_status()

            html = response.text
            soup = BeautifulSoup(html, 'html.parser')

            # Extract metadata
            og_title = soup.find('meta', property='og:title')
            if og_title:
                content.title = og_title.get('content', '')

            og_desc = soup.find('meta', property='og:description')
            if og_desc:
                content.description = og_desc.get('content', '')

            og_site = soup.find('meta', property='og:site_name')
            if og_site:
                content.site_name = og_site.get('content', '')

            # Fallback title
            if not content.title:
                title_tag = soup.find('title')
                if title_tag:
                    content.title = title_tag.text.strip()

            # Check for paywall indicators
            paywall_indicators = ['paywall', 'subscribe', 'subscription required', 'premium content']
            page_text = soup.get_text().lower()
            if any(ind in page_text for ind in paywall_indicators) and self._is_paywall_site(expanded_url):
                content.paywall_detected = True

                # Try bypass
                bypass_html = self._try_archive_bypass(expanded_url)
                if bypass_html:
                    soup = BeautifulSoup(bypass_html, 'html.parser')
                    content.extra_data['bypass_used'] = True

            # Extract main content
            main_content, lists, code_blocks = self._extract_article_content(soup, expanded_url)
            content.main_content = main_content
            content.lists_extracted = lists
            content.code_blocks = code_blocks

            # Determine content type
            if lists:
                content.content_type = 'list/guide'
            elif code_blocks:
                content.content_type = 'code/tutorial'
            else:
                content.content_type = 'article'

        except requests.exceptions.Timeout:
            content.fetch_error = "Timeout fetching URL"
        except requests.exceptions.HTTPError as e:
            content.fetch_error = f"HTTP error: {e.response.status_code}"
        except Exception as e:
            content.fetch_error = str(e)

        # Add delay
        time.sleep(self.delay)

        # Save to cache
        self._save_to_cache(content)

        return content

    def fetch_urls_from_tweet(self, tweet_text: str) -> List[FetchedContent]:
        """Extract and fetch all URLs from a tweet"""
        urls = self.extract_urls(tweet_text)
        results = []

        for url in urls:
            content = self.fetch_content(url)
            results.append(content)

        return results

    def get_content_summary(self, content: FetchedContent) -> str:
        """Get a formatted summary of fetched content"""
        parts = []

        if content.title:
            parts.append(f"**Title:** {content.title}")

        if content.site_name:
            parts.append(f"**Source:** {content.site_name}")

        if content.author:
            parts.append(f"**Author:** {content.author}")

        if content.content_type:
            parts.append(f"**Type:** {content.content_type}")

        if content.description:
            parts.append(f"\n**Description:** {content.description}")

        if content.main_content:
            # Truncate for summary
            preview = content.main_content[:2000]
            if len(content.main_content) > 2000:
                preview += "..."
            parts.append(f"\n**Content:**\n{preview}")

        if content.lists_extracted:
            parts.append(f"\n**Lists found:** {len(content.lists_extracted)}")
            for i, lst in enumerate(content.lists_extracted[:2], 1):
                parts.append(f"\nList {i}:\n{lst[:500]}")

        if content.code_blocks:
            parts.append(f"\n**Code blocks found:** {len(content.code_blocks)}")

        if content.extra_data:
            if 'stars' in content.extra_data:
                parts.append(f"\n**GitHub Stars:** {content.extra_data['stars']}")
            if 'topics' in content.extra_data:
                parts.append(f"**Topics:** {', '.join(content.extra_data['topics'][:5])}")

        if content.paywall_detected:
            parts.append("\n[Paywall detected - content may be incomplete]")

        if content.fetch_error:
            parts.append(f"\n[Error: {content.fetch_error}]")

        return '\n'.join(parts)


# Example usage
if __name__ == "__main__":
    fetcher = ContentFetcher()

    # Test with a sample tweet text
    test_tweet = """
    Check out this amazing guide: https://t.co/abc123
    And this repo: https://github.com/anthropics/anthropic-quickstarts
    """

    print("Extracting URLs...")
    urls = fetcher.extract_urls(test_tweet)
    print(f"Found {len(urls)} URLs")

    for url in urls:
        print(f"\nFetching: {url}")
        content = fetcher.fetch_content(url)
        print(fetcher.get_content_summary(content))
