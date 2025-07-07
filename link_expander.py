"""
Link Expander Module
Expands shortened URLs and extracts metadata from web pages
"""

import re
import time
import requests
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dataclasses import dataclass
from rich.console import Console

console = Console()


@dataclass
class LinkMetadata:
    """Metadata extracted from a URL"""
    url: str
    expanded_url: str
    title: Optional[str] = None
    description: Optional[str] = None
    image: Optional[str] = None
    site_name: Optional[str] = None
    content_type: Optional[str] = None
    extra_data: Optional[Dict] = None


class LinkExpander:
    """Expands shortened URLs and extracts metadata"""
    
    def __init__(self, timeout: int = 10, delay: float = 0.5):
        self.timeout = timeout
        self.delay = delay  # Delay between requests to avoid rate limiting
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
    def extract_urls(self, text: str) -> List[str]:
        """Extract all URLs from text"""
        # Pattern to match URLs (including t.co)
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:[/](?:[-\w._~!$&\'()*+,;=:@]|%[\da-fA-F]{2})*)*(?:\?(?:[-\w._~!$&\'()*+,;=:@/?]|%[\da-fA-F]{2})*)?(?:#(?:[-\w._~!$&\'()*+,;=:@/?]|%[\da-fA-F]{2})*)?'
        urls = re.findall(url_pattern, text)
        
        # Filter out Twitter media URLs (photos/videos) as they require authentication
        filtered_urls = []
        for url in urls:
            if 'twitter.com' in url and ('/photo/' in url or '/video/' in url):
                continue  # Skip Twitter media URLs
            filtered_urls.append(url)
        
        return filtered_urls
    
    def expand_url(self, short_url: str) -> str:
        """Expand a shortened URL to its final destination"""
        try:
            # Skip Twitter status URLs without media
            if 'twitter.com' in short_url and '/status/' in short_url and not ('/photo/' in short_url or '/video/' in short_url):
                return short_url  # Don't try to expand Twitter status URLs
                
            response = self.session.head(short_url, allow_redirects=True, timeout=self.timeout)
            return response.url
        except requests.exceptions.Timeout:
            console.print(f"[yellow]Timeout expanding URL: {short_url}[/yellow]")
            return short_url
        except requests.exceptions.ConnectionError:
            console.print(f"[yellow]Connection error for URL: {short_url}[/yellow]")
            return short_url
        except Exception as e:
            console.print(f"[yellow]Error expanding URL {short_url}: {str(e)}[/yellow]")
            return short_url
    
    def extract_metadata(self, url: str) -> LinkMetadata:
        """Extract metadata from a URL"""
        expanded_url = url
        
        # First, expand the URL if it's shortened
        if 't.co' in url or 'bit.ly' in url or 'tinyurl' in url:
            expanded_url = self.expand_url(url)
        
        metadata = LinkMetadata(url=url, expanded_url=expanded_url)
        
        # Determine content type from URL
        parsed = urlparse(expanded_url)
        domain = parsed.netloc.lower()
        
        # Special handling for known sites
        if 'github.com' in domain:
            metadata.content_type = 'github'
            metadata.extra_data = self._extract_github_data(expanded_url)
        elif 'youtube.com' in domain or 'youtu.be' in domain:
            metadata.content_type = 'youtube'
            metadata.extra_data = self._extract_youtube_data(expanded_url)
        elif any(img_ext in expanded_url.lower() for img_ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
            metadata.content_type = 'image'
            metadata.title = "Image"
            metadata.description = f"Direct image link: {parsed.path.split('/')[-1]}"
        else:
            # Generic webpage - extract Open Graph data
            metadata.content_type = 'article'
            self._extract_webpage_metadata(expanded_url, metadata)
        
        # Add delay to avoid rate limiting
        time.sleep(self.delay)
        
        return metadata
    
    def _extract_webpage_metadata(self, url: str, metadata: LinkMetadata) -> None:
        """Extract Open Graph and meta tags from a webpage"""
        try:
            # Skip problematic URLs
            if 'twitter.com' in url or 'x.com' in url:
                metadata.title = "Twitter/X Link"
                metadata.description = "Link to Twitter/X content"
                return
                
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract Open Graph tags
            og_title = soup.find('meta', property='og:title')
            if og_title:
                metadata.title = og_title.get('content', '')
            
            og_description = soup.find('meta', property='og:description')
            if og_description:
                metadata.description = og_description.get('content', '')
            
            og_image = soup.find('meta', property='og:image')
            if og_image:
                metadata.image = og_image.get('content', '')
            
            og_site = soup.find('meta', property='og:site_name')
            if og_site:
                metadata.site_name = og_site.get('content', '')
            
            # Fallback to regular meta tags
            if not metadata.title:
                title_tag = soup.find('title')
                if title_tag:
                    metadata.title = title_tag.text.strip()
            
            if not metadata.description:
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc:
                    metadata.description = meta_desc.get('content', '')
                    
        except requests.exceptions.Timeout:
            console.print(f"[yellow]Timeout fetching metadata for: {url}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Error fetching metadata for {url}: {str(e)}[/yellow]")
    
    def _extract_github_data(self, url: str) -> Dict:
        """Extract GitHub-specific data"""
        # Parse GitHub URL to get owner/repo
        parts = urlparse(url).path.strip('/').split('/')
        if len(parts) >= 2:
            owner, repo = parts[0], parts[1]
            
            # Try to get repo data from GitHub API (without auth for now)
            try:
                api_url = f"https://api.github.com/repos/{owner}/{repo}"
                response = self.session.get(api_url, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'owner': owner,
                        'repo': repo,
                        'stars': data.get('stargazers_count', 0),
                        'description': data.get('description', ''),
                        'language': data.get('language', ''),
                        'topics': data.get('topics', [])
                    }
            except:
                pass
        
        return {'owner': parts[0] if parts else '', 'repo': parts[1] if len(parts) > 1 else ''}
    
    def _extract_youtube_data(self, url: str) -> Dict:
        """Extract YouTube video ID"""
        video_id = None
        
        # Extract video ID from various YouTube URL formats
        if 'youtube.com/watch' in url:
            match = re.search(r'v=([a-zA-Z0-9_-]+)', url)
            if match:
                video_id = match.group(1)
        elif 'youtu.be/' in url:
            match = re.search(r'youtu\.be/([a-zA-Z0-9_-]+)', url)
            if match:
                video_id = match.group(1)
        
        return {'video_id': video_id}
    
    def process_bookmark_urls(self, bookmark: Dict) -> Dict:
        """Process all URLs in a bookmark and add metadata"""
        text = bookmark.get('text', '')
        urls = self.extract_urls(text)
        
        if urls:
            bookmark['expanded_urls'] = []
            bookmark['url_metadata'] = []
            
            for url in urls:
                try:
                    metadata = self.extract_metadata(url)
                    bookmark['expanded_urls'].append(metadata.expanded_url)
                    bookmark['url_metadata'].append({
                        'original_url': metadata.url,
                        'expanded_url': metadata.expanded_url,
                        'title': metadata.title,
                        'description': metadata.description,
                        'content_type': metadata.content_type,
                        'extra_data': metadata.extra_data
                    })
                except Exception as e:
                    console.print(f"[red]Error processing URL {url}: {str(e)}[/red]")
                    bookmark['expanded_urls'].append(url)
                    bookmark['url_metadata'].append({
                        'original_url': url,
                        'expanded_url': url,
                        'error': str(e)
                    })
        
        return bookmark
    
    def process_bookmarks_batch(self, bookmarks: List[Dict], limit: Optional[int] = None) -> List[Dict]:
        """Process a batch of bookmarks to expand their URLs"""
        processed = []
        bookmarks_to_process = bookmarks[:limit] if limit else bookmarks
        
        console.print(f"[cyan]Processing URLs in {len(bookmarks_to_process)} bookmarks...[/cyan]")
        
        for i, bookmark in enumerate(bookmarks_to_process, 1):
            console.print(f"[dim]Processing bookmark {i}/{len(bookmarks_to_process)}...[/dim]")
            processed_bookmark = self.process_bookmark_urls(bookmark.copy())
            processed.append(processed_bookmark)
        
        return processed