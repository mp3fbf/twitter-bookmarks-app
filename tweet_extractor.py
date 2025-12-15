"""
Tweet Extractor - Extracts complete content from tweets including quote tweets and threads

This module handles the extraction of all tweet content from Twillot JSON exports,
including quoted tweets, threads, their links, and media URLs.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ThreadInfo:
    """Information about a thread"""
    is_thread_start: bool  # É o primeiro tweet de uma thread?
    has_full_thread: bool  # Twillot capturou a thread completa?
    tweets: List[Dict] = field(default_factory=list)  # Tweets da thread (se capturados)
    total_tweets: int = 0
    needs_fetch: bool = False  # Precisa buscar via ThreadReaderApp?


@dataclass
class QuotedTweetInfo:
    """Information about a quoted tweet"""
    text: str
    author: str
    tweet_id: str
    urls: List[str]
    image_urls: List[str]
    video_urls: List[str]
    video_thumbnails: List[str]


def get_best_video_url(media_item: Dict) -> Optional[str]:
    """
    Extract the best quality video URL from a media item

    Args:
        media_item: Media dict from Twitter API with video_info

    Returns:
        URL of highest quality MP4, or None
    """
    video_info = media_item.get('video_info', {})
    variants = video_info.get('variants', [])

    # Filter to MP4 only
    mp4_variants = [v for v in variants if v.get('content_type') == 'video/mp4']

    if not mp4_variants:
        return None

    # Get highest bitrate
    best = max(mp4_variants, key=lambda x: x.get('bitrate', 0))
    return best.get('url')


def get_quoted_tweet_info(bookmark: Dict) -> Optional[QuotedTweetInfo]:
    """
    Extract information from a quoted tweet if present

    Args:
        bookmark: Bookmark dict from Twillot export

    Returns:
        QuotedTweetInfo if this is a quote tweet, None otherwise
    """
    # Navigate to quoted tweet in nested structure
    quoted = (bookmark.get('_data', {})
              .get('tweet_results', {})
              .get('result', {})
              .get('quoted_status_result', {})
              .get('result', {}))

    if not quoted:
        return None

    legacy = quoted.get('legacy', {})

    # Get author info
    author = (quoted.get('core', {})
              .get('user_results', {})
              .get('result', {})
              .get('legacy', {})
              .get('screen_name', 'unknown'))

    # Get tweet ID
    tweet_id = quoted.get('rest_id', '')

    # Get text
    text = legacy.get('full_text', '')

    # Extract URLs
    urls = [
        u.get('expanded_url')
        for u in legacy.get('entities', {}).get('urls', [])
        if u.get('expanded_url')
    ]

    # Extract media
    media_items = legacy.get('extended_entities', {}).get('media', [])

    image_urls = [
        m.get('media_url_https')
        for m in media_items
        if m.get('type') == 'photo' and m.get('media_url_https')
    ]

    video_urls = []
    video_thumbnails = []
    for m in media_items:
        if m.get('type') in ('video', 'animated_gif'):
            url = get_best_video_url(m)
            if url:
                video_urls.append(url)
            thumb = m.get('media_url_https')
            if thumb:
                video_thumbnails.append(thumb)

    return QuotedTweetInfo(
        text=text,
        author=author,
        tweet_id=tweet_id,
        urls=urls,
        image_urls=image_urls,
        video_urls=video_urls,
        video_thumbnails=video_thumbnails
    )


def is_quote_tweet(bookmark: Dict) -> bool:
    """Check if a bookmark is a quote tweet"""
    return get_quoted_tweet_info(bookmark) is not None


def get_tweet_url_from_quoted(quoted: QuotedTweetInfo) -> str:
    """Build the URL for a quoted tweet"""
    if quoted.author and quoted.tweet_id:
        return f"https://x.com/{quoted.author}/status/{quoted.tweet_id}"
    return ""


def format_quoted_tweet_for_prompt(quoted: QuotedTweetInfo) -> str:
    """
    Format quoted tweet info for inclusion in LLM prompt

    Args:
        quoted: QuotedTweetInfo object

    Returns:
        Formatted string to include in prompt
    """
    lines = [
        "",
        "--- QUOTE TWEET (tweet citado) ---",
        f"Autor: @{quoted.author}",
        f"Texto: {quoted.text}",
    ]

    if quoted.urls:
        lines.append(f"Links: {', '.join(quoted.urls)}")

    if quoted.image_urls:
        lines.append(f"Imagens: {len(quoted.image_urls)} imagem(ns) anexada(s)")
        for i, url in enumerate(quoted.image_urls, 1):
            lines.append(f"  Imagem {i}: {url}")

    if quoted.video_urls:
        lines.append(f"Videos: {len(quoted.video_urls)} video(s) anexado(s)")
        for i, thumb in enumerate(quoted.video_thumbnails, 1):
            lines.append(f"  Thumbnail {i}: {thumb}")

    lines.append("--- FIM DO QUOTE TWEET ---")
    lines.append("")

    return "\n".join(lines)


def extract_all_urls_from_bookmark(bookmark: Dict) -> List[str]:
    """
    Extract all URLs from a bookmark, including from quoted tweets

    Args:
        bookmark: Bookmark dict

    Returns:
        List of all URLs found
    """
    urls = []

    # URLs from main tweet text
    text = bookmark.get('full_text', bookmark.get('text', ''))

    # Get from entities
    for u in bookmark.get('entities', {}).get('urls', []):
        if u.get('expanded_url'):
            urls.append(u.get('expanded_url'))

    # Get from quoted tweet
    quoted = get_quoted_tweet_info(bookmark)
    if quoted:
        urls.extend(quoted.urls)

    return list(set(urls))  # Remove duplicates


def get_thread_info(bookmark: Dict) -> Optional[ThreadInfo]:
    """
    Extract thread information from a bookmark

    Detects if the bookmark is a thread start and whether Twillot captured
    the full thread or just the first tweet.

    Args:
        bookmark: Bookmark dict from Twillot export

    Returns:
        ThreadInfo if it's a thread, None otherwise
    """
    conversation_id = str(bookmark.get('conversation_id', ''))
    tweet_id = str(bookmark.get('tweet_id', ''))
    is_reply = bookmark.get('is_reply', False)
    post_type = bookmark.get('post_type', 'post')

    # É início de thread?
    # Um tweet é início de thread se:
    # 1. conversation_id == tweet_id (é o primeiro da conversa)
    # 2. Não é uma resposta a outro usuário
    is_thread_start = (conversation_id == tweet_id) and not is_reply

    # Twillot capturou a thread completa?
    # Se post_type == 'thread' e conversations[] está populado
    conversations = bookmark.get('conversations', [])
    has_full_thread = post_type == 'thread' and len(conversations) > 0

    # Se não é início de thread e não tem thread completa, retorna None
    if not is_thread_start and not has_full_thread:
        return None

    # Extrair tweets da thread se disponível
    tweets = []
    if has_full_thread:
        for conv in conversations:
            tweets.append({
                'text': conv.get('full_text', ''),
                'author': conv.get('screen_name', ''),
                'tweet_id': conv.get('tweet_id', ''),
                'is_reply': conv.get('is_reply', False),
            })

    # Precisa buscar via ThreadReaderApp?
    needs_fetch = is_thread_start and not has_full_thread

    return ThreadInfo(
        is_thread_start=is_thread_start,
        has_full_thread=has_full_thread,
        tweets=tweets,
        total_tweets=len(tweets) if tweets else 0,
        needs_fetch=needs_fetch
    )


def is_thread_start(bookmark: Dict) -> bool:
    """Check if a bookmark is the start of a thread"""
    info = get_thread_info(bookmark)
    return info is not None and info.is_thread_start


def format_thread_for_prompt(thread: ThreadInfo, bookmark: Dict) -> str:
    """
    Format thread info for inclusion in LLM prompt

    Args:
        thread: ThreadInfo object
        bookmark: Original bookmark dict

    Returns:
        Formatted string to include in prompt
    """
    if thread.has_full_thread and thread.tweets:
        # Thread completa capturada pelo Twillot
        lines = ["\n═══ THREAD COMPLETA ═══"]
        for i, tweet in enumerate(thread.tweets, 1):
            lines.append(f"\n[{i}/{len(thread.tweets)}] @{tweet['author']}:")
            lines.append(tweet['text'])
        lines.append("\n═══ FIM DA THREAD ═══")
        return "\n".join(lines)
    elif thread.needs_fetch:
        # Thread precisa ser buscada externamente
        author = bookmark.get('screen_name', 'unknown')
        tweet_id = bookmark.get('tweet_id', '')
        return f"""
⚠️ AVISO: Este é o PRIMEIRO TWEET de uma THREAD.
Os tweets seguintes NÃO foram capturados pelo Twillot.

Se o tweet menciona uma lista, guia, ou tutorial - o conteúdo completo
provavelmente está nos tweets seguintes desta thread.

URL da thread: https://x.com/{author}/status/{tweet_id}
"""
    return ""


# Example usage
if __name__ == "__main__":
    import json
    from pathlib import Path

    # Load bookmarks
    with open(Path("~/Downloads/twillot-bookmark.json").expanduser()) as f:
        bookmarks = json.load(f)

    # Find quote tweets
    quote_count = 0
    for b in bookmarks:
        quoted = get_quoted_tweet_info(b)
        if quoted:
            quote_count += 1
            if quote_count <= 3:  # Show first 3
                print(f"\n=== Quote Tweet #{quote_count} ===")
                print(f"Main: @{b.get('screen_name')}: {b.get('full_text', '')[:50]}...")
                print(format_quoted_tweet_for_prompt(quoted))

    print(f"\nTotal quote tweets: {quote_count}")
