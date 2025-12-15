"""
Vision Model Comparison Test

Tests all LLM providers with different media types:
- GPT-5.2 (OpenAI)
- Claude Opus 4.5 (Anthropic)
- Gemini 2.5 Flash (Google)
- Gemini 3 Pro Preview (Google) - best for video

Test categories:
- 5 text-only bookmarks
- 5 image bookmarks
- 5 video bookmarks (shortest duration)
- 2 GIF bookmarks (all available)
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from media_linker import MediaLinker

# Load environment variables
load_dotenv()
console = Console()

# Default media folder path
MEDIA_FOLDER = "~/Downloads/twillot-media-files-by-date/"


def get_video_duration_ms(bookmark: Dict) -> int:
    """Extract video duration in milliseconds from nested _data structure"""
    try:
        media_list = (bookmark.get('_data', {})
                     .get('tweet_results', {})
                     .get('result', {})
                     .get('legacy', {})
                     .get('extended_entities', {})
                     .get('media', []))
        if media_list:
            return media_list[0].get('video_info', {}).get('duration_millis', 0)
    except (KeyError, IndexError, TypeError):
        pass
    return 0


def format_duration(ms: int) -> str:
    """Format milliseconds as MM:SS"""
    if ms <= 0:
        return "N/A"
    seconds = ms // 1000
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}:{secs:02d}"


def load_bookmarks(filepath: str) -> List[Dict]:
    """Load bookmarks from JSON file"""
    with open(Path(filepath).expanduser(), 'r', encoding='utf-8') as f:
        return json.load(f)


def select_test_samples(bookmarks: List[Dict]) -> Dict[str, List[Dict]]:
    """Select test samples by media type"""
    text_only = []
    with_image = []
    with_video = []
    with_gif = []

    for b in bookmarks:
        has_image = b.get('has_image', False)
        has_video = b.get('has_video', False)
        has_gif = b.get('has_gif', False)

        if has_gif:
            with_gif.append(b)
        elif has_video:
            # Add duration info
            duration = get_video_duration_ms(b)
            b['_video_duration_ms'] = duration
            with_video.append(b)
        elif has_image:
            with_image.append(b)
        else:
            text_only.append(b)

    # Sort videos by duration (shortest first)
    with_video.sort(key=lambda x: x.get('_video_duration_ms', float('inf')))

    return {
        'text_only': text_only[:5],
        'with_image': with_image[:5],
        'with_video': with_video[:5],  # 5 shortest videos
        'with_gif': with_gif[:2]  # All GIFs (max 2)
    }


def test_text_analysis(provider_name: str, model: str, bookmarks: List[Dict]) -> List[Dict]:
    """Test text-only analysis"""
    from llm_providers import LLMFactory

    results = []

    try:
        if provider_name == 'gemini_pro':
            from llm_providers import GeminiProvider
            llm = GeminiProvider(model='gemini-3-pro-preview')
        else:
            llm = LLMFactory.create(provider_name, model=model)
    except Exception as e:
        console.print(f"[red]Failed to create {provider_name} provider: {e}[/red]")
        return [{'error': str(e), 'provider': provider_name}]

    for i, bookmark in enumerate(bookmarks):
        try:
            prompt = f"""Analyze this tweet and provide key insights:

Tweet: {bookmark.get('full_text', bookmark.get('text', ''))}
Author: @{bookmark.get('screen_name', 'unknown')}
Likes: {bookmark.get('favorite_count', 0)}

Provide a brief analysis (2-3 sentences) of the main message and value."""

            response = llm.generate(prompt)
            results.append({
                'index': i + 1,
                'tweet_id': bookmark.get('tweet_id'),
                'author': bookmark.get('screen_name'),
                'text_preview': bookmark.get('full_text', '')[:100],
                'response': response.content,
                'model': response.model
            })
        except Exception as e:
            results.append({
                'index': i + 1,
                'tweet_id': bookmark.get('tweet_id'),
                'error': str(e)
            })

    return results


def test_vision_analysis(provider_name: str, model: str, bookmarks: List[Dict], media_type: str) -> List[Dict]:
    """Test vision analysis with images/videos/gifs using LOCAL media files"""
    from llm_providers import LLMFactory

    results = []

    try:
        if provider_name == 'gemini_pro':
            from llm_providers import GeminiProvider
            llm = GeminiProvider(model='gemini-3-pro-preview')
        else:
            llm = LLMFactory.create(provider_name, model=model)
    except Exception as e:
        console.print(f"[red]Failed to create {provider_name} provider: {e}[/red]")
        return [{'error': str(e), 'provider': provider_name}]

    for i, bookmark in enumerate(bookmarks):
        try:
            # Get LOCAL media files (enriched by media_linker)
            local_media_paths = bookmark.get('local_media_paths', [])
            local_media_types = bookmark.get('local_media_types', [])

            # Filter by media type if needed
            if media_type == 'image':
                media_files = [p for p, t in zip(local_media_paths, local_media_types) if t == 'image']
            elif media_type == 'video':
                media_files = [p for p, t in zip(local_media_paths, local_media_types) if t == 'video']
            elif media_type == 'gif':
                media_files = [p for p, t in zip(local_media_paths, local_media_types) if t == 'gif']
            else:
                media_files = local_media_paths

            duration_info = ""
            if media_type == 'video':
                duration = bookmark.get('_video_duration_ms', 0)
                duration_info = f"\nVideo duration: {format_duration(duration)}"

            prompt = f"""Analyze this tweet and its {media_type} content.

Tweet text: {bookmark.get('full_text', bookmark.get('text', ''))}
Author: @{bookmark.get('screen_name', 'unknown')}
Media type: {media_type}{duration_info}
Number of {media_type}s attached: {len(media_files)}

IMPORTANT: Look at the attached {media_type}(s) carefully. Extract ALL useful information visible in the {media_type}.
- If it's a screenshot of text/code, transcribe the relevant parts
- If it's a diagram/chart, describe what it shows
- If it's a tutorial or guide image, extract the steps
- If it contains a list, extract the complete list

Provide a comprehensive analysis that extracts REAL VALUE from this bookmark. Think: "Why did the user save this? What information do they want to keep?"
"""

            # Use generate_with_vision if we have local media files
            if media_files:
                console.print(f"    [dim]Using {len(media_files)} local {media_type} file(s)[/dim]")
                response = llm.generate_with_vision(prompt, media_files)
            else:
                # Fallback to regular generate if no local files
                console.print(f"    [yellow]No local media found, using text-only analysis[/yellow]")
                response = llm.generate(prompt)

            results.append({
                'index': i + 1,
                'tweet_id': bookmark.get('tweet_id'),
                'author': bookmark.get('screen_name'),
                'text_preview': bookmark.get('full_text', '')[:80],
                'media_type': media_type,
                'media_count': len(media_files),
                'local_files_used': len(media_files),
                'duration_ms': bookmark.get('_video_duration_ms', 0) if media_type == 'video' else None,
                'response': response.content,
                'model': response.model,
                'images_processed': getattr(response, 'images_processed', 0)
            })
        except Exception as e:
            import traceback
            console.print(f"    [red]Error: {e}[/red]")
            results.append({
                'index': i + 1,
                'tweet_id': bookmark.get('tweet_id'),
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    return results


def print_results(results: List[Dict], title: str):
    """Print test results"""
    console.print(f"\n[bold cyan]{title}[/bold cyan]")

    for r in results:
        if 'error' in r:
            console.print(f"  [{r.get('index', '?')}] [red]Error: {r['error']}[/red]")
        else:
            duration = ""
            if r.get('duration_ms'):
                duration = f" [{format_duration(r['duration_ms'])}]"

            console.print(f"\n  [{r['index']}] @{r.get('author', 'unknown')}{duration}")
            console.print(f"  [dim]{r.get('text_preview', '')[:60]}...[/dim]")
            console.print(Panel(r.get('response', 'No response')[:500], border_style="green"))


def save_results(all_results: Dict, output_file: str):
    """Save all results to markdown file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Vision Model Comparison Test Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for provider, provider_results in all_results.items():
            f.write(f"\n## {provider}\n\n")

            for media_type, results in provider_results.items():
                f.write(f"### {media_type.replace('_', ' ').title()}\n\n")

                for r in results:
                    if 'error' in r:
                        f.write(f"- **Error:** {r['error']}\n\n")
                    else:
                        f.write(f"#### Tweet {r.get('index', '?')}: @{r.get('author', 'unknown')}\n\n")
                        if r.get('duration_ms'):
                            f.write(f"**Duration:** {format_duration(r['duration_ms'])}\n\n")
                        f.write(f"> {r.get('text_preview', '')[:100]}...\n\n")
                        f.write(f"**Model:** `{r.get('model', 'unknown')}`\n\n")
                        f.write(f"**Response:**\n{r.get('response', 'No response')}\n\n")
                        f.write("---\n\n")

    console.print(f"\n[green]Results saved to {output_file}[/green]")


def main():
    console.print(Panel.fit(
        "[bold cyan]Vision Model Comparison Test[/bold cyan]\n"
        "Testing GPT-5.2, Claude Opus 4.5, Gemini 2.5 Flash, Gemini 3 Pro Preview\n"
        "[green]Now using LOCAL media files for real vision analysis![/green]",
        border_style="cyan"
    ))

    # Load bookmarks
    bookmarks_file = "~/Downloads/twillot-bookmark.json"
    console.print(f"\n[cyan]Loading bookmarks from {bookmarks_file}...[/cyan]")

    try:
        bookmarks = load_bookmarks(bookmarks_file)
        console.print(f"[green]Loaded {len(bookmarks)} bookmarks[/green]")
    except FileNotFoundError:
        console.print(f"[red]File not found: {bookmarks_file}[/red]")
        return
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON: {e}[/red]")
        return

    # Initialize MediaLinker and enrich bookmarks with local media paths
    console.print(f"\n[cyan]Scanning local media folder: {MEDIA_FOLDER}...[/cyan]")
    try:
        media_linker = MediaLinker(MEDIA_FOLDER)
        media_stats = media_linker.get_stats()
        console.print(f"[green]Found {media_stats['total_files']} media files for {media_stats['tweets_with_media']} tweets[/green]")
        console.print(f"  Images: {media_stats['images']}, Videos: {media_stats['videos']}, GIFs: {media_stats['gifs']}")

        # Enrich bookmarks with local media paths
        bookmarks = media_linker.enrich_bookmarks(bookmarks)
        with_local_media = sum(1 for b in bookmarks if b.get('local_media_paths'))
        console.print(f"[green]Enriched {with_local_media}/{len(bookmarks)} bookmarks with local media paths[/green]")
    except FileNotFoundError as e:
        console.print(f"[yellow]Warning: Media folder not found: {e}[/yellow]")
        console.print("[yellow]Continuing without local media files...[/yellow]")

    # Select test samples (now with local_media_paths)
    samples = select_test_samples(bookmarks)

    console.print("\n[bold]Test Samples Selected:[/bold]")
    table = Table(show_header=True)
    table.add_column("Type")
    table.add_column("Count")
    table.add_column("Details")

    table.add_row("Text Only", str(len(samples['text_only'])), "No media")
    table.add_row("With Image", str(len(samples['with_image'])), "Image attachments")

    # Show video durations
    video_details = ", ".join([
        format_duration(v.get('_video_duration_ms', 0))
        for v in samples['with_video']
    ])
    table.add_row("With Video", str(len(samples['with_video'])), f"Durations: {video_details}")
    table.add_row("With GIF", str(len(samples['with_gif'])), "Animated GIFs")

    console.print(table)

    # Define models to test
    models = [
        ('openai', 'gpt-5.2', 'GPT-5.2'),
        ('anthropic', 'claude-opus-4-5-20251101', 'Claude Opus 4.5'),
        ('gemini', 'gemini-2.5-flash', 'Gemini 2.5 Flash'),
        ('gemini_pro', 'gemini-3-pro-preview', 'Gemini 3 Pro Preview'),
    ]

    all_results = {}

    for provider_key, model, display_name in models:
        console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
        console.print(f"[bold yellow]Testing {display_name} ({model})[/bold yellow]")
        console.print(f"[bold yellow]{'='*60}[/bold yellow]")

        provider_results = {}

        # Test text-only
        console.print("\n[cyan]Testing text-only bookmarks...[/cyan]")
        provider_results['text_only'] = test_text_analysis(
            provider_key, model, samples['text_only']
        )
        print_results(provider_results['text_only'], f"{display_name} - Text Only")

        # Test with images
        console.print("\n[cyan]Testing image bookmarks...[/cyan]")
        provider_results['with_image'] = test_vision_analysis(
            provider_key, model, samples['with_image'], 'image'
        )
        print_results(provider_results['with_image'], f"{display_name} - With Images")

        # Test with videos
        console.print("\n[cyan]Testing video bookmarks...[/cyan]")
        provider_results['with_video'] = test_vision_analysis(
            provider_key, model, samples['with_video'], 'video'
        )
        print_results(provider_results['with_video'], f"{display_name} - With Videos")

        # Test with GIFs
        if samples['with_gif']:
            console.print("\n[cyan]Testing GIF bookmarks...[/cyan]")
            provider_results['with_gif'] = test_vision_analysis(
                provider_key, model, samples['with_gif'], 'gif'
            )
            print_results(provider_results['with_gif'], f"{display_name} - With GIFs")

        all_results[display_name] = provider_results

    # Save results
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    save_results(all_results, output_file)

    console.print("\n[bold green]Test complete![/bold green]")


if __name__ == "__main__":
    main()
