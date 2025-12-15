"""
Media Linker - Links Twillot JSON exports with local media files

The Twillot extension exports media files with the naming pattern:
{screen_name}-{tweet_id}-{media_id}.{ext}

This module scans the media folder and builds an index to match
tweets with their corresponding local media files.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MediaFile:
    """Represents a local media file"""
    path: str
    tweet_id: str
    screen_name: str
    media_id: str
    extension: str
    media_type: str  # 'image', 'video', 'gif'


class MediaLinker:
    """Links Twillot JSON bookmarks with local media files"""

    # Supported media extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
    VIDEO_EXTENSIONS = {'.mp4', '.mov', '.webm'}
    GIF_EXTENSIONS = {'.gif'}

    # Filename pattern: {screen_name}-{tweet_id}-{media_id}.{ext}
    # Example: johnrushx-1891684077994123426-GkCco5taAAARYUs.png
    FILENAME_PATTERN = re.compile(
        r'^(?P<screen_name>[^-]+)-(?P<tweet_id>\d+)-(?P<media_id>[^.]+)\.(?P<ext>\w+)$'
    )

    def __init__(self, media_folder: str):
        """
        Initialize the MediaLinker

        Args:
            media_folder: Path to the Twillot media folder
                         (e.g., ~/Downloads/twillot-media-files-by-date/)
        """
        self.media_folder = Path(media_folder).expanduser().resolve()
        self.media_index: Dict[str, List[MediaFile]] = {}  # tweet_id -> [MediaFile]
        self._indexed = False

    def build_index(self) -> Dict[str, List[str]]:
        """
        Scan media folder and build an index of tweet_id -> file paths

        Returns:
            Dictionary mapping tweet_id to list of file paths
        """
        if not self.media_folder.exists():
            raise FileNotFoundError(f"Media folder not found: {self.media_folder}")

        self.media_index.clear()

        # Walk through all subdirectories (organized by date)
        for root, dirs, files in os.walk(self.media_folder):
            for filename in files:
                media_file = self._parse_filename(filename, root)
                if media_file:
                    if media_file.tweet_id not in self.media_index:
                        self.media_index[media_file.tweet_id] = []
                    self.media_index[media_file.tweet_id].append(media_file)

        self._indexed = True

        # Return simplified dict for compatibility
        return {
            tweet_id: [mf.path for mf in files]
            for tweet_id, files in self.media_index.items()
        }

    def _parse_filename(self, filename: str, directory: str) -> Optional[MediaFile]:
        """
        Parse a media filename and create a MediaFile object

        Args:
            filename: The filename to parse
            directory: The directory containing the file

        Returns:
            MediaFile object if parsing succeeds, None otherwise
        """
        match = self.FILENAME_PATTERN.match(filename)
        if not match:
            return None

        ext = f".{match.group('ext').lower()}"

        # Determine media type from extension
        if ext in self.IMAGE_EXTENSIONS:
            media_type = 'image'
        elif ext in self.VIDEO_EXTENSIONS:
            media_type = 'video'
        elif ext in self.GIF_EXTENSIONS:
            media_type = 'gif'
        else:
            return None  # Unknown extension, skip

        return MediaFile(
            path=os.path.join(directory, filename),
            tweet_id=match.group('tweet_id'),
            screen_name=match.group('screen_name'),
            media_id=match.group('media_id'),
            extension=ext,
            media_type=media_type
        )

    def get_media_for_tweet(self, tweet_id: str) -> List[str]:
        """
        Get local media file paths for a specific tweet

        Args:
            tweet_id: The tweet ID to look up

        Returns:
            List of file paths for the tweet's media
        """
        if not self._indexed:
            self.build_index()

        media_files = self.media_index.get(str(tweet_id), [])
        return [mf.path for mf in media_files]

    def get_media_files_for_tweet(self, tweet_id: str) -> List[MediaFile]:
        """
        Get MediaFile objects for a specific tweet (with full metadata)

        Args:
            tweet_id: The tweet ID to look up

        Returns:
            List of MediaFile objects
        """
        if not self._indexed:
            self.build_index()

        return self.media_index.get(str(tweet_id), [])

    def get_images_for_tweet(self, tweet_id: str) -> List[str]:
        """
        Get only image file paths for a tweet (excludes videos/gifs)

        Args:
            tweet_id: The tweet ID to look up

        Returns:
            List of image file paths
        """
        media_files = self.get_media_files_for_tweet(tweet_id)
        return [mf.path for mf in media_files if mf.media_type == 'image']

    def enrich_bookmarks(self, bookmarks: List[Dict]) -> List[Dict]:
        """
        Add local_media_paths field to each bookmark

        Args:
            bookmarks: List of bookmark dictionaries

        Returns:
            Enriched bookmarks with local_media_paths field
        """
        if not self._indexed:
            self.build_index()

        enriched = []
        for bookmark in bookmarks:
            # Create a copy to avoid mutating the original
            enriched_bookmark = bookmark.copy()

            # Get tweet ID (handle different field names)
            # IMPORTANT: Use tweet_id first, not id (id has format bookmark_xxx_yyy)
            tweet_id = str(
                bookmark.get('tweet_id') or
                bookmark.get('id') or
                ''
            )

            if tweet_id:
                media_files = self.media_index.get(tweet_id, [])
                enriched_bookmark['local_media_paths'] = [mf.path for mf in media_files]
                enriched_bookmark['local_media_types'] = [mf.media_type for mf in media_files]
            else:
                enriched_bookmark['local_media_paths'] = []
                enriched_bookmark['local_media_types'] = []

            enriched.append(enriched_bookmark)

        return enriched

    def get_stats(self) -> Dict:
        """
        Get statistics about the indexed media

        Returns:
            Dictionary with media statistics
        """
        if not self._indexed:
            self.build_index()

        total_files = sum(len(files) for files in self.media_index.values())
        total_tweets = len(self.media_index)

        # Count by type
        images = videos = gifs = 0
        for files in self.media_index.values():
            for mf in files:
                if mf.media_type == 'image':
                    images += 1
                elif mf.media_type == 'video':
                    videos += 1
                elif mf.media_type == 'gif':
                    gifs += 1

        return {
            'total_files': total_files,
            'tweets_with_media': total_tweets,
            'images': images,
            'videos': videos,
            'gifs': gifs,
            'media_folder': str(self.media_folder)
        }

    def filter_bookmarks_with_media(
        self,
        bookmarks: List[Dict],
        media_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Filter bookmarks to only those with local media files

        Args:
            bookmarks: List of bookmark dictionaries
            media_type: Optional filter - 'image', 'video', 'gif', or None for all

        Returns:
            Filtered list of bookmarks that have local media
        """
        enriched = self.enrich_bookmarks(bookmarks)

        filtered = []
        for bookmark in enriched:
            local_paths = bookmark.get('local_media_paths', [])
            local_types = bookmark.get('local_media_types', [])

            if not local_paths:
                continue

            if media_type is None:
                # Include all bookmarks with any media
                filtered.append(bookmark)
            else:
                # Filter by specific media type
                if media_type in local_types:
                    filtered.append(bookmark)

        return filtered


# Example usage
if __name__ == "__main__":
    import json

    # Example paths
    media_folder = "~/Downloads/twillot-media-files-by-date/"
    json_file = "~/Downloads/twillot-bookmark.json"

    # Initialize linker
    linker = MediaLinker(media_folder)

    try:
        # Build index
        index = linker.build_index()
        print(f"Indexed {len(index)} tweets with media")

        # Get stats
        stats = linker.get_stats()
        print(f"Stats: {stats}")

        # Load bookmarks and enrich
        with open(Path(json_file).expanduser(), 'r') as f:
            bookmarks = json.load(f)

        enriched = linker.enrich_bookmarks(bookmarks)

        # Count bookmarks with local media
        with_media = [b for b in enriched if b.get('local_media_paths')]
        print(f"Bookmarks with local media: {len(with_media)}/{len(bookmarks)}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
