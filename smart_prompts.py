"""
Smart Prompts - Intelligent prompt selection based on tweet content type

This module analyzes tweets and selects the most appropriate prompt
to extract maximum value from each bookmark. Instead of generic
"summarize this tweet" prompts, it asks specific questions based on
what the user likely wants to extract.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ContentType(Enum):
    """Types of content we can identify in tweets"""
    ARTICLE_LINK = "article_link"           # Link to article/blog post
    TOP_LIST = "top_list"                   # Top 10, best of, rankings
    TUTORIAL_GUIDE = "tutorial_guide"       # How-to, guide, tutorial
    TOOL_ANNOUNCEMENT = "tool_announcement"  # New tool/library release
    CODE_SNIPPET = "code_snippet"           # Code or prompt example
    OPINION_TAKE = "opinion_take"           # Hot take, opinion, commentary
    NEWS_UPDATE = "news_update"             # Breaking news, announcement
    THREAD_CONTENT = "thread_content"       # Multi-tweet thread
    VIDEO_CONTENT = "video_content"         # Video explanation/demo
    SCREENSHOT_INFO = "screenshot_info"     # Screenshot with information
    MEME_HUMOR = "meme_humor"               # Meme or humorous content
    UNKNOWN = "unknown"


@dataclass
class SmartPrompt:
    """A prompt tailored to a specific content type"""
    content_type: ContentType
    prompt: str
    system_prompt: str
    expected_output: str  # Description of what the output should contain


# Detection patterns for content types
CONTENT_TYPE_PATTERNS = {
    ContentType.TOP_LIST: [
        r'top\s+\d+', r'best\s+\d+', r'\d+\s+best', r'\d+\s+things',
        r'my\s+favorite', r'ranking', r'list\s+of', r'\d+\s+ways',
        r'individual\s+top\s+tens?', r'\d+\s+tips', r'\d+\s+tools'
    ],
    ContentType.TUTORIAL_GUIDE: [
        r'how\s+to', r'guide\s+to', r'tutorial', r'step\s+by\s+step',
        r'best\s+practices', r'tips\s+for', r'learn\s+how', r'checklist',
        r'prompt\s+guide', r'cheatsheet', r'cheat\s+sheet'
    ],
    ContentType.TOOL_ANNOUNCEMENT: [
        r'v\d+\.\d+', r'is\s+out', r'just\s+launched', r'releasing',
        r'announcing', r'introducing', r'new\s+release', r'open\s+source',
        r'just\s+dropped', r'npm\s+i', r'pip\s+install'
    ],
    ContentType.CODE_SNIPPET: [
        r'```', r'code:', r'prompt:', r'here\'s\s+how', r'example:',
        r'function\s+\w+', r'def\s+\w+', r'const\s+\w+', r'class\s+\w+'
    ],
    ContentType.OPINION_TAKE: [
        r'hot\s+take', r'unpopular\s+opinion', r'i\s+think', r'imo',
        r'my\s+take', r'controversial', r'change\s+my\s+mind'
    ],
    ContentType.NEWS_UPDATE: [
        r'breaking', r'just\s+in', r'announced', r'officially',
        r'confirmed', r'report:', r'update:'
    ],
    ContentType.THREAD_CONTENT: [
        r'thread', r'\d+/', r'a\s+thread', r'\(thread\)', r'1/'
    ],
}

# Keywords that suggest specific value extraction
VALUE_KEYWORDS = {
    'list': ['top', 'best', 'ranking', 'favorite', 'list'],
    'steps': ['step', 'how to', 'guide', 'tutorial', 'process'],
    'code': ['code', 'prompt', 'script', 'function', 'snippet'],
    'tool': ['tool', 'library', 'framework', 'package', 'release'],
    'opinion': ['think', 'believe', 'opinion', 'take', 'view'],
}


class SmartPromptSelector:
    """Selects the best prompt for a given tweet"""

    PROMPTS = {
        ContentType.ARTICLE_LINK: SmartPrompt(
            content_type=ContentType.ARTICLE_LINK,
            prompt="""This tweet links to an article or blog post.

Tweet: {tweet_text}
{link_content}

TASK: Extract the KEY INFORMATION from this article that the user wanted to save.
Provide:
1. **Main Thesis**: What is the article arguing or explaining? (1-2 sentences)
2. **Key Points**: The 3-5 most important takeaways (bullet points)
3. **Practical Value**: What can the reader DO with this information?
4. **Notable Quotes**: Any particularly insightful quotes worth saving

Be specific and extract REAL INFORMATION, not vague descriptions.""",
            system_prompt="You are an expert at extracting key information from articles. Focus on actionable insights and specific details.",
            expected_output="Main thesis, key points, practical applications, notable quotes"
        ),

        ContentType.TOP_LIST: SmartPrompt(
            content_type=ContentType.TOP_LIST,
            prompt="""This tweet contains or links to a list/ranking.

Tweet: {tweet_text}
{link_content}
{image_analysis}

TASK: Extract THE COMPLETE LIST with details.
Provide:
1. **List Title**: What is being ranked/listed?
2. **Full List**: Every item with brief explanation (numbered)
3. **Source/Author**: Who created this ranking?
4. **Key Insight**: What's the most surprising or valuable item?

The user saved this to reference the list later - GIVE THEM THE LIST.""",
            system_prompt="You are an expert at extracting and organizing lists. Extract every item with relevant details.",
            expected_output="Complete numbered list with descriptions"
        ),

        ContentType.TUTORIAL_GUIDE: SmartPrompt(
            content_type=ContentType.TUTORIAL_GUIDE,
            prompt="""This tweet contains a tutorial, guide, or best practices.

Tweet: {tweet_text}
{link_content}
{image_analysis}

TASK: Extract ACTIONABLE STEPS the user can follow.
Provide:
1. **Goal**: What does this guide help you achieve?
2. **Prerequisites**: What do you need before starting?
3. **Steps**: Numbered, actionable steps (be specific!)
4. **Key Tips**: Important warnings or pro tips
5. **Example**: A concrete example if available

The user saved this to learn HOW TO DO something - TEACH THEM.""",
            system_prompt="You are a technical educator. Extract clear, actionable instructions that someone can follow.",
            expected_output="Step-by-step guide with clear instructions"
        ),

        ContentType.TOOL_ANNOUNCEMENT: SmartPrompt(
            content_type=ContentType.TOOL_ANNOUNCEMENT,
            prompt="""This tweet announces or discusses a tool/library/product.

Tweet: {tweet_text}
{link_content}
{image_analysis}

TASK: Extract PRACTICAL INFORMATION about this tool.
Provide:
1. **What It Is**: One-sentence description
2. **Problem It Solves**: Why would someone use this?
3. **Key Features**: Main capabilities (bullet points)
4. **Installation**: How to get started (command or link)
5. **When to Use**: Specific use cases

The user saved this to potentially USE this tool - HELP THEM GET STARTED.""",
            system_prompt="You are a developer advocate. Explain tools in practical, actionable terms.",
            expected_output="Tool overview with installation and use cases"
        ),

        ContentType.CODE_SNIPPET: SmartPrompt(
            content_type=ContentType.CODE_SNIPPET,
            prompt="""This tweet contains code, prompts, or technical snippets.

Tweet: {tweet_text}
{link_content}
{image_analysis}

TASK: Extract the COMPLETE CODE/PROMPT that the user wanted to save.
Provide:
1. **Purpose**: What does this code/prompt do?
2. **Full Code/Prompt**: The complete, copy-pasteable content (in code block)
3. **How to Use**: Instructions for using it
4. **Customization**: What parts can/should be modified?

The user saved this to USE THIS CODE/PROMPT LATER - GIVE IT TO THEM COMPLETE.""",
            system_prompt="You are a code expert. Extract and format code/prompts for easy copy-pasting.",
            expected_output="Complete code/prompt with usage instructions"
        ),

        ContentType.VIDEO_CONTENT: SmartPrompt(
            content_type=ContentType.VIDEO_CONTENT,
            prompt="""This tweet contains a video.

Tweet: {tweet_text}
{video_analysis}

TASK: Extract KEY INFORMATION from the video content.
Provide:
1. **Video Summary**: What happens in the video? (2-3 sentences)
2. **Key Moments**: Important points or demonstrations shown
3. **Main Takeaway**: What should the viewer remember?
4. **Action Items**: What can someone do after watching?

The user saved this video for a reason - CAPTURE WHY IT'S VALUABLE.""",
            system_prompt="You are an expert at video analysis. Extract the essential information someone would want to remember.",
            expected_output="Video summary with key moments and takeaways"
        ),

        ContentType.SCREENSHOT_INFO: SmartPrompt(
            content_type=ContentType.SCREENSHOT_INFO,
            prompt="""This tweet contains screenshot(s) with information.

Tweet: {tweet_text}
{image_analysis}

TASK: Extract ALL VISIBLE TEXT AND INFORMATION from the screenshot(s).
Provide:
1. **What It Shows**: What is in the screenshot?
2. **Text Content**: Transcribe any visible text (especially lists, code, or instructions)
3. **Key Information**: What's the important data or insight?
4. **Context**: How does the tweet text relate to the screenshot?

The user saved this for the INFORMATION IN THE IMAGE - EXTRACT IT ALL.""",
            system_prompt="You are an OCR and image analysis expert. Extract every piece of useful text and information from images.",
            expected_output="Complete transcription of visible text and information"
        ),

        ContentType.OPINION_TAKE: SmartPrompt(
            content_type=ContentType.OPINION_TAKE,
            prompt="""This tweet expresses an opinion or perspective.

Tweet: {tweet_text}
Author: @{author}
Engagement: {likes} likes

TASK: Capture the ESSENCE of this perspective.
Provide:
1. **The Take**: What is the author arguing? (1-2 sentences)
2. **Supporting Points**: How do they support their argument?
3. **Why It Matters**: Why might this perspective be valuable?
4. **Counter-View**: What's the opposing perspective?

The user saved this opinion for a reason - CAPTURE THE INSIGHT.""",
            system_prompt="You are a critical thinker. Capture opinions clearly while noting different perspectives.",
            expected_output="Clear summary of the opinion with context"
        ),

        ContentType.NEWS_UPDATE: SmartPrompt(
            content_type=ContentType.NEWS_UPDATE,
            prompt="""This tweet contains news or an announcement.

Tweet: {tweet_text}
{link_content}

TASK: Extract the NEWS FACTS.
Provide:
1. **What Happened**: The key news in 1-2 sentences
2. **Who/What**: Key entities involved
3. **When**: When did this happen?
4. **Impact**: Why does this matter? Who does it affect?
5. **Source**: Where is this information from?

The user saved this news - GIVE THEM THE FACTS.""",
            system_prompt="You are a journalist. Extract facts clearly and accurately.",
            expected_output="News summary with key facts and impact"
        ),

        ContentType.UNKNOWN: SmartPrompt(
            content_type=ContentType.UNKNOWN,
            prompt="""Analyze this tweet and extract maximum value.

Tweet: {tweet_text}
Author: @{author}
{link_content}
{image_analysis}

TASK: Determine WHY the user might have saved this and extract that value.
Consider:
- Is there information to extract (lists, steps, code)?
- Is there a link with valuable content?
- Is there an image with information?
- Is it an insight or perspective worth remembering?

Provide the most USEFUL summary based on what this tweet contains.""",
            system_prompt="You are an expert at extracting value from content. Focus on actionable, specific information.",
            expected_output="Comprehensive analysis based on content type"
        ),
    }

    @classmethod
    def detect_content_type(cls, tweet: Dict) -> ContentType:
        """Detect the type of content in a tweet"""
        text = tweet.get('full_text', tweet.get('text', '')).lower()
        has_link = bool(tweet.get('urls')) or 'http' in text
        has_image = tweet.get('has_image', False)
        has_video = tweet.get('has_video', False)

        # Check video first
        if has_video:
            return ContentType.VIDEO_CONTENT

        # Check for specific patterns
        for content_type, patterns in CONTENT_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return content_type

        # Image with limited text suggests screenshot info
        if has_image and len(text) < 100:
            return ContentType.SCREENSHOT_INFO

        # Link suggests article
        if has_link:
            return ContentType.ARTICLE_LINK

        # Default to unknown
        return ContentType.UNKNOWN

    @classmethod
    def get_prompt(cls, content_type: ContentType) -> SmartPrompt:
        """Get the prompt for a content type"""
        return cls.PROMPTS.get(content_type, cls.PROMPTS[ContentType.UNKNOWN])

    @classmethod
    def build_prompt(
        cls,
        tweet: Dict,
        link_content: Optional[str] = None,
        image_analysis: Optional[str] = None,
        video_analysis: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Build a complete prompt for a tweet

        Returns:
            Tuple of (prompt, system_prompt)
        """
        content_type = cls.detect_content_type(tweet)
        smart_prompt = cls.get_prompt(content_type)

        # Build substitution dict
        subs = {
            'tweet_text': tweet.get('full_text', tweet.get('text', '')),
            'author': tweet.get('screen_name', 'unknown'),
            'likes': tweet.get('favorite_count', 0),
            'link_content': '',
            'image_analysis': '',
            'video_analysis': '',
        }

        # Add link content if available
        if link_content:
            subs['link_content'] = f"\n---\nLinked Content:\n{link_content}\n---"

        # Add image analysis if available
        if image_analysis:
            subs['image_analysis'] = f"\n---\nImage Content:\n{image_analysis}\n---"

        # Add video analysis if available
        if video_analysis:
            subs['video_analysis'] = f"\n---\nVideo Content:\n{video_analysis}\n---"

        # Format the prompt
        prompt = smart_prompt.prompt.format(**subs)

        return prompt, smart_prompt.system_prompt

    @classmethod
    def get_all_types(cls) -> List[ContentType]:
        """Get all content types"""
        return list(ContentType)

    @classmethod
    def describe_type(cls, content_type: ContentType) -> str:
        """Get a description of what a content type means"""
        descriptions = {
            ContentType.ARTICLE_LINK: "Tweet links to an article or blog post",
            ContentType.TOP_LIST: "Tweet contains or links to a list/ranking",
            ContentType.TUTORIAL_GUIDE: "Tweet contains a how-to or guide",
            ContentType.TOOL_ANNOUNCEMENT: "Tweet announces a tool or library",
            ContentType.CODE_SNIPPET: "Tweet contains code or prompts",
            ContentType.OPINION_TAKE: "Tweet expresses an opinion",
            ContentType.NEWS_UPDATE: "Tweet contains news",
            ContentType.THREAD_CONTENT: "Tweet is part of a thread",
            ContentType.VIDEO_CONTENT: "Tweet contains a video",
            ContentType.SCREENSHOT_INFO: "Tweet contains screenshot with info",
            ContentType.MEME_HUMOR: "Tweet is humorous/meme content",
            ContentType.UNKNOWN: "Content type not detected",
        }
        return descriptions.get(content_type, "Unknown content type")


# Example usage
if __name__ == "__main__":
    # Test detection
    test_tweets = [
        {"text": "Top 10 best AI tools for developers in 2025", "has_image": True},
        {"text": "Here's how to use Claude's new thinking mode:", "has_image": True},
        {"text": "OpenSkills v1.3.0 is out! npm i -g openskills", "has_image": False},
        {"text": "Hot take: LLMs are overrated", "has_image": False},
        {"text": "Check out this article: https://t.co/abc123", "has_image": False},
    ]

    for tweet in test_tweets:
        content_type = SmartPromptSelector.detect_content_type(tweet)
        print(f"Tweet: {tweet['text'][:50]}...")
        print(f"  Type: {content_type.value}")
        print(f"  Description: {SmartPromptSelector.describe_type(content_type)}")
        print()
