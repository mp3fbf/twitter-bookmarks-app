"""
Modular LLM Provider Interface
Supports OpenAI, Anthropic, and Google Gemini
Includes vision/multimodal capabilities for image analysis
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import os
import base64
import mimetypes
from pathlib import Path
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized response from LLM providers"""
    content: str
    model: str
    usage: Optional[Dict] = None
    images_processed: int = 0


def encode_image_to_base64(image_path: str) -> Tuple[str, str]:
    """
    Encode an image file to base64

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (base64_data, media_type)

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the file type is not supported
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Determine media type
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        # Fallback based on extension
        ext = path.suffix.lower()
        mime_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        mime_type = mime_map.get(ext)

    if mime_type is None or not mime_type.startswith('image/'):
        raise ValueError(f"Unsupported image type: {path.suffix}")

    # Read and encode
    with open(path, 'rb') as f:
        image_data = base64.standard_b64encode(f.read()).decode('utf-8')

    return image_data, mime_type


def get_image_size_mb(image_path: str) -> float:
    """Get image file size in MB"""
    return Path(image_path).stat().st_size / (1024 * 1024)


def encode_video_to_base64(video_path: str) -> Tuple[str, str]:
    """
    Encode a video file to base64

    Args:
        video_path: Path to the video file

    Returns:
        Tuple of (base64_data, media_type)
    """
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Determine media type
    ext = path.suffix.lower()
    mime_map = {
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime',
        '.webm': 'video/webm',
        '.avi': 'video/x-msvideo',
    }
    mime_type = mime_map.get(ext)

    if mime_type is None:
        raise ValueError(f"Unsupported video type: {ext}")

    # Read and encode
    with open(path, 'rb') as f:
        video_data = base64.standard_b64encode(f.read()).decode('utf-8')

    return video_data, mime_type


def get_video_size_mb(video_path: str) -> float:
    """Get video file size in MB"""
    return Path(video_path).stat().st_size / (1024 * 1024)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate a response from the LLM"""
        pass

    @abstractmethod
    def generate_with_vision(
        self,
        prompt: str,
        images: List[str],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate a response with image understanding

        Args:
            prompt: Text prompt for the model
            images: List of image file paths
            system_prompt: Optional system prompt

        Returns:
            LLMResponse with content and metadata
        """
        pass

    @abstractmethod
    def summarize_tweets(self, tweets: List[Dict], max_length: int = 500) -> str:
        """Summarize a list of tweets"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        # Check if API key looks valid
        if self.api_key.startswith("your_") or self.api_key.endswith("_here"):
            raise ValueError(f"Invalid API key format. Please set a real OpenAI API key in your .env file")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Check if this is a model that uses max_completion_tokens (o1/o3/gpt-5.x)
            if self.model.startswith('o1') or self.model.startswith('o3') or self.model.startswith('gpt-5'):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=4000,  # Increased for longer responses
                    timeout=120.0
                )
            else:
                # Regular GPT models (gpt-4, gpt-4o, etc.)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000,
                    timeout=60.0
                )
            
            # Handle potential None content
            content = response.choices[0].message.content
            if content is None:
                content = ""

            return LLMResponse(
                content=content,
                model=self.model,
                usage=response.usage.model_dump() if response.usage else None
            )
        except Exception as e:
            # Re-raise with more context if needed
            if "Invalid API key" in str(e) or "Incorrect API key" in str(e):
                raise ValueError(f"OpenAI API key error: {str(e)}. Please check your .env file.")
            raise

    def generate_with_vision(
        self,
        prompt: str,
        images: List[str],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate a response with image understanding using GPT-4o or GPT-5.2

        Args:
            prompt: Text prompt
            images: List of image file paths
            system_prompt: Optional system prompt

        Returns:
            LLMResponse with content
        """
        # Use vision-capable model (GPT-5.2 preferred)
        vision_model = self.model
        if not any(v in self.model.lower() for v in ['gpt-5', 'gpt-4o', 'gpt-4-turbo']):
            vision_model = 'gpt-5.2'  # Default to GPT-5.2 for vision

        # Build content array with text and images
        content = [{"type": "text", "text": prompt}]

        images_processed = 0
        for image_path in images:
            try:
                # Check if it's a URL or local file
                if image_path.startswith('http://') or image_path.startswith('https://'):
                    # Direct URL - OpenAI supports this natively
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": image_path,
                            "detail": "high"
                        }
                    })
                    images_processed += 1
                else:
                    # Local file - encode to base64
                    base64_data, media_type = encode_image_to_base64(image_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{base64_data}",
                            "detail": "high"  # Use high detail for better analysis
                        }
                    })
                    images_processed += 1
            except (FileNotFoundError, ValueError) as e:
                print(f"[yellow]Warning: Skipping image {image_path}: {e}[/yellow]")
                continue

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        try:
            # GPT-5.x uses max_completion_tokens instead of max_tokens
            if vision_model.startswith('gpt-5'):
                response = self.client.chat.completions.create(
                    model=vision_model,
                    messages=messages,
                    max_completion_tokens=4096,
                    timeout=180.0
                )
            else:
                response = self.client.chat.completions.create(
                    model=vision_model,
                    messages=messages,
                    max_tokens=4096,
                    timeout=180.0
                )

            return LLMResponse(
                content=response.choices[0].message.content,
                model=vision_model,
                usage=response.usage.model_dump() if response.usage else None,
                images_processed=images_processed
            )
        except Exception as e:
            if "Invalid API key" in str(e) or "Incorrect API key" in str(e):
                raise ValueError(f"OpenAI API key error: {str(e)}. Please check your .env file.")
            raise

    def generate_with_video(
        self,
        prompt: str,
        video_path: str,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate a response with video understanding using GPT-5.2

        Args:
            prompt: Text prompt
            video_path: Path to video file
            system_prompt: Optional system prompt

        Returns:
            LLMResponse with content
        """
        # Use video-capable model
        vision_model = self.model
        if not self.model.startswith('gpt-5'):
            vision_model = 'gpt-5.2'

        # Check video size (limit to 50MB for reasonable processing)
        video_size = get_video_size_mb(video_path)
        if video_size > 50:
            raise ValueError(f"Video too large: {video_size:.1f}MB (max 50MB)")

        # Encode video
        video_data, media_type = encode_video_to_base64(video_path)

        # Build content with video
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "video_url",
                "video_url": {
                    "url": f"data:{media_type};base64,{video_data}"
                }
            }
        ]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        try:
            response = self.client.chat.completions.create(
                model=vision_model,
                messages=messages,
                max_completion_tokens=4096,
                timeout=300.0  # Longer timeout for video
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                model=vision_model,
                usage=response.usage.model_dump() if response.usage else None,
                images_processed=1  # Count video as 1 media item
            )
        except Exception as e:
            raise RuntimeError(f"GPT video analysis error: {e}")

    def summarize_tweets(self, tweets: List[Dict], max_length: int = 500) -> str:
        tweet_texts = "\n\n".join([
            f"Tweet {i+1}: {tweet.get('text', '')}"
            for i, tweet in enumerate(tweets[:10])  # Limit to 10 tweets for context
        ])

        prompt = f"""Please summarize these Twitter bookmarks into key insights and actionable learnings:

{tweet_texts}

Provide a concise summary (max {max_length} words) that captures the main themes, important tips, and actionable advice."""

        response = self.generate(prompt)
        return response.content


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = model
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.messages.create(
            model=self.model,
            messages=messages,
            system=system_prompt if system_prompt else "You are a helpful assistant.",
            max_tokens=1000,
            temperature=0.7
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            usage={"input_tokens": response.usage.input_tokens,
                   "output_tokens": response.usage.output_tokens}
        )

    def generate_with_vision(
        self,
        prompt: str,
        images: List[str],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate a response with image understanding using Claude vision

        Args:
            prompt: Text prompt
            images: List of image file paths (max 20 images, 5MB each)
            system_prompt: Optional system prompt

        Returns:
            LLMResponse with content
        """
        # Claude vision models
        vision_model = self.model
        if 'claude-3' not in self.model and 'claude-opus-4' not in self.model:
            vision_model = 'claude-opus-4-5-20251101'  # Default to Opus 4.5 for vision

        # Build content blocks with images
        content = []

        # Validate and add images (max 20 images, 5MB each for Claude)
        images_processed = 0
        for i, image_path in enumerate(images[:20]):  # Claude limit: 20 images
            try:
                # Check size limit (5MB)
                size_mb = get_image_size_mb(image_path)
                if size_mb > 5:
                    print(f"[yellow]Warning: Skipping {image_path} - exceeds 5MB limit ({size_mb:.1f}MB)[/yellow]")
                    continue

                base64_data, media_type = encode_image_to_base64(image_path)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data
                    }
                })
                images_processed += 1
            except (FileNotFoundError, ValueError) as e:
                print(f"[yellow]Warning: Skipping image {image_path}: {e}[/yellow]")
                continue

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        response = self.client.messages.create(
            model=vision_model,
            messages=messages,
            system=system_prompt if system_prompt else "You are a helpful assistant that analyzes images.",
            max_tokens=4096,
        )

        return LLMResponse(
            content=response.content[0].text,
            model=vision_model,
            usage={"input_tokens": response.usage.input_tokens,
                   "output_tokens": response.usage.output_tokens},
            images_processed=images_processed
        )

    def summarize_tweets(self, tweets: List[Dict], max_length: int = 500) -> str:
        tweet_texts = "\n\n".join([
            f"Tweet {i+1}: {tweet.get('text', '')}"
            for i, tweet in enumerate(tweets[:10])
        ])

        prompt = f"""Please summarize these Twitter bookmarks into key insights and actionable learnings:

{tweet_texts}

Provide a concise summary (max {max_length} words) that captures the main themes, important tips, and actionable advice."""

        response = self.generate(prompt)
        return response.content


class GeminiProvider(LLMProvider):
    """Google Gemini provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        response = self.model.generate_content(full_prompt)

        return LLMResponse(
            content=response.text,
            model=self.model_name,
            usage=None  # Gemini doesn't provide token usage in the same way
        )

    def generate_with_vision(
        self,
        prompt: str,
        images: List[str],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate a response with image understanding using Gemini vision

        Gemini 2.5 Flash is recommended for high-volume, cost-effective vision tasks.
        Gemini 3 Pro Preview offers best quality for complex document/image analysis.

        Args:
            prompt: Text prompt
            images: List of image file paths
            system_prompt: Optional system prompt

        Returns:
            LLMResponse with content
        """
        # Build content parts
        content_parts = []

        # Add system prompt if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Load and add images using base64 (no PIL needed)
        images_processed = 0
        for image_path in images:
            try:
                # Encode image to base64 - Gemini accepts inline_data format
                base64_data, media_type = encode_image_to_base64(image_path)
                content_parts.append({
                    "mime_type": media_type,
                    "data": base64_data
                })
                images_processed += 1
            except FileNotFoundError:
                print(f"Warning: Image not found: {image_path}")
                continue
            except Exception as e:
                print(f"Warning: Error loading {image_path}: {e}")
                continue

        # Add text prompt
        content_parts.append(full_prompt)

        try:
            response = self.model.generate_content(content_parts)

            return LLMResponse(
                content=response.text,
                model=self.model_name,
                usage=None,
                images_processed=images_processed
            )
        except Exception as e:
            raise RuntimeError(f"Gemini vision error: {e}")

    def generate_with_video(
        self,
        prompt: str,
        video_path: str,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate a response with video understanding using Gemini

        Gemini has native video support and can process video files directly.

        Args:
            prompt: Text prompt
            video_path: Path to video file
            system_prompt: Optional system prompt

        Returns:
            LLMResponse with content
        """
        import google.generativeai as genai

        # Check video exists and size
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        video_size = get_video_size_mb(video_path)
        if video_size > 100:  # Gemini has higher limits
            raise ValueError(f"Video too large: {video_size:.1f}MB (max 100MB)")

        # Upload video to Gemini
        video_file = genai.upload_file(path=video_path)

        # Wait for processing
        import time
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise RuntimeError(f"Video processing failed: {video_file.state.name}")

        # Build prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            response = self.model.generate_content([video_file, full_prompt])

            return LLMResponse(
                content=response.text,
                model=self.model_name,
                usage=None,
                images_processed=1  # Count video as 1 media item
            )
        except Exception as e:
            raise RuntimeError(f"Gemini video error: {e}")
        finally:
            # Clean up uploaded file
            try:
                genai.delete_file(video_file.name)
            except:
                pass

    def summarize_tweets(self, tweets: List[Dict], max_length: int = 500) -> str:
        tweet_texts = "\n\n".join([
            f"Tweet {i+1}: {tweet.get('text', '')}"
            for i, tweet in enumerate(tweets[:10])
        ])

        prompt = f"""Please summarize these Twitter bookmarks into key insights and actionable learnings:

{tweet_texts}

Provide a concise summary (max {max_length} words) that captures the main themes, important tips, and actionable advice."""

        response = self.generate(prompt)
        return response.content


class LLMFactory:
    """Factory class to create LLM providers"""
    
    @staticmethod
    def create(provider: str, api_key: Optional[str] = None, model: Optional[str] = None) -> LLMProvider:
        """
        Create an LLM provider instance
        
        Args:
            provider: One of 'openai', 'anthropic', 'gemini'
            api_key: API key (optional if set in environment)
            model: Model name (optional, uses defaults)
        
        Returns:
            LLMProvider instance
        """
        providers = {
            'openai': (OpenAIProvider, 'gpt-5.2'),
            'anthropic': (AnthropicProvider, 'claude-opus-4-5-20251101'),
            'gemini': (GeminiProvider, 'gemini-2.5-flash')
        }
        
        if provider.lower() not in providers:
            raise ValueError(f"Unknown provider: {provider}. Choose from: {list(providers.keys())}")
        
        provider_class, default_model = providers[provider.lower()]
        actual_model = model or default_model
        
        return provider_class(api_key=api_key, model=actual_model)


# Example usage
if __name__ == "__main__":
    # Create different providers
    # llm = LLMFactory.create('openai')
    # llm = LLMFactory.create('anthropic')
    # llm = LLMFactory.create('gemini')
    
    # Use the provider
    # response = llm.generate("What is the capital of France?")
    # print(response.content)
    pass