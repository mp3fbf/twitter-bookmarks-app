"""
Modular LLM Provider Interface
Supports OpenAI, Anthropic, and Google Gemini
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import os
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized response from LLM providers"""
    content: str
    model: str
    usage: Optional[Dict] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate a response from the LLM"""
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
            # Check if this is an o1/o3 model which uses different parameters
            if self.model.startswith('o1') or self.model.startswith('o3'):
                print(f"[dim]Note: {self.model} models can take 30-60 seconds to respond...[/dim]")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=1000,
                    timeout=120.0  # 2 minute timeout for o3
                )
            else:
                # Regular GPT models
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000,
                    timeout=60.0  # 1 minute timeout
                )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                usage=response.usage.model_dump() if response.usage else None
            )
        except Exception as e:
            # Re-raise with more context if needed
            if "Invalid API key" in str(e) or "Incorrect API key" in str(e):
                raise ValueError(f"OpenAI API key error: {str(e)}. Please check your .env file.")
            raise
    
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
            'openai': (OpenAIProvider, 'o3'),
            'anthropic': (AnthropicProvider, 'claude-3-haiku-20240307'),
            'gemini': (GeminiProvider, 'gemini-1.5-flash')
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