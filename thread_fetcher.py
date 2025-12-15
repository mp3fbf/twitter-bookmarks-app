"""
Thread Fetcher - Busca threads completas via ThreadReaderApp

Quando o Twillot salva um bookmark como `post_type: post` em vez de `thread`,
os tweets seguintes da thread não são capturados. Este módulo busca
esses tweets via ThreadReaderApp.

ThreadReaderApp é um serviço gratuito que "desenrola" threads do Twitter.
Não requer autenticação e retorna a thread completa.
"""

import json
import sqlite3
import requests
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from bs4 import BeautifulSoup


@dataclass
class ThreadTweet:
    """Um tweet individual dentro de uma thread"""
    tweet_id: str
    text: str
    author: str
    position: int = 0  # Posição na thread (1, 2, 3...)
    created_at: Optional[str] = None
    media_urls: List[str] = field(default_factory=list)


@dataclass
class ThreadResult:
    """Resultado da busca de uma thread"""
    success: bool
    tweets: List[ThreadTweet] = field(default_factory=list)
    error: Optional[str] = None
    source: str = "threadreaderapp"
    cached: bool = False


class ThreadFetcher:
    """
    Busca threads completas via ThreadReaderApp

    ThreadReaderApp é um serviço que "desenrola" threads do Twitter,
    facilitando a leitura e extração de conteúdo.
    """

    THREADREADER_URL = "https://threadreaderapp.com/thread/{tweet_id}.html"

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    def __init__(self, cache_path: str = "thread_cache.db"):
        """
        Inicializa o ThreadFetcher

        Args:
            cache_path: Caminho para o banco SQLite de cache
        """
        self.cache_path = Path(cache_path)
        self._init_cache()
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def _init_cache(self):
        """Inicializa o banco de dados SQLite para cache"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        # Tabela de threads cacheadas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS thread_cache (
                conversation_id TEXT PRIMARY KEY,
                tweets_json TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                source TEXT DEFAULT 'threadreaderapp'
            )
        """)

        # Tabela de erros (para não re-tentar tweets que falharam)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fetch_errors (
                tweet_id TEXT PRIMARY KEY,
                error TEXT NOT NULL,
                attempted_at TEXT NOT NULL,
                retry_after TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _get_cached_thread(self, conversation_id: str) -> Optional[ThreadResult]:
        """Busca thread no cache"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT tweets_json, source FROM thread_cache WHERE conversation_id = ?",
            (conversation_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            tweets_data = json.loads(row[0])
            tweets = [ThreadTweet(**t) for t in tweets_data]
            return ThreadResult(
                success=True,
                tweets=tweets,
                source=row[1],
                cached=True
            )
        return None

    def _cache_thread(self, conversation_id: str, tweets: List[ThreadTweet], source: str = "threadreaderapp"):
        """Salva thread no cache"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        tweets_json = json.dumps([
            {
                'tweet_id': t.tweet_id,
                'text': t.text,
                'author': t.author,
                'position': t.position,
                'created_at': t.created_at,
                'media_urls': t.media_urls
            }
            for t in tweets
        ])

        cursor.execute("""
            INSERT OR REPLACE INTO thread_cache (conversation_id, tweets_json, fetched_at, source)
            VALUES (?, ?, ?, ?)
        """, (conversation_id, tweets_json, datetime.now().isoformat(), source))

        conn.commit()
        conn.close()

    def _should_retry(self, tweet_id: str) -> bool:
        """Verifica se devemos tentar buscar novamente (evita spam de requests)"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT retry_after FROM fetch_errors WHERE tweet_id = ?",
            (tweet_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return True

        retry_after = row[0]
        if retry_after:
            try:
                retry_time = datetime.fromisoformat(retry_after)
                return datetime.now() > retry_time
            except ValueError:
                return True

        return False

    def _record_error(self, tweet_id: str, error: str, retry_hours: int = 24):
        """Registra erro para evitar re-tentativas frequentes"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        retry_after = (datetime.now() + timedelta(hours=retry_hours)).isoformat()

        cursor.execute("""
            INSERT OR REPLACE INTO fetch_errors (tweet_id, error, attempted_at, retry_after)
            VALUES (?, ?, ?, ?)
        """, (tweet_id, error, datetime.now().isoformat(), retry_after))

        conn.commit()
        conn.close()

    def _fetch_via_threadreader(self, tweet_id: str, author: str) -> List[ThreadTweet]:
        """
        Busca thread completa via ThreadReaderApp

        Args:
            tweet_id: ID do primeiro tweet da thread
            author: Screen name do autor

        Returns:
            Lista de ThreadTweet ou lista vazia se falhar
        """
        url = self.THREADREADER_URL.format(tweet_id=tweet_id)

        try:
            response = self.session.get(url, timeout=15)

            if response.status_code == 404:
                return []  # Thread não existe no ThreadReaderApp

            if response.status_code != 200:
                print(f"ThreadReaderApp returned {response.status_code}")
                return []

            # Parsear HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Buscar tweets (seletor: div.content-tweet)
            tweet_divs = soup.find_all('div', class_='content-tweet')

            if not tweet_divs:
                return []

            tweets = []
            for i, tweet_div in enumerate(tweet_divs, 1):
                # Extrair texto do tweet (está direto no div)
                text = tweet_div.get_text(strip=True)
                # Limpar whitespace excessivo
                text = ' '.join(text.split())

                # Extrair imagens se houver
                media_urls = []
                images = tweet_div.find_all('img')
                for img in images:
                    src = img.get('src', '')
                    if 'twimg.com' in src or 'pbs.twimg.com' in src:
                        media_urls.append(src)

                tweets.append(ThreadTweet(
                    tweet_id=tweet_id if i == 1 else f"{tweet_id}_{i}",
                    text=text,
                    author=author,
                    position=i,
                    media_urls=media_urls
                ))

            return tweets

        except requests.RequestException as e:
            print(f"Request error: {e}")
            return []
        except Exception as e:
            print(f"Parse error: {e}")
            return []

    def fetch_thread(self, first_tweet_id: str, author: str, max_tweets: int = 50) -> ThreadResult:
        """
        Busca uma thread completa a partir do primeiro tweet

        Args:
            first_tweet_id: ID do primeiro tweet da thread
            author: Screen name do autor
            max_tweets: Máximo de tweets a retornar

        Returns:
            ThreadResult com tweets encontrados
        """
        # Verificar cache primeiro
        cached = self._get_cached_thread(first_tweet_id)
        if cached:
            return cached

        # Verificar se já tentamos e falhou recentemente
        if not self._should_retry(first_tweet_id):
            return ThreadResult(
                success=False,
                error="Tentativa anterior falhou, aguardando retry"
            )

        # Buscar via ThreadReaderApp
        tweets = self._fetch_via_threadreader(first_tweet_id, author)

        if tweets:
            # Limitar quantidade
            tweets = tweets[:max_tweets]

            # Cachear resultado
            self._cache_thread(first_tweet_id, tweets, "threadreaderapp")

            return ThreadResult(
                success=True,
                tweets=tweets,
                source="threadreaderapp"
            )
        else:
            self._record_error(first_tweet_id, "Thread não encontrada no ThreadReaderApp")
            return ThreadResult(
                success=False,
                error="Thread não encontrada no ThreadReaderApp"
            )

    def format_thread_for_prompt(self, result: ThreadResult) -> str:
        """
        Formata thread para incluir em um prompt de LLM

        Args:
            result: ThreadResult com tweets

        Returns:
            String formatada para o prompt
        """
        if not result.success or not result.tweets:
            return ""

        lines = ["\n═══ THREAD COMPLETA ═══"]
        for tweet in result.tweets:
            lines.append(f"\n[{tweet.position}/{len(result.tweets)}] @{tweet.author}:")
            lines.append(tweet.text)
        lines.append("\n═══ FIM DA THREAD ═══")

        return "\n".join(lines)

    def get_cache_stats(self) -> Dict:
        """Retorna estatísticas do cache"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM thread_cache")
        cached_threads = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM fetch_errors")
        errors = cursor.fetchone()[0]

        conn.close()

        return {
            'cached_threads': cached_threads,
            'recorded_errors': errors,
            'cache_path': str(self.cache_path)
        }

    def clear_cache(self):
        """Limpa todo o cache"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM thread_cache")
        cursor.execute("DELETE FROM fetch_errors")
        conn.commit()
        conn.close()


# Teste básico
if __name__ == "__main__":
    fetcher = ThreadFetcher()

    # Tweet do @minchoi sobre Anthropic Prompt Guide
    test_tweet_id = "1999874571806355477"
    test_author = "minchoi"

    print(f"Buscando thread do tweet {test_tweet_id}...")
    result = fetcher.fetch_thread(test_tweet_id, test_author)

    if result.success:
        print(f"\nEncontrados {len(result.tweets)} tweet(s) (source: {result.source}, cached: {result.cached}):")
        for tweet in result.tweets[:12]:  # Mostrar primeiros 12
            print(f"\n[{tweet.position}] @{tweet.author}:")
            print(f"    {tweet.text[:150]}...")
            if tweet.media_urls:
                print(f"    Mídia: {len(tweet.media_urls)} arquivo(s)")
    else:
        print(f"Erro: {result.error}")

    # Stats
    print(f"\nCache stats: {fetcher.get_cache_stats()}")
