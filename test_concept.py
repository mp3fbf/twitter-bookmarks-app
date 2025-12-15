"""
Teste de Conceito - Um tweet de cada tipo para validar a análise

Tipos:
1. Tweet puro (só texto)
2. Tweet com imagem
3. Tweet com link
4. Tweet com vídeo
5. Tweet com imagem + link (combinação)

Modelos:
- Vídeo: Gemini 3 Pro Preview (melhor para vídeo)
- Imagem: Todos os 4 modelos
- Texto: GPT-5.2 (principal)
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from media_linker import MediaLinker
from tweet_extractor import get_quoted_tweet_info, format_quoted_tweet_for_prompt, get_thread_info
from thread_fetcher import ThreadFetcher

load_dotenv()
console = Console()

MEDIA_FOLDER = "~/Downloads/twillot-media-files-by-date/"
BOOKMARKS_FILE = "~/Downloads/twillot-bookmark.json"

# Known good thread on ThreadReaderApp (tested and works)
KNOWN_THREAD_TWEET_ID = "1999874571806355477"  # @minchoi - 10 best practices


def get_tweet_url(bookmark: Dict) -> str:
    """Build the original tweet URL"""
    screen_name = bookmark.get('screen_name', '')
    tweet_id = bookmark.get('tweet_id', '')
    if screen_name and tweet_id:
        return f"https://x.com/{screen_name}/status/{tweet_id}"
    return ""


def load_bookmarks() -> List[Dict]:
    """Load bookmarks from JSON"""
    with open(Path(BOOKMARKS_FILE).expanduser(), 'r', encoding='utf-8') as f:
        return json.load(f)


def find_test_samples(bookmarks: List[Dict]) -> Dict[str, Dict]:
    """Find one bookmark of each type"""
    samples = {
        'text_only': None,
        'quote_tweet': None,
        'thread': None,  # NEW: threads
        'with_image': None,
        'with_link': None,
        'with_video': None,
        'image_and_link': None,
    }

    # First pass: look for the known-good thread (priority)
    for b in bookmarks:
        tweet_id = str(b.get('tweet_id', ''))
        if tweet_id == KNOWN_THREAD_TWEET_ID:
            samples['thread'] = b
            break

    # Second pass: find other samples
    for b in bookmarks:
        text = b.get('full_text', b.get('text', ''))
        has_image = b.get('has_image', False)
        has_video = b.get('has_video', False)
        has_link = 'http' in text and not has_image and not has_video
        local_media = b.get('local_media_paths', [])
        quoted = get_quoted_tweet_info(b)
        thread_info = get_thread_info(b)

        # Priorizar bookmarks com mídia local
        if has_video and local_media and samples['with_video'] is None:
            samples['with_video'] = b
        elif has_image and has_link and local_media and samples['image_and_link'] is None:
            samples['image_and_link'] = b
        elif has_image and not has_link and local_media and samples['with_image'] is None:
            samples['with_image'] = b
        elif has_link and not has_image and not has_video and samples['with_link'] is None:
            samples['with_link'] = b
        elif quoted and samples['quote_tweet'] is None:
            samples['quote_tweet'] = b
        # Thread check MUST be BEFORE text_only (integrated in elif chain)
        elif thread_info and thread_info.needs_fetch and samples['thread'] is None:
            samples['thread'] = b
        elif not has_image and not has_video and not has_link and not quoted and not thread_info and samples['text_only'] is None:
            samples['text_only'] = b

        # Se já temos todos, para
        if all(v is not None for v in samples.values()):
            break

    return samples


def test_text_only(bookmark: Dict) -> Dict:
    """Testar tweet só com texto - GPT-5.2"""
    from llm_providers import LLMFactory

    console.print("\n[bold cyan]═══ TESTE: Tweet Só Texto ═══[/bold cyan]")
    console.print(f"Tweet: {bookmark.get('full_text', '')[:100]}...")
    console.print(f"Autor: @{bookmark.get('screen_name')}")

    llm = LLMFactory.create('openai', model='gpt-5.2')

    prompt = f"""Analise este tweet e extraia o valor principal.

Tweet: {bookmark.get('full_text', '')}
Autor: @{bookmark.get('screen_name', 'unknown')}
Likes: {bookmark.get('favorite_count', 0)}

Por que alguém salvaria este tweet? Qual é a informação ou insight principal?
Seja específico e extraia valor REAL."""

    response = llm.generate(prompt)

    console.print(Panel(response.content, title="[green]GPT-5.2 Response[/green]", border_style="green"))

    return {
        'type': 'text_only',
        'model': 'gpt-5.2',
        'tweet': bookmark.get('full_text', '')[:100],
        'tweet_url': get_tweet_url(bookmark),
        'response': response.content
    }


def test_quote_tweet(bookmark: Dict) -> Dict:
    """Testar quote tweet - Extrai conteúdo do tweet citado + GPT-5.2"""
    from llm_providers import LLMFactory
    from content_fetcher import ContentFetcher
    from tweet_extractor import get_tweet_url_from_quoted

    console.print("\n[bold cyan]═══ TESTE: Quote Tweet ═══[/bold cyan]")
    console.print(f"Tweet principal: {bookmark.get('full_text', '')[:100]}...")
    console.print(f"Autor: @{bookmark.get('screen_name')}")

    # Extrair informações do quote tweet
    quoted = get_quoted_tweet_info(bookmark)
    if not quoted:
        console.print("[red]Não é um quote tweet![/red]")
        return {'type': 'quote_tweet', 'tweet_url': get_tweet_url(bookmark), 'error': 'Não é quote tweet'}

    console.print(f"\n[yellow]── Quote Tweet Detectado ──[/yellow]")
    console.print(f"Autor citado: @{quoted.author}")
    console.print(f"Texto citado: {quoted.text[:150]}...")

    # Buscar conteúdo de links no quote tweet
    fetcher = ContentFetcher()
    link_content = ""
    if quoted.urls:
        console.print(f"[dim]URLs no quote tweet: {quoted.urls}[/dim]")
        for url in quoted.urls[:2]:
            console.print(f"[yellow]Buscando conteúdo de {url}...[/yellow]")
            content = fetcher.fetch_content(url)
            link_content += f"\n\n--- Conteúdo de {url} ---\n"
            link_content += fetcher.get_content_summary(content)

    # Montar informação de mídia do quote tweet
    media_info = ""
    if quoted.image_urls:
        media_info += f"\nImagens no quote tweet: {len(quoted.image_urls)}"
        for i, img_url in enumerate(quoted.image_urls, 1):
            media_info += f"\n  Imagem {i}: {img_url}"
    if quoted.video_urls:
        media_info += f"\nVídeos no quote tweet: {len(quoted.video_urls)}"
        for i, thumb in enumerate(quoted.video_thumbnails, 1):
            media_info += f"\n  Thumbnail {i}: {thumb}"

    # Montar prompt completo
    quoted_tweet_url = get_tweet_url_from_quoted(quoted)

    prompt = f"""Este é um QUOTE TWEET - o autor está comentando/reagindo a outro tweet.

═══ TWEET PRINCIPAL ═══
Autor: @{bookmark.get('screen_name', 'unknown')}
Texto: {bookmark.get('full_text', '')}

═══ TWEET CITADO ═══
Autor: @{quoted.author}
URL: {quoted_tweet_url}
Texto: {quoted.text}
{media_info if media_info else ''}

{f'═══ CONTEÚDO DOS LINKS ═══{link_content}' if link_content else ''}

TAREFA: Analise este quote tweet considerando AMBOS os tweets.

1. O que está acontecendo no tweet citado?
2. Por que o autor principal está citando isso?
3. Qual é o contexto ou evento importante?
4. Qual é o VALOR REAL que o usuário queria guardar?

NÃO "encha linguiça" - extraia a informação concreta e útil."""

    llm = LLMFactory.create('openai', model='gpt-5.2')

    # Se o quote tweet tem imagens, usar visão com URLs (GPT suporta URLs)
    if quoted.image_urls:
        console.print(f"[yellow]Usando visão para analisar {len(quoted.image_urls)} imagem(ns) do quote tweet...[/yellow]")
        try:
            response = llm.generate_with_vision(prompt, quoted.image_urls)
        except Exception as e:
            console.print(f"[yellow]Visão falhou ({e}), usando texto apenas[/yellow]")
            response = llm.generate(prompt)
    else:
        response = llm.generate(prompt)

    console.print(Panel(response.content, title="[green]GPT-5.2 (Quote Tweet)[/green]", border_style="green"))

    return {
        'type': 'quote_tweet',
        'model': 'gpt-5.2',
        'tweet': bookmark.get('full_text', '')[:100],
        'tweet_url': get_tweet_url(bookmark),
        'quoted_author': quoted.author,
        'quoted_text': quoted.text[:100],
        'quoted_tweet_url': quoted_tweet_url,
        'quoted_has_images': len(quoted.image_urls),
        'quoted_has_videos': len(quoted.video_urls),
        'quoted_urls_fetched': len(quoted.urls),
        'link_content_fetched': bool(link_content),
        'response': response.content
    }


def test_with_image(bookmark: Dict) -> List[Dict]:
    """Testar tweet com imagem - Todos os 4 modelos"""
    from llm_providers import LLMFactory, GeminiProvider

    console.print("\n[bold cyan]═══ TESTE: Tweet com Imagem ═══[/bold cyan]")
    console.print(f"Tweet: {bookmark.get('full_text', '')[:100]}...")
    console.print(f"Autor: @{bookmark.get('screen_name')}")

    local_images = [p for p, t in zip(
        bookmark.get('local_media_paths', []),
        bookmark.get('local_media_types', [])
    ) if t == 'image']

    console.print(f"[dim]Imagens locais: {len(local_images)}[/dim]")
    for img in local_images:
        console.print(f"  [dim]{img}[/dim]")

    if not local_images:
        console.print("[red]Sem imagens locais![/red]")
        return []

    prompt = f"""Analise este tweet e a imagem anexada.

Tweet: {bookmark.get('full_text', '')}
Autor: @{bookmark.get('screen_name', 'unknown')}

IMPORTANTE: Olhe a imagem com atenção. Extraia TODA informação útil visível:
- Se é screenshot de texto/código, transcreva
- Se é diagrama/gráfico, descreva o que mostra
- Se é tutorial/guia, extraia os passos
- Se contém lista, extraia a lista completa

Por que o usuário salvou isso? Extraia o VALOR REAL."""

    results = []
    models = [
        ('openai', 'gpt-5.2', 'GPT-5.2'),
        ('anthropic', 'claude-opus-4-5-20251101', 'Claude Opus 4.5'),
        ('gemini', 'gemini-2.5-flash', 'Gemini 2.5 Flash'),
        ('gemini_pro', 'gemini-3-pro-preview', 'Gemini 3 Pro Preview'),
    ]

    for provider_key, model, display_name in models:
        console.print(f"\n[yellow]Testando {display_name}...[/yellow]")
        try:
            if provider_key == 'gemini_pro':
                llm = GeminiProvider(model='gemini-3-pro-preview')
            else:
                llm = LLMFactory.create(provider_key, model=model)

            response = llm.generate_with_vision(prompt, local_images)

            console.print(Panel(
                response.content[:1000] + ("..." if len(response.content) > 1000 else ""),
                title=f"[green]{display_name}[/green]",
                border_style="green"
            ))

            results.append({
                'type': 'with_image',
                'model': display_name,
                'tweet': bookmark.get('full_text', '')[:100],
                'tweet_url': get_tweet_url(bookmark),
                'images_processed': response.images_processed,
                'response': response.content
            })
        except Exception as e:
            console.print(f"[red]Erro {display_name}: {e}[/red]")
            results.append({
                'type': 'with_image',
                'model': display_name,
                'tweet_url': get_tweet_url(bookmark),
                'error': str(e)
            })

    return results


def test_with_link(bookmark: Dict) -> Dict:
    """Testar tweet com link - Buscar conteúdo + GPT-5.2"""
    from llm_providers import LLMFactory
    from content_fetcher import ContentFetcher

    console.print("\n[bold cyan]═══ TESTE: Tweet com Link ═══[/bold cyan]")
    console.print(f"Tweet: {bookmark.get('full_text', '')[:150]}...")
    console.print(f"Autor: @{bookmark.get('screen_name')}")

    # Buscar conteúdo do link
    fetcher = ContentFetcher()
    tweet_text = bookmark.get('full_text', '')
    urls = fetcher.extract_urls(tweet_text)

    link_content = ""
    if urls:
        console.print(f"[dim]URLs encontradas: {urls}[/dim]")
        for url in urls[:2]:  # Max 2 URLs
            console.print(f"[yellow]Buscando conteúdo de {url}...[/yellow]")
            content = fetcher.fetch_content(url)
            link_content += f"\n\n--- Conteúdo de {url} ---\n"
            link_content += fetcher.get_content_summary(content)

    llm = LLMFactory.create('openai', model='gpt-5.2')

    prompt = f"""Analise este tweet e o conteúdo do link.

Tweet: {tweet_text}
Autor: @{bookmark.get('screen_name', 'unknown')}

{link_content if link_content else '[Nenhum conteúdo de link disponível]'}

TAREFA: Extraia o VALOR PRINCIPAL que o usuário queria guardar.
- Se é artigo, resuma os pontos principais
- Se é lista/ranking, extraia a lista completa
- Se é tutorial, extraia os passos
- Se é ferramenta, explique o que é e como usar

Seja ESPECÍFICO. Não descreva o tweet, extraia a INFORMAÇÃO."""

    response = llm.generate(prompt)

    console.print(Panel(response.content, title="[green]GPT-5.2 + Link Content[/green]", border_style="green"))

    return {
        'type': 'with_link',
        'model': 'gpt-5.2',
        'tweet': tweet_text[:100],
        'tweet_url': get_tweet_url(bookmark),
        'urls_found': urls,
        'link_content_fetched': bool(link_content),
        'response': response.content
    }


def test_with_video(bookmark: Dict) -> Dict:
    """Testar tweet com vídeo - Gemini 3 Pro Preview"""
    from llm_providers import GeminiProvider

    console.print("\n[bold cyan]═══ TESTE: Tweet com Vídeo ═══[/bold cyan]")
    console.print(f"Tweet: {bookmark.get('full_text', '')[:100]}...")
    console.print(f"Autor: @{bookmark.get('screen_name')}")

    local_videos = [p for p, t in zip(
        bookmark.get('local_media_paths', []),
        bookmark.get('local_media_types', [])
    ) if t == 'video']

    console.print(f"[dim]Vídeos locais: {len(local_videos)}[/dim]")
    for vid in local_videos:
        size_mb = Path(vid).stat().st_size / (1024 * 1024)
        console.print(f"  [dim]{vid} ({size_mb:.1f}MB)[/dim]")

    if not local_videos:
        console.print("[red]Sem vídeos locais![/red]")
        return {'type': 'with_video', 'tweet_url': get_tweet_url(bookmark), 'error': 'Sem vídeos locais'}

    video_path = local_videos[0]

    prompt = f"""Analise este tweet e o vídeo anexado.

Tweet: {bookmark.get('full_text', '')}
Autor: @{bookmark.get('screen_name', 'unknown')}

TAREFA: Assista o vídeo e extraia TODA informação útil:
1. O que acontece no vídeo? (resumo)
2. Quais são os momentos/pontos principais?
3. Se é tutorial/demo, extraia os passos
4. Qual é a mensagem principal ou takeaway?

Por que o usuário salvou este vídeo? Extraia o VALOR REAL."""

    console.print(f"[yellow]Enviando vídeo para Gemini 3 Pro Preview...[/yellow]")

    try:
        llm = GeminiProvider(model='gemini-3-pro-preview')
        response = llm.generate_with_video(prompt, video_path)

        console.print(Panel(response.content, title="[green]Gemini 3 Pro Preview (Video)[/green]", border_style="green"))

        return {
            'type': 'with_video',
            'model': 'gemini-3-pro-preview',
            'tweet': bookmark.get('full_text', '')[:100],
            'tweet_url': get_tweet_url(bookmark),
            'video_path': video_path,
            'response': response.content
        }
    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
        return {
            'type': 'with_video',
            'model': 'gemini-3-pro-preview',
            'tweet_url': get_tweet_url(bookmark),
            'error': str(e)
        }


def test_image_and_link(bookmark: Dict) -> Dict:
    """Testar tweet com imagem + link - GPT-5.2 com tudo"""
    from llm_providers import LLMFactory
    from content_fetcher import ContentFetcher

    console.print("\n[bold cyan]═══ TESTE: Tweet com Imagem + Link ═══[/bold cyan]")
    console.print(f"Tweet: {bookmark.get('full_text', '')[:100]}...")
    console.print(f"Autor: @{bookmark.get('screen_name')}")

    # Imagens locais
    local_images = [p for p, t in zip(
        bookmark.get('local_media_paths', []),
        bookmark.get('local_media_types', [])
    ) if t == 'image']

    console.print(f"[dim]Imagens locais: {len(local_images)}[/dim]")

    # Links
    fetcher = ContentFetcher()
    tweet_text = bookmark.get('full_text', '')
    urls = fetcher.extract_urls(tweet_text)

    link_content = ""
    if urls:
        console.print(f"[dim]URLs encontradas: {urls}[/dim]")
        for url in urls[:1]:
            console.print(f"[yellow]Buscando conteúdo de {url}...[/yellow]")
            content = fetcher.fetch_content(url)
            link_content += fetcher.get_content_summary(content)

    prompt = f"""Analise este tweet, a imagem e o conteúdo do link.

Tweet: {tweet_text}
Autor: @{bookmark.get('screen_name', 'unknown')}

Conteúdo do Link:
{link_content if link_content else '[Nenhum conteúdo disponível]'}

TAREFA: Combine a análise da imagem com o conteúdo do link.
Extraia TODA informação útil. Por que o usuário salvou isso?"""

    llm = LLMFactory.create('openai', model='gpt-5.2')

    if local_images:
        response = llm.generate_with_vision(prompt, local_images)
    else:
        response = llm.generate(prompt)

    console.print(Panel(response.content, title="[green]GPT-5.2 (Imagem + Link)[/green]", border_style="green"))

    return {
        'type': 'image_and_link',
        'model': 'gpt-5.2',
        'tweet': tweet_text[:100],
        'tweet_url': get_tweet_url(bookmark),
        'images_used': len(local_images),
        'link_fetched': bool(link_content),
        'response': response.content
    }


def test_thread(bookmark: Dict) -> Dict:
    """Testar thread - Busca via ThreadReaderApp + GPT-5.2"""
    from llm_providers import LLMFactory

    console.print("\n[bold cyan]═══ TESTE: Thread ═══[/bold cyan]")
    console.print(f"Tweet: {bookmark.get('full_text', '')[:100]}...")
    console.print(f"Autor: @{bookmark.get('screen_name')}")

    # Verificar se é thread
    thread_info = get_thread_info(bookmark)
    if not thread_info:
        console.print("[red]Não é início de thread![/red]")
        return {'type': 'thread', 'tweet_url': get_tweet_url(bookmark), 'error': 'Não é thread'}

    console.print(f"[yellow]Este é o primeiro tweet de uma thread[/yellow]")
    console.print(f"[yellow]Thread completa capturada pelo Twillot: {thread_info.has_full_thread}[/yellow]")

    # Buscar thread via ThreadReaderApp
    fetcher = ThreadFetcher()
    tweet_id = bookmark.get('tweet_id', '')
    author = bookmark.get('screen_name', '')

    console.print(f"[yellow]Buscando thread via ThreadReaderApp...[/yellow]")
    thread_result = fetcher.fetch_thread(tweet_id, author)

    if not thread_result.success:
        console.print(f"[red]Falha ao buscar thread: {thread_result.error}[/red]")
        return {
            'type': 'thread',
            'model': 'gpt-5.2',
            'tweet_url': get_tweet_url(bookmark),
            'error': thread_result.error
        }

    console.print(f"[green]Encontrados {len(thread_result.tweets)} tweets na thread![/green]")
    console.print(f"[dim]Source: {thread_result.source}, Cached: {thread_result.cached}[/dim]")

    # Formatar thread para o prompt
    thread_content = fetcher.format_thread_for_prompt(thread_result)

    # Mostrar preview da thread
    console.print("\n[dim]Preview dos primeiros tweets:[/dim]")
    for tweet in thread_result.tweets[:5]:
        console.print(f"  [{tweet.position}] {tweet.text[:80]}...")

    # Montar prompt
    prompt = f"""Analise esta THREAD completa do Twitter.

{thread_content}

TAREFA: Esta thread foi salva pelo usuário. Extraia TODO o valor:

1. Qual é o tema/assunto principal?
2. Se é uma lista/guia, extraia TODOS os itens
3. Se é tutorial, extraia TODOS os passos
4. Se é análise/opinião, resuma os pontos principais
5. Qual é o insight/valor principal para o usuário?

Seja COMPLETO. A thread inteira está acima - extraia TUDO que é útil."""

    console.print(f"[dim]Prompt length: {len(prompt)} chars[/dim]")

    llm = LLMFactory.create('openai', model='gpt-5.2')
    try:
        response = llm.generate(prompt)
        console.print(f"[dim]Response content length: {len(response.content) if response.content else 0}[/dim]")
        if not response.content:
            console.print("[red]WARNING: Response content is empty![/red]")
    except Exception as e:
        console.print(f"[red]LLM Error: {e}[/red]")
        return {'type': 'thread', 'tweet_url': get_tweet_url(bookmark), 'error': str(e)}

    console.print(Panel(response.content, title="[green]GPT-5.2 (Thread Analysis)[/green]", border_style="green"))

    return {
        'type': 'thread',
        'model': 'gpt-5.2',
        'tweet': bookmark.get('full_text', '')[:100],
        'tweet_url': get_tweet_url(bookmark),
        'thread_tweets_found': len(thread_result.tweets),
        'thread_source': thread_result.source,
        'thread_cached': thread_result.cached,
        'response': response.content
    }


def save_results(results: List[Dict]):
    """Salvar resultados em markdown"""
    output_file = f"concept_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Teste de Conceito - Resultados\n\n")
        f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for r in results:
            f.write(f"## {r.get('type', 'unknown').replace('_', ' ').title()}\n\n")
            f.write(f"**Modelo:** {r.get('model', 'unknown')}\n\n")

            if 'error' in r:
                f.write(f"**Erro:** {r['error']}\n\n")
            else:
                # Add original tweet link
                if r.get('tweet_url'):
                    f.write(f"**Link Original:** [{r.get('tweet_url')}]({r.get('tweet_url')})\n\n")
                f.write(f"**Tweet:** {r.get('tweet', '')}\n\n")
                f.write(f"**Resposta:**\n\n{r.get('response', 'Sem resposta')}\n\n")

            f.write("---\n\n")

    console.print(f"\n[green]Resultados salvos em {output_file}[/green]")


def main():
    console.print(Panel.fit(
        "[bold cyan]Teste de Conceito - Validação[/bold cyan]\n"
        "Um tweet de cada tipo para provar o conceito",
        border_style="cyan"
    ))

    # Carregar bookmarks
    console.print("\n[cyan]Carregando bookmarks...[/cyan]")
    bookmarks = load_bookmarks()
    console.print(f"[green]Carregados {len(bookmarks)} bookmarks[/green]")

    # Indexar mídia local
    console.print(f"\n[cyan]Indexando mídia local em {MEDIA_FOLDER}...[/cyan]")
    try:
        media_linker = MediaLinker(MEDIA_FOLDER)
        stats = media_linker.get_stats()
        console.print(f"[green]Encontrados {stats['total_files']} arquivos de mídia[/green]")
        console.print(f"  Imagens: {stats['images']}, Vídeos: {stats['videos']}, GIFs: {stats['gifs']}")

        bookmarks = media_linker.enrich_bookmarks(bookmarks)
        with_media = sum(1 for b in bookmarks if b.get('local_media_paths'))
        console.print(f"[green]{with_media} bookmarks com mídia local[/green]")
    except FileNotFoundError as e:
        console.print(f"[red]Pasta de mídia não encontrada: {e}[/red]")
        return

    # Encontrar amostras de teste
    console.print("\n[cyan]Selecionando amostras de teste...[/cyan]")
    samples = find_test_samples(bookmarks)

    table = Table(title="Amostras Selecionadas")
    table.add_column("Tipo")
    table.add_column("Encontrado")
    table.add_column("Autor")
    table.add_column("Mídia Local")

    for tipo, b in samples.items():
        if b:
            local_count = len(b.get('local_media_paths', []))
            table.add_row(
                tipo.replace('_', ' ').title(),
                "✓",
                f"@{b.get('screen_name', '?')}",
                str(local_count) if local_count else "-"
            )
        else:
            table.add_row(tipo.replace('_', ' ').title(), "✗", "-", "-")

    console.print(table)

    # Executar testes
    results = []

    # 1. Tweet só texto
    if samples['text_only']:
        result = test_text_only(samples['text_only'])
        results.append(result)
    else:
        console.print("[yellow]Pulando teste de texto puro - nenhum encontrado[/yellow]")

    # 2. Quote Tweet (extrai conteúdo citado)
    if samples['quote_tweet']:
        result = test_quote_tweet(samples['quote_tweet'])
        results.append(result)
    else:
        console.print("[yellow]Pulando teste de quote tweet - nenhum encontrado[/yellow]")

    # 3. Thread (busca via ThreadReaderApp)
    if samples['thread']:
        result = test_thread(samples['thread'])
        results.append(result)
    else:
        console.print("[yellow]Pulando teste de thread - nenhum encontrado[/yellow]")

    # 4. Tweet com imagem (todos os modelos)
    if samples['with_image']:
        image_results = test_with_image(samples['with_image'])
        results.extend(image_results)
    else:
        console.print("[yellow]Pulando teste de imagem - nenhum encontrado[/yellow]")

    # 5. Tweet com link
    if samples['with_link']:
        result = test_with_link(samples['with_link'])
        results.append(result)
    else:
        console.print("[yellow]Pulando teste de link - nenhum encontrado[/yellow]")

    # 6. Tweet com vídeo (Gemini 3 Pro)
    if samples['with_video']:
        result = test_with_video(samples['with_video'])
        results.append(result)
    else:
        console.print("[yellow]Pulando teste de vídeo - nenhum encontrado[/yellow]")

    # 7. Tweet com imagem + link
    if samples['image_and_link']:
        result = test_image_and_link(samples['image_and_link'])
        results.append(result)
    else:
        console.print("[yellow]Pulando teste de imagem+link - nenhum encontrado[/yellow]")

    # Salvar resultados
    save_results(results)

    console.print("\n[bold green]Teste de conceito completo![/bold green]")


if __name__ == "__main__":
    main()
