# Plano de Implementação - Twitter Bookmarks App

**Data:** 2025-12-15
**Status:** Em andamento

---

## Fase 1: Remover API do Twitter (CONCLUÍDO)

### Decisão
Removemos a API do Twitter em favor do Twillot porque:

| Aspecto | API do Twitter | Twillot |
|---------|---------------|---------|
| Setup | Complexo (Developer Portal, OAuth) | Simples (extensão Chrome) |
| Rate Limits | 1 req/15 min | Nenhum |
| Limite | 800 bookmarks | Sem limite |
| Dados | Truncados | Completos (mídia, threads) |

### Arquivos Modificados
- `twitter_auth.py` - DELETADO
- `bookmarks_fetcher.py` - Removido dependências de auth
- `main.py` - Menu simplificado (13→10 opções)
- `.env.example` - Só chaves LLM
- `SETUP_GUIDE.md` - Focado em Twillot
- `README.md` - Documentação atualizada
- `CLAUDE.md` - Atualizado
- `requirements.txt` - Removido tweepy, flask

---

## Fase 2: Análise Multimodal (CONCLUÍDO)

### Contexto
Export do Twillot disponível:
- **JSON**: `~/Downloads/twillot-bookmark.json` (786 bookmarks)
  - 342 com imagem
  - 243 com vídeo
  - 2 com GIF
- **Mídias**: `~/Downloads/twillot-media-files-by-date/`
  - Formato: `{username}-{tweet_id}-{media_id}.{ext}`
  - ~230 pastas por data

### Objetivo
Processar bookmarks com análise de imagens para entender conteúdo completo.

### Arquivos a Criar/Modificar

#### 1. CRIAR: `media_linker.py`
```python
class MediaLinker:
    """Vincula JSON do Twillot com arquivos de mídia locais"""

    def __init__(self, media_folder: str):
        self.media_folder = media_folder
        self.media_index = {}  # tweet_id -> [file_paths]

    def build_index(self):
        """Percorre pastas, extrai tweet_id do nome, mapeia arquivos"""

    def get_media_for_tweet(self, tweet_id: str) -> List[str]:
        """Retorna caminhos dos arquivos de mídia para um tweet"""

    def enrich_bookmarks(self, bookmarks: List[Dict]) -> List[Dict]:
        """Adiciona campo 'local_media_paths' a cada bookmark"""
```

#### 2. MODIFICAR: `llm_providers.py`
Adicionar método `generate_with_vision()` em todos os providers:

```python
class LLMProvider(ABC):
    @abstractmethod
    def generate_with_vision(
        self,
        prompt: str,
        images: List[str],  # caminhos locais ou base64
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        pass

# Implementar para:
# - OpenAIProvider (GPT-5.2)
# - AnthropicProvider (Claude Opus 4.5)
# - GeminiProvider (Gemini 3 Pro / 2.5 Flash)
```

#### 3. MODIFICAR: `twillot_scraper.py`
Adaptar `TwillotImporter` para formato real do export:

```python
# Mapeamento de campos Twillot → interno:
tweet_id      → id
full_text     → text
media_items   → media_urls
screen_name   → author_username
username      → author_name
created_at    → created_at
has_image/has_video/has_gif → media_types
```

#### 4. MODIFICAR: `main.py`
- Nova opção de import com mídias locais
- Nova opção de análise multimodal

### Novo Menu
```
=== Twitter Bookmarks App ===

Fetch Bookmarks:
1. Fetch via Twillot (browser automation)
2. Import Twillot export (JSON + mídias locais)  ← MODIFICADO
3. Load saved bookmarks from file

View & Export:
4. View bookmark statistics
5. Export bookmarks to Markdown
6. Expand URLs in loaded bookmarks

Analysis:
7. Analyze bookmark topics
8. Smart processing (by topic)
9. Multimodal analysis (text + images)  ← NOVO

Settings:
10. Configure LLM provider
11. Exit
```

---

## Modelos de Visão (Dezembro 2025)

### GPT-5.2 (OpenAI) - Lançado 11/12/2025
- **Variantes**: Instant, Thinking, Pro
- **Visão**: Erro 50% menor em chart reasoning
- **Contexto**: 400K tokens, 128K output
- **Preço**: $1.75/M input, $14/M output
- **Model IDs**: `gpt-5.2-instant`, `gpt-5.2-thinking`, `gpt-5.2-pro`
- **Fonte**: https://openai.com/index/introducing-gpt-5-2/

### Claude Opus 4.5 (Anthropic) - Lançado 24/11/2025
- **Visão**: 80.7% MMMU
- **Contexto**: 200K tokens, 64K output
- **Preço**: $5/M input, $25/M output
- **Model ID**: `claude-opus-4-5-20251101`
- **Fonte**: https://www.anthropic.com/news/claude-opus-4-5

### Gemini 3 Pro (Google) - Lançado 18/11/2025
- **Visão**: 81% MMMU-Pro, 87.6% Video-MMMU
- **Destaques**: Document derendering, Video 10 FPS
- **Contexto**: 1M tokens
- **Model ID**: `gemini-3-pro`
- **Fonte**: https://blog.google/products/gemini/gemini-3/

### Gemini 2.5 Flash (Google) - Disponível
- **Preço**: $0.10/M input, $0.40/M output (~$0.001/imagem)
- **Melhor custo-benefício para alto volume**
- **Model ID**: `gemini-2.5-flash`
- **Fonte**: https://ai.google.dev/gemini-api/docs/pricing

---

## Estratégia de Uso dos Modelos

| Caso de Uso | Modelo | Motivo |
|-------------|--------|--------|
| Triagem em massa | Gemini 2.5 Flash | Mais barato |
| Qualidade máxima | GPT-5.2 Thinking | Melhor precisão |
| Código/diagramas | Claude Opus 4.5 | Melhor para coding |
| Documentos complexos | Gemini 3 Pro | Derendering superior |

### Pipeline Sugerido
1. **Triagem** (Gemini Flash): Classificar bookmarks por tipo
2. **Análise profunda** (GPT-5.2/Claude): Conteúdo técnico
3. **Batch processing** (Gemini Flash): Volume alto

---

## Limites dos Modelos

| Modelo | Max Imagens | Contexto | Output |
|--------|-------------|----------|--------|
| GPT-5.2 | TBD | 400K | 128K |
| Claude Opus 4.5 | 20 imgs, 5MB cada | 200K | 64K |
| Gemini 3 Pro | Ilimitado* | 1M | - |
| Gemini 2.5 Flash | Ilimitado* | 1M | - |

*Limitado por tokens totais

---

## Ordem de Execução (Fase 2)

1. [x] Criar `media_linker.py`
2. [x] Modificar `twillot_scraper.py` - formato real
3. [x] Modificar `llm_providers.py` - visão
4. [x] Modificar `main.py` - import + análise
5. [ ] Testar com subset
6. [x] Atualizar documentação

---

## Considerações Técnicas

### Vídeos
- Extrair thumbnail/primeiro frame
- Usar URL do thumbnail (`media_items` no JSON)
- Gemini 3 Pro pode processar vídeo direto (10 FPS)

### Performance
- Processar em batches
- Cache de análises já feitas
- Fallback automático entre providers

### Formato do Export Twillot
```json
{
  "tweet_id": "2000582653053993313",
  "full_text": "texto do tweet...",
  "media_items": ["https://pbs.twimg.com/..."],
  "has_image": false,
  "has_video": true,
  "screen_name": "username",
  "username": "Display Name",
  "created_at": "2025-12-15T15:04:08.000Z",
  "favorite_count": 440,
  "retweet_count": 37,
  "views_count": "47143"
}
```

### Formato dos Arquivos de Mídia
```
twillot-media-files-by-date/
├── 2025-02-18/
│   ├── johnrushx-1891684077994123426-GkCco5taAAARYUs.png
│   └── tedx_ai-1891888277940003051-GkFWXIXWwAE3Zxs.jpg
└── 2025-03-01/
    └── ...
```

Pattern: `{screen_name}-{tweet_id}-{media_id}.{ext}`
