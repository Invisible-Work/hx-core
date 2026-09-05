# AGENTS.md — hx-core

Instruções para agentes e contribuidores que trabalham neste repositório.

## O que é

`hx-core` é uma biblioteca Python pequena para o loop de agente: providers plugáveis, memory drivers e estratégias de context engineering. A ideia é melhorar compactação (e o resto do núcleo) **uma vez** — todo agente herda.

Pacote instalável: `hx` (módulo em `src/hx/`). Python `>=3.14`. Gerenciador: `uv`.

## Princípios

1. **Núcleo magro.** Dependências obrigatórias ficam mínimas (`httpx`, `pydantic`). SDKs de LLM, memória, telemetria etc. entram só como **extras opcionais**.
2. **Contratos via Protocol.** Extensões implementam protocols em `hx.contracts` (`Provider`, `MemoryDriver`, `ContextStrategy`, `Sandbox`, `Hook`). Não há classe base obrigatória — `isinstance(obj, Provider)` funciona porque o protocol é `@runtime_checkable`.
3. **Tipos neutros.** O loop fala só `Message`, `ToolCall`, `Completion`, `Usage`, `StopReason` (`hx.types`). Nunca tipos do SDK do vendor.
4. **Sem vazamento de SDK.** Importar `hx.types`, `hx.errors`, `hx.contracts`, `hx.core`, `hx.tools` **não** pode puxar `anthropic`, `openai`, `litellm`, `mem0`, `opentelemetry`, etc. O teste `tests/test_no_provider_imports.py` guarda isso. Por isso `hx.providers.__init__` fica vazio: quem quiser o adaptador importa o módulo do provider explicitamente.
5. **Tradução encapsulada.** Cada adapter traduz ida (`_to_wire` / `_tool_to_wire`) e volta (`_from_wire`). O núcleo não conhece o formato do vendor.
6. **Histórico é nosso.** O estado da conversa vive em `list[Message]`. Não use handles stateful do vendor que vazam para o núcleo (ex.: `previous_response_id` da OpenAI Responses API).

## Layout

```
src/hx/
  types.py          # Message, ToolCall, Completion, Usage, StopReason
  errors.py         # HxError, ProviderError, MissingExtra, …
  contracts.py      # Protocols (Provider, MemoryDriver, …)
  providers/        # adapters de LLM (extras opcionais)
    anthropic.py
    openai.py
    gemini.py
    __init__.py     # vazio de propósito
  memory/           # (futuro) drivers de memória
  context/          # (futuro) estratégias de contexto
  hooks/            # (futuro) hooks
examples/           # scripts manuais de verificação
tests/              # pytest
```

## Ambiente de desenvolvimento

```bash
uv sync --extra anthropic --extra openai --extra gemini   # + dependency-group dev via uv sync --group dev se precisar
uv run pytest
uv run ruff check .
uv run mypy src
```

Extras atuais:

| Extra       | Pacote         | Módulo                         |
|-------------|----------------|--------------------------------|
| `anthropic` | `anthropic`    | `hx.providers.anthropic`       |
| `openai`    | `openai`       | `hx.providers.openai`          |
| `gemini`    | `google-genai` | `hx.providers.gemini`          |

Variáveis de ambiente típicas: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` / `GOOGLE_API_KEY`.

Prova rápida dos providers:

```bash
uv run python examples/t02_provider.py
```

## Estilo de código

- `from __future__ import annotations` no topo dos módulos.
- Imports no topo do arquivo — sem imports inline (exceto o `try/except ModuleNotFoundError` do SDK no provider, que é o padrão de extra opcional).
- Dataclasses frozen para tipos públicos imutáveis.
- Async para I/O de provider (`async def complete`).
- Erros do vendor → `ProviderError(f"{name}: {e}") from e`.
- Extra ausente → `MissingExtra("<extra>")`.
- Comentários só quando o “porquê” não é óbvio; não narrar o óbvio.
- Foco na tarefa pedida — sem refactors ou features laterais.

## Checklist antes de abrir PR

- [ ] Núcleo continua sem importar SDKs externos (`pytest tests/test_no_provider_imports.py`)
- [ ] Novo SDK (se houver) está em `[project.optional-dependencies]`, não em `dependencies`
- [ ] Adapter novo passa no `isinstance(..., Provider)` e responde ao mesmo `complete(messages, tools)`
- [ ] Sem reexportar o SDK / provider em `hx.providers.__init__` ou no núcleo
- [ ] `ruff` / `mypy` limpos no que você tocou

---

## Padrão: como incluir um novo provider

Um provider é um **adaptador de tradução**: recebe tipos neutros do hx, fala o fio do vendor, devolve `Completion`. Use `AnthropicProvider` e `OpenAIProvider` como referências — se o segundo adapter não couber na mesma interface, a abstração está errada.

### 1. Extra opcional (nunca dependência obrigatória)

```bash
uv add --optional <nome> <pacote-sdk>
uv sync --extra <nome>
```

Isso grava em `[project.optional-dependencies].<nome>` no `pyproject.toml`. Quem instala `hx-core` sem o extra continua com o núcleo leve.

Adicione o nome do pacote SDK à lista `PROIBIDOS` em `tests/test_no_provider_imports.py` se ainda não estiver lá.

### 2. Arquivo do adapter

Crie `src/hx/providers/<nome>.py`. **Não** reexporte em `providers/__init__.py`.

Consumo esperado:

```python
from hx.providers.<nome> import <Nome>Provider
```

### 3. Import preguiçoso do SDK

```python
from hx.errors import MissingExtra, ProviderError

try:
    import <sdk>
except ModuleNotFoundError as e:
    raise MissingExtra("<nome>") from e
```

`MissingExtra` já instrui `uv add 'hx-core[<nome>]'` / `pip install 'hx-core[<nome>]'`.

### 4. Interface obrigatória (`Provider`)

Não herde de nada. Implemente o protocol estruturalmente:

| Membro | Contrato |
|--------|----------|
| `name: str` | Identificador curto (`"anthropic"`, `"openai"`, …) |
| `async def complete(self, messages, tools, **kw) -> Completion` | Uma chamada ao modelo |
| `def count_tokens(self, messages) -> int` | Estimativa para compactação (hoje: ~4 chars/token) |

Construtor típico:

```python
def __init__(self, model: str = "<default>", max_tokens: int = 4096, **client_kw: Any):
    self._client = <SdkAsyncClient>(**client_kw)
    self._model = model
    self._max_tokens = max_tokens
```

- `**client_kw` vai para o cliente do vendor (api_key, base_url, timeout, …).
- `max_tokens` no construtor/kwargs do hx é o nome **neutro**; mapeie para o campo do vendor (`max_tokens`, `max_output_tokens`, …) dentro do adapter.
- Cliente **async**.

Checkpoint: `isinstance(<Nome>Provider(...), Provider)` deve ser `True`.

### 5. Tradução (o trabalho de verdade)

Métodos privados convencionais:

| Método | Responsabilidade |
|--------|------------------|
| `_split_system(messages)` | Se o vendor tira system do array (Anthropic `system=`, OpenAI Responses `instructions=`). Se o vendor mantém system nas mensagens, não precisa. |
| `_tool_to_wire(t: dict) -> dict` | Schema neutro `{name, description?, input_schema}` → formato do vendor |
| `_to_wire(convo) -> …` | `list[Message]` → payload do vendor |
| `_from_wire(resp) -> Completion` | Resposta do vendor → `Completion` neutro |

#### Formato neutro (entrada/saída do hx)

```python
Message(role="assistant", tool_calls=(ToolCall("t1", "saldo", {"conta": "1010"}),))
Message(role="tool", content="42.0", tool_call_id="t1")
```

- `ToolCall.arguments` é sempre `dict` no hx.
- `StopReason` ∈ `{"end_turn", "tool_use", "max_tokens", "stop"}`.
- `Usage` usa `input_tokens` / `output_tokens` / `cached_tokens` (normalize nomes do vendor).
- Guarde a resposta bruta em `Completion.raw` se útil para debug — o loop não depende dela.

#### Diferenças comuns entre vendors (o que o adapter esconde)

| Tema | Anthropic | OpenAI (Responses) | Gemini (`google-genai`) |
|------|-----------|--------------------|-------------------------|
| Chamada | `client.messages.create` | `client.responses.create` | `client.aio.models.generate_content` |
| System | param `system=` | param `instructions=` | `GenerateContentConfig.system_instruction` |
| Tool schema | `input_schema` | `parameters` (+ `type: "function"` plano) | `FunctionDeclaration.parameters_json_schema` |
| Tool call | bloco `tool_use`, `input` = dict | item `function_call`, `arguments` = JSON **string** | `Part.function_call`, `args` = dict (+ `id` opcional) |
| Tool result | `role=user` + `tool_result` (agrupar consecutivos) | item `function_call_output` | `role=tool` + `function_response` (agrupar consecutivos) |
| Stop | `stop_reason` | `status` + presença de `function_call` | `finish_reason` + presença de `function_call` |

Ao adicionar um terceiro vendor, documente a linha equivalente nessa tabela mental e encapsule **tudo** no módulo do provider.

#### Regras que não negociar

- **Não** vaze IDs de sessão/response do vendor para o núcleo. Sempre reenvie o histórico traduzido a partir de `list[Message]`.
- **Não** deixe `tools=[]` quebrar a API: se o vendor rejeita lista vazia, omita o parâmetro (padrão OpenAI Responses).
- Converta `arguments` string↔dict **só** na borda do adapter.
- Mapeie stop reasons para o literal neutro; desconhecido → `"stop"`.
- Exceções do SDK → `ProviderError`; nunca deixe o tipo do vendor subir.

### 6. Esqueleto mínimo de `complete`

```python
async def complete(self, messages, tools, **kw) -> Completion:
    # 1. separar system se o vendor exigir
    # 2. montar kwargs (model, max_*, input/messages, tools opcional)
    try:
        resp = await self._client.<api>(...)
    except Exception as e:
        raise ProviderError(f"{self.name}: {e}") from e
    return self._from_wire(resp)
```

### 7. `count_tokens`

Enquanto não houver contagem oficial por vendor, use a mesma aproximação dos adapters existentes (suficiente para decisão de compactação):

```python
def count_tokens(self, messages) -> int:
    chars = sum(len(m.content) for m in messages)
    chars += sum(len(str(c.arguments)) for m in messages for c in m.tool_calls)
    return chars // 4
```

Se o SDK oferecer contagem barata e sync/async compatível, pode evoluir **dentro** do adapter — sem mudar o contrato.

### 8. Provar que funciona

1. `isinstance(p, Provider)` verdadeiro.
2. Mesmo código de chamada que os outros:

```python
c = await p.complete([Message("user", "Diga apenas: funcionou")], tools=[])
# c.message.content, c.stop_reason, c.usage
```

3. Idealmente estender `examples/t02_provider.py` (ou um script irmão) para o novo provider — o checkpoint pedagógico é: **dois (ou N) providers diferentes respondendo ao mesmo código de chamada**.

### 9. Checklist do novo provider

- [ ] Extra em `pyproject.toml` via `uv add --optional …`
- [ ] Módulo `src/hx/providers/<nome>.py` com import preguiçoso + `MissingExtra`
- [ ] `name`, `complete`, `count_tokens` alinhados ao protocol
- [ ] `_to_wire` / `_from_wire` / `_tool_to_wire` cobrem texto, tool calls e tool results
- [ ] `StopReason` e `Usage` normalizados
- [ ] Sem estado stateful do vendor no núcleo
- [ ] SDK listado em `PROIBIDOS` no teste de vazamento
- [ ] Sem reexport em `providers/__init__.py`
- [ ] Smoke test manual com API key real
