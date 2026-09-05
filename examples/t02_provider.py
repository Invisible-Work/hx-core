import asyncio

from hx.contracts import Provider
from hx.providers.anthropic import AnthropicProvider
from hx.providers.gemini import GeminiProvider
from hx.providers.ollama import OllamaProvider
from hx.providers.openai import OpenAIProvider
from hx.types import Message


async def probe(p):
    c = await p.complete([Message("user", "Diga apenas: funcionou")], tools=[])
    print(p.name, "→", c.message.content, "|", c.stop_reason, "|", c.usage)


async def main():
    assert isinstance(AnthropicProvider(), Provider)
    assert isinstance(OpenAIProvider(), Provider)
    assert isinstance(GeminiProvider(), Provider)
    assert isinstance(OllamaProvider(), Provider)
    await probe(AnthropicProvider())
    await probe(OpenAIProvider())
    await probe(GeminiProvider())
    await probe(OllamaProvider())


asyncio.run(main())
