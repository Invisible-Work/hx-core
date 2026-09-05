import importlib
import sys

PROIBIDOS = {"anthropic", "openai", "google.genai", "litellm", "mem0", "opentelemetry"}
NUCLEO = ["hx.types", "hx.errors", "hx.contracts", "hx.core", "hx.tools"]


def test_nucleo_nao_importa_sdk_externo():
    for mod in NUCLEO:
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError:
            pass
    vazados = PROIBIDOS & set(sys.modules)
    assert not vazados, f"o núcleo vazou dependências: {vazados}"