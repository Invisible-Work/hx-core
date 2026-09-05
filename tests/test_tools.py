import pytest

from hx.errors import PermissionDenied, ToolNotFound
from hx.tools import ToolRegistry
from hx.types import ToolCall

reg = ToolRegistry()


@reg.tool(risk="read_only")
async def saldo(conta: str, incluir_pendentes: bool = False) -> float:
    """Retorna o saldo atual da conta."""
    return 42.0 if not incluir_pendentes else 40.0


@reg.tool(risk="destructive")
def apagar(conta: str) -> str:
    """Apaga a conta."""
    return "apagada"


def test_schema_derivado():
    s = reg.schemas()[0]
    assert s["name"] == "saldo"
    assert s["description"].startswith("Retorna o saldo")
    props = s["input_schema"]["properties"]
    assert props["conta"]["type"] == "string"
    assert props["incluir_pendentes"]["default"] is False
    assert s["input_schema"]["required"] == ["conta"]


async def test_invoca_async_e_sync():
    assert await reg.invoke(ToolCall("1", "saldo", {"conta": "1010"})) == 42.0
    assert await reg.invoke(ToolCall("2", "apagar", {"conta": "1010"})) == "apagada"


async def test_argumento_invalido():
    call = ToolCall("3", "saldo", {})
    with pytest.raises(ValueError):
        await reg.invoke(call)


async def test_ferramenta_inexistente():
    call = ToolCall("4", "inexistente", {})
    with pytest.raises(ToolNotFound):
        await reg.invoke(call)


async def test_filtro_de_risco():
    somente_leitura = reg.filtered(allow=("read_only",))
    assert len(somente_leitura.schemas()) == 1
    call = ToolCall("5", "apagar", {"conta": "1"})
    with pytest.raises(PermissionDenied):
        await somente_leitura.invoke(call)
    assert len(reg.schemas()) == 2   # o original não foi mutado