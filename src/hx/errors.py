class HxError(Exception):
    """Base de todos os erros da biblioteca."""


class MaxTurnsExceeded(HxError):
    def __init__(self, state):
        self.state = state
        super().__init__(f"limite de {state.turn} turnos excedido")


class BudgetExceeded(HxError):
    def __init__(self, state, kind: str):
        self.state, self.kind = state, kind
        super().__init__(f"orçamento excedido: {kind}")


class ToolNotFound(HxError): ...
class PermissionDenied(HxError): ...
class ProviderError(HxError): ...
class MissingExtra(HxError):
    def __init__(self, extra: str):
        super().__init__(
            f"esta funcionalidade exige o extra '{extra}'. "
            f"Instale com: uv add 'hx-core[{extra}]' "
            f"(ou pip install 'hx-core[{extra}]')"
        )