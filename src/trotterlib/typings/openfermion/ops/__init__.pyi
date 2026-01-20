from typing import Any

QubitOperator = Any
FermionOperator = Any

def __getattr__(name: str) -> Any: ...
