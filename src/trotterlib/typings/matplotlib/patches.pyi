from typing import Any

Patch = Any

def __getattr__(name: str) -> Any: ...
