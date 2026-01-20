from typing import Any

from . import library

def __getattr__(name: str) -> Any: ...
