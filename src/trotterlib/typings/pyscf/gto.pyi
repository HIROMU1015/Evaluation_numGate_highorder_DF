from typing import Any

class Mole:
    atom: Any
    basis: Any
    spin: Any
    charge: Any
    symmetry: Any

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def build(self, *args: Any, **kwargs: Any) -> Any: ...
