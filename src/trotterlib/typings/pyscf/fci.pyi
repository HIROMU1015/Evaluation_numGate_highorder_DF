from typing import Any, Tuple

cistring: Any

class FCI:
    norb: int
    nelec: Any
    mo_coeff: Any

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def kernel(self, *args: Any, **kwargs: Any) -> Tuple[Any, Any]: ...
