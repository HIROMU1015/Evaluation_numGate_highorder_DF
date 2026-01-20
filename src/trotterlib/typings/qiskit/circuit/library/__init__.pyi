from typing import Any

GlobalPhaseGate = Any
PauliEvolutionGate = Any

def __getattr__(name: str) -> Any: ...
