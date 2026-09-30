"""Application planning policy, independent of the cuRobo Python package."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PlanOptions:
    max_attempts: int = 20
    graph_attempt: int | None = 1
