"""Result history for a :class:`TabraData` analysis session."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResultRecord:
    """One completed command and its associated objects."""

    command: str
    result: Any
    model: Any = None


class ResultStore:
    """Keep completed results without changing the legacy ``tab.result`` API."""

    def __init__(self) -> None:
        self._records: list[ResultRecord] = []
        self._named: dict[str, ResultRecord] = {}

    @property
    def last(self):
        """Return the latest result, or ``None`` before the first command."""
        if not self._records:
            return None
        return self._records[-1].result

    @property
    def history(self) -> tuple[ResultRecord, ...]:
        """Return an immutable view of the command history."""
        return tuple(self._records)

    def add(self, result, *, command: str, model=None) -> ResultRecord:
        """Append a completed command to the session history."""
        record = ResultRecord(command=command, result=result, model=model)
        self._records.append(record)
        return record

    def store(self, name: str, result=None) -> None:
        """Store the latest or a supplied result under a reusable name."""
        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")

        if result is None:
            if not self._records:
                raise ValueError("No result is available to store")
            record = self._records[-1]
        else:
            record = next(
                (item for item in reversed(self._records) if item.result is result),
                ResultRecord(command="", result=result),
            )
        self._named[name] = record

    def get(self, name: str):
        """Return a named result."""
        try:
            return self._named[name].result
        except KeyError as exc:
            raise KeyError(f"Stored result '{name}' was not found") from exc

    def clear(self) -> None:
        """Clear command history and named results."""
        self._records.clear()
        self._named.clear()

    def __len__(self) -> int:
        return len(self._records)
