"""What a run leaves behind: one pickle holding its configuration and results.

A record has three fields and no more. Anything further would be a guess about
what an experiment cares about, and experiments here do not all solve the same
problem with the same planner. Timings, metrics, a mission plan, a search trace,
another solver's own logs -- all of it is keys in `result`, chosen by whoever
produced them.
"""

from __future__ import annotations

import pickle
import secrets
import string
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from . import config as configuration

# How a record names itself: sortable by time, unique without coordination.
STAMP = "%Y%m%d_%H%M%S"
# Length of the random suffix, in lowercase letters. 26^8 is about 2e11, so two
# tasks writing in the same second are not going to collide.
TOKEN = 8


@dataclass
class RunRecord:
    """One run: what it was asked to do, and what came of it."""

    # Stem of the file this record lives in.
    run_id: str
    # The exact resolved configuration, copied verbatim.
    config: Dict[str, Any] = field(default_factory=dict)
    # Whatever the experiment's `execute` returned.
    result: Dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        """Whether the run raised instead of finishing."""
        return "error" in self.result

    def row(self) -> Dict[str, Any]:
        """The record as one flat row: configuration and results together.

        Values a table cannot hold -- a mission plan, a trace -- are left out
        rather than stringified; they stay on the record for whoever wants them.
        """
        row = {"run_id": self.run_id}
        row.update(configuration.flatten(self.config))
        row.update(configuration.flatten(self.result))
        return row


def token(length: int = TOKEN) -> str:
    """A random string of lowercase letters.

    Letters only, no digits: a name that is all one alphabet is easier to read
    back, to select with a double-click, and to tell apart from the timestamp it
    follows. Drawn from `secrets` rather than `random` because array tasks start
    at the same instant and must not share a seeded sequence.
    """
    return "".join(
        secrets.choice(string.ascii_lowercase) for _ in range(length)
    )


def name(when: datetime | None = None) -> str:
    """A fresh name, `YYYYMMDD_HHMMSS_xxxxxxxx`.

    The timestamp leads so a directory listing is chronological, and the random
    suffix keeps two writers in the same second from colliding without having to
    coordinate. Records and saved figures share this format, so a figure sorts
    alongside the runs it was drawn from.
    """
    when = when or datetime.now()
    return f"{when.strftime(STAMP)}_{token()}"


def write(out: Path, config: Dict[str, Any], result: Dict[str, Any]) -> Path:
    """Write one record, returning where it landed."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # Vanishingly unlikely, but a collision would silently destroy a result, so
    # it costs nothing to look first.
    while True:
        run_id = name()
        path = out / f"{run_id}.pkl"
        if not path.exists():
            break

    record = RunRecord(run_id=run_id, config=config, result=result)
    with open(path, "wb") as handle:
        pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def read(path: Path) -> RunRecord:
    """Read one record back."""
    with open(path, "rb") as handle:
        return pickle.load(handle)


def paths(source: Path) -> List[Path]:
    """Every record under `source`, oldest first."""
    return sorted(Path(source).glob("*.pkl"))


def load(source: Path) -> Iterator[RunRecord]:
    """Every record under `source`, in name order.

    A record that cannot be unpickled is reported and skipped rather than
    stopping the load: one truncated file from a killed job should not make a
    whole campaign unreadable.
    """
    for path in paths(source):
        try:
            yield read(path)
        except Exception as error:
            print(f"skipping {path.name}: {type(error).__name__}: {error}")


def rows(records: Iterable[RunRecord]) -> List[Dict[str, Any]]:
    """Records as flat rows, ready for selection or a table."""
    return [record.row() for record in records]
