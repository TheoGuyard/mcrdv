"""I-O utilities for orbital states."""

from pathlib import Path
from typing import List

from .orbit import KepState

HEADER = "a[m],e[prop],i[rad],r[rad],o[rad],t[rad]"


def load_states(path: Path) -> List[KepState]:
    """Read a state file with states given in Keplerian elements.

    The file is expected to be formatted as follows, with one row per state:

    ```
    a[m],e[prop],i[rad],r[rad],o[rad],t[rad]
    7000000,0.001,0.1,0.2,0.3,0.4
    7100000,0.002,0.2,0.3,0.4,0.5
    ...
    ```
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()

    header = lines[0] if lines else ""
    if header.strip() != HEADER:
        raise ValueError(f"{path}: invalid header, expected '{HEADER}'")

    states: List[KepState] = []
    for offset, line in enumerate(lines[1:]):
        if not line.strip():
            continue
        number = offset + 2
        values = []
        for field in line.split(",")[:6]:
            try:
                values.append(float(field.strip()))
            except ValueError:
                raise ValueError(f"{path}, line {number}: cannot parse fields")
        if len(values) < 6:
            raise ValueError(f"{path}, line {number}: expected 6 columns")
        states.append(KepState(*values))
    return states


def save_states(path: Path, states: List[KepState]) -> None:
    """Write a state file with states given in Keplerian elements."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as file:
        file.write(HEADER + "\n")
        for state in states:
            file.write(",".join(str(f) for f in state.as_array()) + "\n")
