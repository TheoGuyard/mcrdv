#!/usr/bin/env python3
"""Build `data/iridium33_rivera.csv`: the Iridium-33 debris cloud López Rivera
& Naeije (2025) sample their missions from, as of October 2024.

CelesTrak serves today's elements only. The objects still in orbit are carried
back to `EPOCH` by their secular J2 drift (node and argument of perigee; the
decay of the semi-major axis since then is ignored), and the fragments that
have decayed meanwhile are lost. The first row is where their chaser departs:
the centroid of the cloud, weighted by radar cross-section (RCS), angles
averaged on the circle; an object with no RCS on record weighs nothing.

    ./cloud.py                    # fetch from CelesTrak, write the catalogue
    ./cloud.py --gp gp.json --satcat satcat.json    # from saved downloads
"""

import argparse
import json
import math
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcrdv.io import save_states  # noqa: E402
from mcrdv.orbit import MU, TAU, KepState  # noqa: E402

from _common.planners import DATA  # noqa: E402

GP = "https://celestrak.org/NORAD/elements/gp.php?GROUP={}&FORMAT=json"
SATCAT = "https://celestrak.org/satcat/records.php?GROUP={}&FORMAT=json"
GROUP = "iridium-33-debris"

# The date of the cloud they model (their Figs. 7 to 9: "as of October 2024").
EPOCH = datetime(2024, 10, 15, tzinfo=timezone.utc)


def _read(path, url):
    """A saved download if given, else the live answer of CelesTrak."""
    if path is not None:
        return json.loads(Path(path).read_text())
    with urllib.request.urlopen(url, timeout=120) as reply:
        return json.loads(reply.read())


def _true_anomaly(mean: float, e: float) -> float:
    """True anomaly [rad] from the mean anomaly, by Newton on Kepler."""
    big = mean
    for _ in range(50):
        big -= (big - e * math.sin(big) - mean) / (1.0 - e * math.cos(big))
    return 2.0 * math.atan2(
        math.sqrt(1.0 + e) * math.sin(0.5 * big),
        math.sqrt(1.0 - e) * math.cos(0.5 * big),
    ) % TAU


def _state(record: dict) -> KepState:
    """A CelesTrak element set as a Keplerian state, carried to `EPOCH`."""
    n = record["MEAN_MOTION"] * TAU / 86400.0
    e = float(record["ECCENTRICITY"])
    state = KepState(
        (MU / (n * n)) ** (1.0 / 3.0),
        e,
        math.radians(record["INCLINATION"]),
        math.radians(record["RA_OF_ASC_NODE"]),
        math.radians(record["ARG_OF_PERICENTER"]),
        _true_anomaly(math.radians(record["MEAN_ANOMALY"]), e),
    )
    epoch = datetime.fromisoformat(record["EPOCH"]).replace(
        tzinfo=timezone.utc
    )
    moved = state.propagate((EPOCH - epoch).total_seconds())
    return KepState(
        moved.a, moved.e, moved.i, moved.r % TAU, moved.o % TAU, moved.t
    )


def _mean(angles: np.ndarray, weights: np.ndarray) -> float:
    return math.atan2(
        float(weights @ np.sin(angles)), float(weights @ np.cos(angles))
    ) % TAU


def centroid(states, weights) -> KepState:
    """The centroid of `states` weighted by `weights`, angles on the circle."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    column = {
        name: np.array([getattr(s, name) for s in states])
        for name in "aeirot"
    }
    return KepState(
        float(w @ column["a"]),
        float(w @ column["e"]),
        float(w @ column["i"]),
        _mean(column["r"], w),
        _mean(column["o"], w),
        _mean(column["t"], w),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gp", type=Path, help="saved CelesTrak GP JSON")
    parser.add_argument("--satcat", type=Path, help="saved SATCAT JSON")
    parser.add_argument(
        "--out", type=Path, default=DATA / "iridium33_rivera.csv"
    )
    args = parser.parse_args()

    elements = _read(args.gp, GP.format(GROUP))
    satcat = _read(args.satcat, SATCAT.format(GROUP))
    catalogue = {record["NORAD_CAT_ID"]: record for record in satcat}

    states, weights = [], []
    for record in sorted(elements, key=lambda r: r["NORAD_CAT_ID"]):
        states.append(_state(record))
        rcs = catalogue.get(record["NORAD_CAT_ID"], {}).get("RCS")
        weights.append(float(rcs) if rcs is not None else 0.0)

    save_states(args.out, [centroid(states, weights)] + states)
    dates = sorted(record["EPOCH"][:10] for record in elements)
    print(
        f"wrote {args.out}: {len(states)} objects (elements of "
        f"{dates[0]} to {dates[-1]}, carried to {EPOCH.date()}), "
        f"{sum(w > 0 for w in weights)} with an RCS"
    )
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
