"""Reading catalogues and rendering mission plans."""

from __future__ import annotations

import pytest

from mcrdv import ChaserPlan, MissionPlan, load_states
from mcrdv.io import HEADER

from .conftest import DATA

SIZES = {
    "odrc": 13,
    "cosmos1408": 4,
    "iridium33": 110,
    "cosmos2251": 583,
    "fengyun1C": 1847,
}


@pytest.mark.parametrize("name", list(SIZES))
def test_shipped_catalogues_load_at_their_documented_size(name):
    targets = load_states(DATA / f"{name}.csv")
    assert len(targets) == SIZES[name]
    assert all(d.a > 0 and 0.0 <= d.e < 1.0 for d in targets)


def test_first_row_is_parsed_into_the_six_elements(tmp_path):
    path = tmp_path / "one.csv"
    path.write_text(
        f"{HEADER}\n7164040.5518,0.0019,1.5079,2.8765,0.8909,5.3923\n"
    )
    (d,) = load_states(path)
    assert (d.a, d.e, d.i, d.r, d.o, d.t) == (
        7164040.5518,
        0.0019,
        1.5079,
        2.8765,
        0.8909,
        5.3923,
    )


def test_a_path_given_as_a_string_is_accepted(tmp_path):
    path = tmp_path / "one.csv"
    path.write_text(f"{HEADER}\n7.0e6,0.001,1.5,0.1,0.2,0.3\n")
    assert len(load_states(str(path))) == 1


def test_surplus_columns_are_ignored(tmp_path):
    # Only the six documented elements are read, so a catalogue carrying extra
    # bookkeeping columns still loads.
    path = tmp_path / "wide.csv"
    path.write_text(f"{HEADER}\n7.0e6,0.001,1.5,0.1,0.2,0.3,SL-16 R/B,1994\n")
    (d,) = load_states(path)
    assert (d.a, d.t) == (7.0e6, 0.3)


def test_surrounding_spaces_are_tolerated(tmp_path):
    path = tmp_path / "spaced.csv"
    path.write_text(f"{HEADER}\n 7.0e6 , 0.001 ,1.5, 0.1,0.2 , 0.3 \n")
    (d,) = load_states(path)
    assert (d.a, d.e, d.i) == (7.0e6, 0.001, 1.5)


def test_a_header_only_file_reads_as_an_empty_catalogue(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text(f"{HEADER}\n")
    assert load_states(path) == []


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "gaps.csv"
    path.write_text(
        f"{HEADER}\n7.0e6,0.001,1.5,0.1,0.2,0.3\n\n7.1e6,0.001,1.5,0.1,0.2,0.3\n\n"
    )
    assert len(load_states(path)) == 2


@pytest.mark.parametrize(
    "body, message",
    [
        ("a,b,c\n", "invalid header"),
        (f"{HEADER}\n7.0e6,0.001,1.5,0.1,0.2\n", "expected 6 columns"),
        (f"{HEADER}\n7.0e6,oops,1.5,0.1,0.2,0.3\n", "cannot parse fields"),
    ],
)
def test_malformed_files_are_rejected_with_a_located_message(
    tmp_path, body, message
):
    path = tmp_path / "bad.csv"
    path.write_text(body)
    with pytest.raises(ValueError, match=message):
        load_states(path)


def test_parse_error_names_the_file_line_number(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        f"{HEADER}\n7.0e6,0.001,1.5,0.1,0.2,0.3\n7.0e6,x,1.5,0.1,0.2,0.3\n"
    )
    with pytest.raises(ValueError, match="line 3"):
        load_states(path)


def test_save_emits_one_row_per_visited_object(tmp_path):
    mission = MissionPlan(
        [
            ChaserPlan.of(
                [0, 2, 5], [0.0, 1.0, 2.0], [0.0, 3.0, 4.0], [0.0, 7.0, 8.0]
            ),
            ChaserPlan.empty(),
        ]
    )
    path = tmp_path / "mission.csv"
    mission.save(path, dec=2)
    rows = path.read_text().strip().split("\n")
    assert rows[0] == "chaser,state,depart,arrive,cost"
    assert rows[1] == "0,0,0.00,0.00,0.00"
    assert rows[3] == "0,5,2.00,4.00,8.00"
    assert rows[4] == "1,0,0.00,0.00,0.00"
    assert len(rows) == 5


def test_save_honours_the_requested_precision(tmp_path):
    mission = MissionPlan(
        [ChaserPlan.of([0, 2], [0.0, 1.0 / 3.0], [0.0, 2.0 / 3.0], [0.0, 1.5])]
    )
    path = tmp_path / "mission.csv"
    mission.save(path, dec=4)
    assert path.read_text().splitlines()[2] == "0,2,0.3333,0.6667,1.5000"


def test_save_overwrites_rather_than_appends(tmp_path):
    path = tmp_path / "mission.csv"
    mission = MissionPlan([ChaserPlan.empty()])
    mission.save(path)
    mission.save(path)
    assert len(path.read_text().strip().split("\n")) == 2  # header + one row
