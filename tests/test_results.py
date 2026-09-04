"""Tests for :mod:`gsim_tools.results` -- both real result formats, and running the simulator.

The CSV tests use the fixture traces the real ``general-simulator`` binary produced (see
``conftest.py``). The kernel-sidecar manifest tests build a result directory in exactly the
shape ``general-simulator-gui``'s ``result_writer.py``/``hdf5_store.py`` write, and skip when
``numpy``/``h5py`` are absent -- those are the optional ``sidecar`` extra, and the point of the
whole package layout is that everything else still works without them.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from conftest import FIXTURES, fixture_text
from gsim_tools import logs, results


# ---------------------------------------------------------------------------
# general-simulator CSV
# ---------------------------------------------------------------------------


def test_read_csv_columns_and_time_base():
    trace = results.read_csv(fixture_text("sine.csv"))
    # Header from sine.cir's own run: t,V(a),I(VSIG),SIG,SIGV -- the first column is time and
    # never becomes a signal.
    assert trace.names == ["V(a)", "I(VSIG)", "SIG", "SIGV"]
    assert "t" not in trace
    assert len(trace) == 2000
    assert len(trace["V(a)"]) == len(trace)
    assert trace.metadata["time_column"] == "t"
    assert trace.time[0] == pytest.approx(1e-4)
    assert trace.time[-1] == pytest.approx(0.2)


def test_samples_bridge_into_measure():
    trace = results.read_csv(fixture_text("sine.csv"))
    samples = trace.samples("V(a)")
    assert len(samples) == len(trace)
    assert samples[0].t == trace.time[0]
    assert samples[0].v == trace["V(a)"].values[0]
    # Strictly increasing time, which every measurement requires.
    assert all(b.t > a.t for a, b in zip(samples, samples[1:]))


def test_unknown_signal_name_lists_what_is_available():
    trace = results.read_csv(fixture_text("sine.csv"))
    with pytest.raises(KeyError, match="V\\(a\\)"):
        trace["V(nope)"]


def test_empty_csv_is_an_empty_trace_not_an_error():
    # print_csv writes nothing at all -- not even a header -- for a run that resolved no
    # points, so an empty string is a legitimate input.
    trace = results.read_csv("")
    assert len(trace) == 0
    assert trace.names == []


def test_ragged_csv_is_rejected_with_a_clear_message():
    with pytest.raises(results.ResultError, match="header"):
        results.read_csv("t,a,b\n1,2,3\n4,5\n")


def test_non_numeric_csv_is_rejected():
    with pytest.raises(results.ResultError, match="non-numeric"):
        results.read_csv("t,a\n0,oops\n")


def test_read_csv_file_matches_read_csv():
    from_file = results.read_csv_file(FIXTURES / "ramp.csv")
    from_text = results.read_csv(fixture_text("ramp.csv"))
    assert from_file.names == from_text.names
    assert from_file.time == from_text.time


def test_iter_csv_rows_streams_without_buffering_the_trace():
    text = fixture_text("ramp.csv")
    rows = list(results.iter_csv_rows(text.splitlines()))
    trace = results.read_csv(text)
    assert len(rows) == len(trace)
    headers, first = rows[0]
    assert headers[0] == "t"
    assert first[0] == trace.time[0]
    # The streaming reader yields the same headers object every row, so a consumer can index
    # by column name without re-parsing.
    assert all(h is headers for h, _ in rows)


def test_uniform_vs_adaptive_step_detection():
    assert results.read_csv(fixture_text("ramp.csv")).is_uniformly_stepped()
    assert not results.read_csv(fixture_text("sine_adaptive.csv")).is_uniformly_stepped()


# ---------------------------------------------------------------------------
# kernel-sidecar SIMULATE_RESULT manifests
# ---------------------------------------------------------------------------


def write_manifest(result_dir: pathlib.Path, entries, extra=None):
    """The manifest.json result_writer.write() produces, with the same fields and ordering."""
    manifest = {
        "n_points": 4,
        "schema_version": 2,
        "signals": entries,
        "sim_id": "sim-1",
        "status": "ok",
        "t_end": 0.3,
        "t_start": 0.0,
    }
    manifest.update(extra or {})
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


SIGNAL_ENTRIES = [
    {
        "analysis": "transient",
        "block_id": "R1",
        "domain": "time",
        "dtype": "float32",
        "name": "v(out)",
        "port_id": "p2",
        "quantity": "voltage",
        "shape": [4],
        "unit": "V",
        "url": "results/sim-1/signals/R1__p2.npy",
    }
]


def test_read_manifest_npy_layout(tmp_path):
    np = pytest.importorskip("numpy", reason="the .npy layout needs the [sidecar] extra")
    result_dir = tmp_path / "results" / "sim-1"
    (result_dir / "signals").mkdir(parents=True)
    np.save(result_dir / "time.npy", np.array([0.0, 0.1, 0.2, 0.3]))
    np.save(result_dir / "signals" / "R1__p2.npy", np.array([0.0, 1.0, 2.0, 3.0], dtype="float32"))
    write_manifest(result_dir, SIGNAL_ENTRIES)

    trace = results.read_manifest(result_dir)
    assert trace.names == ["v(out)"]
    assert trace.time == [0.0, 0.1, 0.2, 0.3]
    assert trace["v(out)"].values == [0.0, 1.0, 2.0, 3.0]
    # Manifest metadata the CSV format has no place for survives the read.
    assert trace["v(out)"].unit == "V"
    assert trace["v(out)"].metadata["quantity"] == "voltage"
    assert trace["v(out)"].metadata["block_id"] == "R1"
    assert trace.metadata["sim_id"] == "sim-1"
    assert trace.metadata["schema_version"] == 2
    assert "signals" not in trace.metadata

    # And it feeds measurement like any other trace: a 0->3 ramp over 0.3s integrates to 0.45.
    from gsim_tools import measure

    assert measure.integ(trace.samples("v(out)")) == pytest.approx(0.45)


def test_read_manifest_hdf5_layout(tmp_path):
    h5py = pytest.importorskip("h5py", reason="the data.h5 layout needs the [sidecar] extra")
    np = pytest.importorskip("numpy")
    result_dir = tmp_path / "results" / "sim-1"
    result_dir.mkdir(parents=True)
    with h5py.File(result_dir / "data.h5", "w") as fh:
        # Same layout hdf5_store.write_hdf5 produces: a `time` dataset plus one
        # `signals/<block_id>__<port_id>` dataset per signal.
        fh.attrs["sim_id"] = "sim-1"
        fh.create_dataset("time", data=np.array([0.0, 0.1, 0.2, 0.3]))
        fh.create_dataset(
            "signals/R1__p2", data=np.array([0.0, 1.0, 2.0, 3.0], dtype="float32")
        )
    write_manifest(result_dir, SIGNAL_ENTRIES)

    trace = results.read_manifest(result_dir)
    assert trace.time == [0.0, 0.1, 0.2, 0.3]
    assert trace["v(out)"].values == [0.0, 1.0, 2.0, 3.0]
    # Passing the manifest itself rather than its directory works too.
    assert results.read_manifest(result_dir / "manifest.json").names == trace.names


def test_read_manifest_carries_op_and_ac_extras(tmp_path):
    np = pytest.importorskip("numpy")
    result_dir = tmp_path / "results" / "sim-1"
    (result_dir / "signals").mkdir(parents=True)
    np.save(result_dir / "time.npy", np.array([0.0, 0.1]))
    write_manifest(
        result_dir,
        [],
        extra={
            "dc_op_table": [
                {"node_or_branch": "out", "quantity": "voltage", "unit": "V", "value": 2.5}
            ],
            "bode_margins": {"phase_margin_deg": 47.0},
        },
    )
    trace = results.read_manifest(result_dir)
    assert trace.metadata["dc_op_table"][0]["value"] == 2.5
    assert trace.metadata["bode_margins"]["phase_margin_deg"] == 47.0


def test_missing_manifest_is_a_clear_error(tmp_path):
    with pytest.raises(results.ResultError, match="no manifest.json"):
        results.read_manifest(tmp_path)


# ---------------------------------------------------------------------------
# Running the real binary
# ---------------------------------------------------------------------------


def simulator_or_skip() -> pathlib.Path:
    try:
        return results.find_simulator()
    except results.ResultError as exc:
        pytest.skip(str(exc))


def test_find_simulator_error_names_the_build_command(monkeypatch):
    monkeypatch.delenv("GSIM_SIMULATOR", raising=False)
    monkeypatch.setattr(results.shutil, "which", lambda _: None)
    with pytest.raises(results.ResultError, match="cargo build --release"):
        results.find_simulator()


@pytest.mark.simulator
def test_run_simulator_reproduces_a_committed_fixture(tmp_path):
    """A live run must produce exactly the CSV and log the committed fixture holds.

    This is what keeps the fixtures honest: if the simulator's output ever changes shape, this
    fails rather than the fixtures quietly describing a version of the tool that no longer
    exists.
    """
    simulator_or_skip()
    netlist = tmp_path / "ramp.cir"
    netlist.write_text(fixture_text("ramp.cir"))

    run = results.run_simulator(netlist, mode="transient", tfinal=10, dt=0.01)
    assert run.ok
    assert run.errors == []
    assert run.stdout == fixture_text("ramp.csv")
    assert run.trace.names == results.read_csv(fixture_text("ramp.csv")).names
    assert run.measurements == logs.parse_measurement_log(fixture_text("ramp.log"))
    # The .log the run wrote sits next to the netlist, exactly where measure/_lib.py expects.
    assert (tmp_path / "ramp.log").exists()


@pytest.mark.simulator
def test_run_simulator_surfaces_a_failure_as_a_typed_error(tmp_path):
    simulator_or_skip()
    netlist = tmp_path / "missing.cir"
    with pytest.raises(results.ResultError, match="exited 1"):
        results.run_simulator(netlist, mode="transient", tfinal=1)

    # With check=False the failure is data, not an exception, and it is classified.
    run = results.run_simulator(netlist, mode="transient", tfinal=1, check=False)
    assert not run.ok
    assert run.errors[0].kind is logs.ErrorKind.IO
    assert run.errors[0].fields["path"].endswith("missing.cir")


@pytest.mark.simulator
def test_run_netlist_text_runs_an_in_memory_netlist():
    simulator_or_skip()
    run = results.run_netlist_text(
        fixture_text("sine.cir"), stem="sine", mode="transient", tfinal=0.2, dt=0.0001
    )
    assert run.ok
    # The measurement log was read before the scratch directory was removed.
    assert run.measurements["MEAS_FOUR_h1_mag"] == pytest.approx(2.0, abs=5e-3)
    assert len(run.trace) == 2000


@pytest.mark.simulator
def test_stale_measurement_log_cannot_leak_into_a_later_run(tmp_path):
    """run_simulator deletes <stem>.log first, the same discipline measure/_lib.py follows."""
    simulator_or_skip()
    netlist = tmp_path / "square.cir"
    netlist.write_text(fixture_text("square.cir"))
    (tmp_path / "square.log").write_text("STALE = 123.0\n")

    run = results.run_simulator(netlist, mode="transient", tfinal=4.0, dt=0.001)
    assert "STALE" not in run.measurements
    assert run.measurements["MEAS_FREQ"] == pytest.approx(1.0, abs=0.01)
