"""Reading simulation results back into Python, from the two real formats that exist today.

Two different pipelines in this project write results in two different shapes, and this module
reads both into one :class:`Trace` type that :mod:`gsim_tools.measure` can measure directly:

1. **``general-simulator``'s own CSV** -- what the CLI prints to stdout (or writes with
   ``--format csv --out``): a header row ``t,V(node),I(branch),<block names>`` followed by one
   comma-joined row per point, each value rendered with Rust's ``f64`` ``Display``. Read with
   :func:`read_csv` / :func:`read_csv_file`, or streamed line-by-line with
   :func:`iter_csv_rows`. Ported from ``general-simulator/tests/raw-output-python/_lib.py``'s
   ``parse_csv`` and ``run``/``run_csv_and_raw``.

2. **``kernel-sidecar``'s ``SIMULATE_RESULT`` manifest** -- what ``general-simulator-gui``'s
   live pipeline writes: a result directory holding ``manifest.json`` plus a single ``data.h5``
   (schema_version 2), or, in the pre-BE-52 layout, ``time.npy`` plus one ``.npy`` per signal.
   Read with :func:`read_manifest`. The manifest shape is taken from
   ``general-simulator-gui/packages/simulator/src/simulator/postprocess/result_writer.py`` and
   the HDF5 layout from its ``hdf5_store.py`` (``time`` dataset, ``signals/<block>__<port>``).

These really are two formats from two pipelines, not one format described twice -- see
``gsim-core/docs/decisions/0003-general-simulator-gui-pipeline-parity.md``. This module invents
no third format; it just reads both.

Dependencies: the CSV half is stdlib-only. The manifest half needs ``numpy`` (and ``h5py`` for
a ``data.h5``-backed result), which are declared as the optional ``sidecar`` extra and imported
lazily, so ``pip install gsim-tools`` with nothing else still gets you CSV reading and every
measurement in :mod:`gsim_tools.measure`.
"""

from __future__ import annotations

import csv
import io
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional, Sequence, Union

from .measure import Sample

__all__ = [
    "Signal",
    "Trace",
    "SimulationRun",
    "ResultError",
    "read_csv",
    "read_csv_file",
    "iter_csv_rows",
    "read_manifest",
    "find_simulator",
    "run_simulator",
]

PathLike = Union[str, "os.PathLike[str]"]


class ResultError(Exception):
    """A result file could not be read, or a simulator run failed."""


@dataclass(frozen=True)
class Signal:
    """One named waveform sharing its trace's time base.

    ``metadata`` carries whatever the source format knew about the signal and the CSV format
    does not (``quantity``, ``unit``, ``block_id``, ``port_id``, ``analysis``, ``domain`` for
    a kernel-sidecar manifest); it is empty for a CSV-sourced signal.
    """

    name: str
    time: Sequence[float]
    values: Sequence[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.values)

    def samples(self) -> list[Sample]:
        """This signal as a ``Sample`` list, ready for :mod:`gsim_tools.measure`."""
        return [Sample(float(t), float(v)) for t, v in zip(self.time, self.values)]

    @property
    def unit(self) -> str:
        return str(self.metadata.get("unit", ""))


@dataclass
class Trace:
    """A set of named signals sharing one time base -- one simulation run's waveform output.

    Index it by signal name (``trace["V(out)"]``) to get a :class:`Signal`, or call
    :meth:`samples` to go straight to a ``Sample`` list for a measurement.
    """

    time: Sequence[float]
    signals: "dict[str, Signal]"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        """Signal names, in the source's own column order."""
        return list(self.signals)

    def __len__(self) -> int:
        return len(self.time)

    def __contains__(self, name: str) -> bool:
        return name in self.signals

    def __iter__(self) -> Iterator[Signal]:
        return iter(self.signals.values())

    def __getitem__(self, name: str) -> Signal:
        try:
            return self.signals[name]
        except KeyError:
            raise KeyError(
                f"no signal {name!r} in this trace; available: {', '.join(self.names)}"
            ) from None

    def samples(self, name: str) -> list[Sample]:
        """The named signal as a ``Sample`` list, ready for :mod:`gsim_tools.measure`."""
        return self[name].samples()

    def is_uniformly_stepped(self, tolerance: float = 1e-9) -> bool:
        """Whether every time step is the same size to within ``tolerance`` (relative).

        Useful as a sanity check: an adaptive-stepping run is *expected* to be non-uniform,
        and every measurement in :mod:`gsim_tools.measure` is written to handle that, so this
        is informational rather than a precondition for anything.
        """
        if len(self.time) < 3:
            return True
        steps = [b - a for a, b in zip(self.time, self.time[1:])]
        lo, hi = min(steps), max(steps)
        return hi - lo <= tolerance * max(abs(hi), 1e-300)


# ---------------------------------------------------------------------------
# general-simulator's own CSV output
# ---------------------------------------------------------------------------


def _trace_from_columns(headers: Sequence[str], rows: Sequence[Sequence[float]]) -> Trace:
    time = [row[0] for row in rows]
    signals = {
        name: Signal(name=name, time=time, values=[row[i] for row in rows])
        for i, name in enumerate(headers)
        if i > 0
    }
    return Trace(time=time, signals=signals, metadata={"time_column": headers[0]})


def read_csv(text: str) -> Trace:
    """Parse ``general-simulator``'s CSV output text into a :class:`Trace`.

    The first column is always the time/sweep column (header ``t``); every remaining column
    becomes a signal named by its header (``V(node)``, ``I(branch)``, or a ``kind=`` block's
    own name). Ported from ``raw-output-python/_lib.py``'s ``parse_csv``, which parses exactly
    this into ``(headers, rows)``.

    An empty string yields an empty trace -- the CLI's ``print_csv`` prints nothing at all,
    not even a header, for a run that resolved no points.
    """
    reader = csv.reader(io.StringIO(text))
    rows = [row for row in reader if row]
    if not rows:
        return Trace(time=[], signals={}, metadata={"time_column": "t"})
    headers = rows[0]
    try:
        data = [[float(v) for v in row] for row in rows[1:]]
    except ValueError as exc:
        raise ResultError(f"non-numeric value in CSV output: {exc}") from exc
    for i, row in enumerate(data):
        if len(row) != len(headers):
            raise ResultError(
                f"CSV row {i + 2} has {len(row)} values but the header has {len(headers)}"
            )
    return _trace_from_columns(headers, data)


def read_csv_file(path: PathLike) -> Trace:
    """Read a ``general-simulator`` CSV file (``--format csv --out <path>``) into a Trace."""
    return read_csv(pathlib.Path(path).read_text())


def iter_csv_rows(lines: Iterable[str]) -> Iterator[tuple[list[str], list[float]]]:
    """Stream ``general-simulator``'s CSV output row by row, never holding the whole trace.

    Yields ``(headers, row_values)`` for every data row. This is the counterpart to the CLI's
    own streaming transient path (``run_transient_streamed`` in
    ``general-simulator-cli/src/main.rs``), which writes CSV rows to stdout as it solves them
    -- so a long run can be measured incrementally or filtered without buffering it all.

    Example -- read a run as it happens::

        proc = subprocess.Popen([sim, "net.cir", "--mode", "transient", "--tfinal", "10"],
                                stdout=subprocess.PIPE, text=True)
        for headers, row in iter_csv_rows(proc.stdout):
            ...
    """
    headers: Optional[list[str]] = None
    for raw in lines:
        raw = raw.rstrip("\r\n")
        if not raw:
            continue
        fields = next(csv.reader([raw]))
        if headers is None:
            headers = fields
            continue
        try:
            values = [float(v) for v in fields]
        except ValueError as exc:
            raise ResultError(f"non-numeric value in CSV row {fields!r}: {exc}") from exc
        yield headers, values


# ---------------------------------------------------------------------------
# kernel-sidecar's SIMULATE_RESULT manifest
# ---------------------------------------------------------------------------


def _dataset_key(block_id: str, port_id: str) -> str:
    """``<block_id>__<port_id>`` -- the HDF5 dataset name / legacy .npy filename stem.

    Mirrors ``hdf5_store.dataset_key``; the manifest's ``url`` field ends in exactly this
    plus ``.npy``, whether or not a ``.npy`` file physically exists.
    """
    return f"{block_id}__{port_id}"


def read_manifest(path: PathLike) -> Trace:
    """Read a ``kernel-sidecar`` ``SIMULATE_RESULT`` result directory into a :class:`Trace`.

    ``path`` may be the result directory itself or its ``manifest.json``. Both storage layouts
    the GUI pipeline has used are read:

    * **schema_version 2 (current)** -- a single ``data.h5`` holding ``time`` and
      ``signals/<block_id>__<port_id>`` datasets (see ``hdf5_store.py``). Requires ``h5py``.
    * **the pre-BE-52 layout** -- ``time.npy`` plus one ``signals/<key>.npy`` per signal, the
      files the manifest's ``url`` fields literally name. Requires only ``numpy``.

    Every manifest field the writer attaches travels through to
    ``Trace.metadata`` (``sim_id``, ``t_start``, ``t_end``, ``n_points``, ``status``,
    ``schema_version``, and ``dc_op_table``/``bode_margins`` when the analysis produced them);
    per-signal ``analysis``/``quantity``/``unit``/``domain``/``block_id``/``port_id`` land in
    each :class:`Signal`'s ``metadata``.
    """
    p = pathlib.Path(path)
    manifest_path = p / "manifest.json" if p.is_dir() else p
    result_dir = manifest_path.parent
    if not manifest_path.exists():
        raise ResultError(f"no manifest.json at {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("signals", [])

    h5_path = result_dir / "data.h5"
    if h5_path.exists():
        time, columns = _read_hdf5(h5_path, entries)
    else:
        time, columns = _read_npy_layout(result_dir, entries)

    signals: dict[str, Signal] = {}
    for entry in entries:
        key = _dataset_key(entry["block_id"], entry["port_id"])
        if key not in columns:
            continue
        name = entry.get("name") or key
        signals[name] = Signal(
            name=name,
            time=time,
            values=columns[key],
            metadata={
                k: entry[k]
                for k in (
                    "analysis",
                    "block_id",
                    "domain",
                    "port_id",
                    "quantity",
                    "swept_param",
                    "unit",
                )
                if k in entry
            },
        )

    metadata = {k: v for k, v in manifest.items() if k != "signals"}
    return Trace(time=time, signals=signals, metadata=metadata)


def _read_hdf5(
    h5_path: pathlib.Path, entries: Sequence[dict[str, Any]]
) -> tuple[list[float], dict[str, list[float]]]:
    try:
        import h5py  # noqa: PLC0415 - optional, only needed for the HDF5 layout
    except ImportError as exc:  # pragma: no cover - exercised only without h5py
        raise ResultError(
            f"{h5_path.name} is an HDF5-backed result, which needs h5py:\n"
            "    pip install 'gsim-tools[sidecar]'\n"
            "(CSV reading and every measurement work without it.)"
        ) from exc

    columns: dict[str, list[float]] = {}
    with h5py.File(h5_path, "r") as fh:
        time = [float(v) for v in fh["time"][()]]
        group = fh.get("signals")
        if group is None:
            return time, columns
        for entry in entries:
            key = _dataset_key(entry["block_id"], entry["port_id"])
            if key in group:
                columns[key] = [float(v) for v in group[key][()]]
    return time, columns


def _read_npy_layout(
    result_dir: pathlib.Path, entries: Sequence[dict[str, Any]]
) -> tuple[list[float], dict[str, list[float]]]:
    try:
        import numpy as np  # noqa: PLC0415 - optional, only needed for the .npy layout
    except ImportError as exc:  # pragma: no cover - exercised only without numpy
        raise ResultError(
            "reading a .npy-backed result directory needs numpy:\n"
            "    pip install 'gsim-tools[sidecar]'\n"
            "(CSV reading and every measurement work without it.)"
        ) from exc

    time_path = result_dir / "time.npy"
    if not time_path.exists():
        raise ResultError(
            f"{result_dir} has neither data.h5 nor time.npy -- not a result directory "
            "in either layout this reader knows"
        )
    time = [float(v) for v in np.load(time_path)]

    columns: dict[str, list[float]] = {}
    for entry in entries:
        key = _dataset_key(entry["block_id"], entry["port_id"])
        # The manifest's url is workdir-relative ("results/<sim_id>/signals/<key>.npy"); the
        # file itself sits under this result directory, so only its tail is used.
        url = entry.get("url", f"signals/{key}.npy")
        candidate = result_dir / pathlib.PurePosixPath(url).name
        signal_path = result_dir / "signals" / f"{key}.npy"
        for path in (signal_path, candidate):
            if path.exists():
                columns[key] = [float(v) for v in np.load(path)]
                break
    return time, columns


# ---------------------------------------------------------------------------
# Running the simulator (ported from raw-output-python/_lib.py and measure/_lib.py)
# ---------------------------------------------------------------------------


@dataclass
class SimulationRun:
    """Everything one ``general-simulator`` invocation produced.

    ``trace`` is the CSV output parsed (empty for a ``--format raw`` run); ``measurements`` is
    the ``<netlist stem>.log`` file's ``name = value`` lines parsed by
    :func:`gsim_tools.logs.parse_measurement_log`; ``errors`` is any ``error:`` line on stderr,
    parsed by :func:`gsim_tools.logs.parse_stderr`.
    """

    returncode: int
    stdout: str
    stderr: str
    log_text: str
    trace: Trace
    measurements: "dict[str, float]"
    errors: list[Any]

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def find_simulator(explicit: Optional[PathLike] = None) -> pathlib.Path:
    """Locate the ``general-simulator`` binary.

    Search order: ``explicit`` argument, the ``GSIM_SIMULATOR`` environment variable (the same
    variable ``gsim-core/examples/*.py`` already use), then ``general-simulator`` on ``PATH``.
    Raises ``ResultError`` naming the build command if none is found -- the same discipline as
    ``raw-output-python/_lib.py``'s ``cli_path()``.
    """
    for candidate in (explicit, os.environ.get("GSIM_SIMULATOR"), shutil.which("general-simulator")):
        if candidate:
            path = pathlib.Path(candidate)
            if path.exists():
                return path
    raise ResultError(
        "general-simulator binary not found. Build it from the general-simulator repo:\n"
        "    cargo build --release -p general-simulator-cli\n"
        "then set GSIM_SIMULATOR=<repo>/target/release/general-simulator, pass "
        "simulator=<path>, or put it on PATH."
    )


def run_simulator(
    netlist: PathLike,
    *,
    mode: str = "transient",
    tfinal: Optional[float] = None,
    dt: Optional[float] = None,
    devices: Optional[PathLike] = None,
    extra_args: Sequence[str] = (),
    simulator: Optional[PathLike] = None,
    check: bool = True,
    cwd: Optional[PathLike] = None,
) -> SimulationRun:
    """Run ``general-simulator`` on ``netlist`` and collect everything it produced.

    ``dt=None`` means adaptive stepping -- no ``--dt`` flag at all -- matching the CLI's own
    "omit --dt for adaptive step-size control" convention. Consolidated from
    ``raw-output-python/_lib.py``'s ``run``/``run_csv_and_raw`` (running the binary as a
    subprocess and parsing its CSV stdout) and ``measure/_lib.py``'s ``run_measure`` (deleting
    a stale ``<stem>.log`` first so a previous run's measurements cannot leak in, then reading
    the log the CLI wrote alongside the netlist).

    With ``check=True`` (the default) a nonzero exit raises ``ResultError`` carrying the
    parsed ``error:`` line. Note a *measurement's* own failure is not a nonzero exit -- it is
    reported per-line in the log file, so one bad measurement never aborts the others.
    """
    from .logs import parse_measurement_log, parse_stderr  # local: avoids a cycle

    binary = find_simulator(simulator)
    netlist_path = pathlib.Path(netlist)
    log_path = netlist_path.with_suffix(".log")
    # Delete a stale log first, exactly as measure/_lib.py's run_measure does, so a previous
    # run's measurements can't leak in if this run fails before writing its own.
    try:
        log_path.unlink()
    except FileNotFoundError:
        pass

    args = [str(binary), str(netlist_path)]
    if devices is not None:
        args += ["--devices", str(devices)]
    args += ["--mode", mode]
    if tfinal is not None:
        args += ["--tfinal", str(tfinal)]
    if dt is not None:
        args += ["--dt", str(dt)]
    args += list(extra_args)

    proc = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
    errors = parse_stderr(proc.stderr)
    if check and proc.returncode != 0:
        detail = errors[0].message if errors else proc.stderr.strip()
        raise ResultError(
            f"general-simulator exited {proc.returncode} on {netlist_path.name}: {detail}"
        )

    log_text = log_path.read_text() if log_path.exists() else ""
    return SimulationRun(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        log_text=log_text,
        trace=read_csv(proc.stdout),
        measurements=parse_measurement_log(log_text),
        errors=errors,
    )


def run_netlist_text(
    netlist_text: str,
    *,
    stem: str = "circuit",
    **kwargs: Any,
) -> SimulationRun:
    """Write ``netlist_text`` to a scratch directory and run it, cleaning up afterwards.

    The convenience wrapper around :func:`run_simulator` for a netlist held in memory -- e.g.
    one just produced by :meth:`gsim_tools.schematic.Schematic.compile`. The measurement log
    is read before the scratch directory is removed.
    """
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="gsim-tools-"))
    try:
        netlist_path = tmpdir / f"{stem}.cir"
        netlist_path.write_text(netlist_text)
        return run_simulator(netlist_path, **kwargs)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


__all__.append("run_netlist_text")
