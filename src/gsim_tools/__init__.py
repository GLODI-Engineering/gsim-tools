"""gsim-tools -- build schematics in Python, run them, and measure what comes back.

The point of this package is the first of those: **programmatic schematic
creation**. A circuit or control diagram is an object you construct, wire up,
validate and compile to a netlist, rather than a file you draw by hand.

    from gsim_tools.schematic import Schematic

    sch = Schematic("rc", "RC low-pass")
    vin = sch.add("source.voltage.dc", id="Vin", at=(0, 0), V=5.0)
    r1 = sch.add("passive.resistor", id="R1", at=(60, 0), R=1000.0)
    c1 = sch.add("passive.capacitor", id="C1", at=(120, 0), C=1e-6)
    gnd = sch.add("routing.ground", id="gnd", at=(60, 80))
    sch.connect(vin.p, r1.p1)
    sch.connect(r1.p2, c1.p1)
    sch.connect(c1.p2, gnd.p)
    sch.connect(vin.n, gnd.p)

    netlist = sch.compile()

Validation happens as you build: a wire to a port that does not exist, a
signal port fed by two sources, a connection across incompatible paradigms or
dimensions each raise their own exception rather than producing a netlist that
fails later for a reason you have to work backwards from.

The modules
-----------

* :mod:`gsim_tools.schematic` -- build, validate and compile schematics. This
  is the main surface, and the reason the package exists.
* :mod:`gsim_tools.results` -- read simulation output back, from either format
  that exists: ``general-simulator``'s own CSV (including streaming transient
  output) and ``kernel-sidecar``'s ``SIMULATE_RESULT`` manifest directories.
* :mod:`gsim_tools.measure` -- RMS, THD, power, peak, mean, frequency,
  crossings and error norms over *non-uniformly sampled* waveforms, which is
  what a variable-step solver actually produces.
* :mod:`gsim_tools.logs` -- structured parsing of ``general-simulator``'s
  ``error:`` diagnostics and its ``kind=measure`` log files.

What needs what
---------------

``results``, ``measure`` and ``logs`` are stdlib-only: ``pip install
gsim-tools`` is enough, no compiler and no Rust toolchain. Reading a
kernel-sidecar manifest additionally wants ``numpy``/``h5py`` -- the
``sidecar`` extra.

``schematic`` needs ``gsim_py``, the PyO3 extension built from ``gsim-core``
-- not only to compile, but from the moment you construct a
:class:`~gsim_tools.schematic.Schematic`: the document model, its validation
rules and the compile registry all live in the Rust core, and this module is
the ergonomic Python layer over them rather than a reimplementation.

Importing ``gsim_tools`` never requires it, so the measurement and result
modules work on a machine that has no Rust toolchain. The first call that
needs the extension raises
:class:`~gsim_tools.schematic.GsimPyNotAvailable` naming the build step. See
the README for installing it.
"""

from . import logs, measure, results, schematic
from .logs import ErrorKind, SimulatorError, parse_measurement_log, parse_stderr
from .measure import Sample, Window, avg, four, peak, rms, thd
from .results import Signal, Trace, read_csv, read_csv_file, read_manifest, run_simulator
from .schematic import (
    Block,
    CompileResult,
    Endpoint,
    GsimPyNotAvailable,
    Schematic,
    SchematicError,
    Wire,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Block",
    "CompileResult",
    "Endpoint",
    "ErrorKind",
    "GsimPyNotAvailable",
    "Sample",
    "Schematic",
    "SchematicError",
    "Signal",
    "SimulatorError",
    "Trace",
    "Window",
    "Wire",
    "avg",
    "four",
    "logs",
    "measure",
    "parse_measurement_log",
    "parse_stderr",
    "peak",
    "read_csv",
    "read_csv_file",
    "read_manifest",
    "results",
    "rms",
    "run_simulator",
    "schematic",
    "thd",
]
