"""gsim-tools -- pure-Python tooling for the general-simulator / gsim-core ecosystem.

Three things, deliberately separable:

* :mod:`gsim_tools.results` -- read simulation output back into Python, from either real format
  that exists today: ``general-simulator``'s own CSV (including its streaming transient
  output), and ``kernel-sidecar``'s ``SIMULATE_RESULT`` manifest directories.
* :mod:`gsim_tools.measure` -- real signal measurements (RMS, THD, power, peak, mean, frequency,
  crossings, error norms) over non-uniformly-sampled waveforms, ported from the
  ``gs-waveform-measurements`` Rust crate that ``general-simulator``'s own ``kind=measure``
  blocks are built on.
* :mod:`gsim_tools.logs` -- structured parsing of ``general-simulator``'s ``error:`` stderr
  diagnostics and its ``kind=measure`` log files.

Those three are stdlib-only and need no Rust toolchain: ``pip install gsim-tools`` is enough.
(Reading a kernel-sidecar manifest additionally needs ``numpy``/``h5py``, the ``sidecar`` extra.)

* :mod:`gsim_tools.schematic` -- an ergonomic layer over ``gsim_py`` for *building* schematics
  and compiling them to netlists. ``gsim_py`` is a local Rust extension built with ``maturin``
  from the sibling ``gsim-core`` repository, so it is optional: this module is not imported
  here, and using it without ``gsim_py`` raises one clear error explaining how to build it.

Quick start::

    from gsim_tools import measure, results

    trace = results.read_csv_file("run.csv")
    v = trace.samples("V(out)")
    print(measure.rms(v), measure.thd(v, fundamental_freq=50.0, num_harmonics=5))
"""

from . import logs, measure, results
from .logs import ErrorKind, SimulatorError, parse_measurement_log, parse_stderr
from .measure import Sample, Window, avg, four, peak, rms, thd
from .results import Signal, Trace, read_csv, read_csv_file, read_manifest, run_simulator

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # submodules -- `schematic` is intentionally absent: importing it here would make the
    # whole package's import path depend on a module whose only purpose needs gsim_py.
    "logs",
    "measure",
    "results",
    # the handful of names worth having at the top level
    "ErrorKind",
    "Sample",
    "Signal",
    "SimulatorError",
    "Trace",
    "Window",
    "avg",
    "four",
    "parse_measurement_log",
    "parse_stderr",
    "peak",
    "read_csv",
    "read_csv_file",
    "read_manifest",
    "rms",
    "run_simulator",
    "thd",
]
