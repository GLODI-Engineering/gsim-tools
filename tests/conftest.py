"""Shared fixture access for the gsim-tools test suite.

``tests/fixtures/`` holds, for each of six netlists, three files that all describe *the same
run*:

* ``<name>.cir`` -- copied verbatim from ``general-simulator/tests/measure/fixtures/``. Each
  one's own header comment carries the hand-derivation of every value it measures.
* ``<name>.csv`` -- that netlist's ``--format csv`` stdout, produced by running the real
  ``general-simulator`` release binary with the arguments in :data:`RUNS` below.
* ``<name>.log`` -- the ``kind=measure`` log the *same invocation* wrote. This is the
  simulator's own answer, computed in Rust by ``gs-waveform-measurements``.

Having the CSV and the log come from one invocation is the whole point: it lets
``test_measure_parity.py`` feed the CSV to :mod:`gsim_tools.measure` and assert the Python
answer matches the log, rather than asserting the Python port against numbers someone typed in.
"""

from __future__ import annotations

import pathlib

import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

#: The exact CLI arguments each fixture's ``.csv``/``.log`` pair was produced with, so the
#: fixtures can be regenerated. ``sine_adaptive`` has no ``--dt`` on purpose -- it is the
#: adaptive-stepping (genuinely non-uniformly sampled) case.
RUNS: dict[str, list[str]] = {
    "sine": ["--mode", "transient", "--tfinal", "0.2", "--dt", "0.0001"],
    "ramp": ["--mode", "transient", "--tfinal", "10", "--dt", "0.01"],
    "square": ["--mode", "transient", "--tfinal", "4.0", "--dt", "0.001"],
    "two_tone": ["--mode", "transient", "--tfinal", "0.1", "--dt", "0.0001"],
    "err": ["--mode", "transient", "--tfinal", "10", "--dt", "0.01"],
    "sine_adaptive": ["--mode", "transient", "--tfinal", "0.2"],
}


def fixture_text(name: str) -> str:
    """The raw text of ``tests/fixtures/<name>``."""
    return (FIXTURES / name).read_text()


@pytest.fixture(scope="session")
def fixtures_dir() -> pathlib.Path:
    return FIXTURES
