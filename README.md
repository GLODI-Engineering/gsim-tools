# gsim-tools

Pure-Python tooling for the [`general-simulator`](../general-simulator) /
[`gsim-core`](../gsim-core) ecosystem. Three things:

- **`gsim_tools.results`** — read simulation output back into Python, from either real format
  that exists today: `general-simulator`'s own CSV (including its streaming transient output),
  and `kernel-sidecar`'s `SIMULATE_RESULT` manifest directories.
- **`gsim_tools.measure`** — real signal measurements (RMS, THD, average power, peak, mean,
  frequency/duty cycle, threshold crossings, error norms) over non-uniformly-sampled waveforms.
  A port of the `gs-waveform-measurements` Rust crate that `general-simulator`'s own
  `kind=measure` blocks are built on, cross-validated against it in `tests/test_measure_parity.py`.
- **`gsim_tools.logs`** — structured parsing of `general-simulator`'s `error:` stderr
  diagnostics and its `kind=measure` log files.
- **`gsim_tools.schematic`** — an ergonomic layer over `gsim_py` for *building* `.gsim_sch`
  schematics and compiling them to netlists.

## Install

```bash
pip install -e .              # results + measure + logs, stdlib only, no Rust toolchain
pip install -e '.[sidecar]'   # adds numpy/h5py, for kernel-sidecar SIMULATE_RESULT manifests
```

`gsim_tools.schematic` additionally needs `gsim_py`, a local PyO3 extension that is **not on
PyPI** — build it from the sibling `gsim-core` repository into the same virtualenv:

```bash
cd ../gsim-core/crates/gsim-py
pip install maturin && maturin develop --release
```

Without it, `gsim_tools.schematic` raises `GsimPyNotAvailable` with these instructions on first
use; the other three modules are unaffected.

## Example

Build a schematic, run it through the real simulator, measure the result:

```python
from gsim_tools import measure, results
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

run = results.run_netlist_text(sch.compile(), mode="transient", tfinal=0.02, dt=1e-5)
print(measure.find_at(run.trace.samples("V(out)"), 1e-3))
```

### Layout

Coordinates need not be hand-picked. `Schematic` wraps all three of `gsim_py`'s layout
entry points:

```python
sch.auto_arrange()                  # layer the whole model over *all* its wires, in place
pos = sch.layered_layout(           # the same layering, but it hands the positions back
    bboxes={"IM": (80, 88)},        # real sizes where you know them
    skip_wires=[feedback.id],       # feedback makes cycles; layer the forward path only
    column_gap=90, row_gap=70,
)
sch.place_without_overlap("VSI", (900, 200))    # drop one block in near a point
```

`layered_layout` is a longest-path (Sugiyama-style) layering via Kahn's algorithm: columns run
left-to-right by signal-flow depth, blocks stack down within a column. It guarantees reading
order and no overlap — it does not minimise wire crossings, and cycle members stay wherever the
layering stalled, so a feedback loop is usually worth excluding via `skip_wires`.

Or, with no `gsim_py` anywhere, just read and measure an existing run:

```python
trace = results.read_csv_file("run.csv")
v = trace.samples("V(a)")
print(measure.rms(v), measure.thd(v, fundamental_freq=50.0, num_harmonics=5))
```

## Tests

```bash
python -m pytest tests -q     # pure-Python suite; needs neither gsim_py nor the simulator
```

Tests marked `simulator` additionally need the `general-simulator` binary (on `PATH` or at
`$GSIM_SIMULATOR`); tests marked `gsim_py` need the extension built as above. Both are skipped
when unavailable. `examples/rc_lowpass_end_to_end.py` runs the whole pipeline and asserts a
known-correct numeric answer; see its own header for how to run it.
