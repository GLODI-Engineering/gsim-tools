"""End-to-end proof that this library's whole pipeline works, against known-correct answers.

Every stage is real -- no mocks, no fixtures, no pre-recorded output:

1. **Build** an RC low-pass with :mod:`gsim_tools.schematic`, which calls into ``gsim_py``,
   which calls into ``gsim-core``, for every document operation.
2. **Compile** it to a ``general-simulator`` netlist through the same path.
3. **Run** the real ``general-simulator`` binary on it via :func:`gsim_tools.results.run_netlist_text`.
4. **Read** the CSV back with :mod:`gsim_tools.results` and **measure** it with
   :mod:`gsim_tools.measure`.
5. **Assert** each measurement against (a) the closed-form analytic answer for a first-order
   RC step response and (b) ``general-simulator``'s own ``kind=measure`` result for the same
   run -- so the Python port is checked against the Rust implementation on the same samples,
   not just against itself.

The circuit is a 5 V DC source behind R = 1 kohm into C = 1 uF, so tau = RC = 1 ms and the
capacitor voltage is the textbook v(t) = V*(1 - exp(-t/tau)). Every expected value below is
that formula evaluated by hand:

    v(tau)          = 5*(1 - 1/e)                = 3.160602794... V
    10%-90% rise    = tau * ln(9)                = 2.197224577... ms
    mean over 5 tau = V*(1 - (1 - e^-5)/5)       = 4.006... V (dt-weighted, see below)

Run it::

    # 1. build gsim_py into a virtualenv (from the sibling gsim-core repo)
    cd ../gsim-core/crates/gsim-py && maturin develop --release
    # 2. install gsim-tools into the same virtualenv
    pip install -e ../gsim-tools
    # 3. point at the general-simulator binary and go
    GSIM_SIMULATOR=../general-simulator/target/release/general-simulator \\
        python examples/rc_lowpass_end_to_end.py
"""

import math
import sys

from gsim_tools import measure, results
from gsim_tools.schematic import Schematic

V_SUPPLY = 5.0
R = 1_000.0
C = 1e-6
TAU = R * C  # 1 ms
T_FINAL = 5 * TAU
DT = 1e-6

# The kind=measure declarations to cross-check against. These are appended to the compiled
# netlist as plain text: they are a general-simulator netlist feature, not a .gsim_sch document
# feature, so the schematic layer has no business emitting them -- but the whole point of this
# script is to compare the two implementations, and this is how you ask the simulator for its
# own answer.
MEASURE_BLOCK = f"""
MEAS_VTAU kind=measure type=find out=V(out) at={TAU}
MEAS_RISE kind=measure type=trig_targ trig_var=V(out) trig_val={0.1 * V_SUPPLY} trig_rise=1 \
targ_var=V(out) targ_val={0.9 * V_SUPPLY} targ_rise=1
MEAS_AVG kind=measure type=avg out=V(out)
MEAS_RMS kind=measure type=rms out=V(out)
MEAS_MAX kind=measure type=max out=V(out)
"""


def build() -> Schematic:
    """The RC low-pass, built entirely through gsim_py."""
    sch = Schematic("rc_lowpass", "RC low-pass step response")

    vin = sch.add("source.voltage.dc", id="Vin", at=(0, 0), V=V_SUPPLY)
    r1 = sch.add(
        "passive.resistor",
        id="R1",
        at=(60, 0),
        R=R,
        comment_before=f"series element -- with C1 gives tau = {TAU} s",
    )
    c1 = sch.add("passive.capacitor", id="C1", at=(120, 0), C=C)
    gnd = sch.add("routing.ground", id="gnd", at=(60, 80))

    # `net=` names the node, so the CSV column is V(out) rather than a generated name. Wire
    # kind is inferred: every one of these blocks is physical-network, so every wire is
    # physical-electrical without saying so.
    sch.connect(vin.p, r1.p1, net="in")
    sch.connect(r1.p2, c1.p1, net="out")
    sch.connect(c1.p2, gnd.p)
    sch.connect(vin.n, gnd.p)
    return sch


def check(label, got, expected, tol, source):
    delta = abs(got - expected)
    status = "ok  " if delta <= tol else "FAIL"
    print(f"  {status} {label:<28} {got:>18.12g}   vs {source} {expected:>18.12g}  "
          f"(|diff| {delta:.3g}, tol {tol:g})")
    return delta <= tol


def main() -> int:
    sch = build()
    print(f"built {sch}")

    netlist = sch.compile() + MEASURE_BLOCK
    print(f"compiled to {len(netlist.splitlines())} netlist lines")

    run = results.run_netlist_text(
        netlist, stem="rc_lowpass", mode="transient", tfinal=T_FINAL, dt=DT
    )
    print(f"ran: {len(run.trace)} points, signals {run.trace.names}")
    if run.measurements == {}:
        print("no kind=measure results came back -- the cross-check cannot run", file=sys.stderr)
        return 1

    out = run.trace.samples("V(out)")
    current = run.trace.samples("I(Vin)")
    sim = run.measurements
    ok = True

    # --- 1. against the closed-form answers -------------------------------------------
    print("\nagainst the analytic first-order step response:")
    # v(tau) = V*(1 - 1/e). Tolerance is the trapezoidal/backward-Euler discretization error
    # at dt = tau/1000, not machine epsilon.
    ok &= check(
        "v(tau) [find_at]",
        measure.find_at(out, TAU),
        V_SUPPLY * (1 - math.exp(-1.0)),
        1e-5,
        "analytic",
    )
    # 10%-90% rise time of a first-order step = tau * ln(9), exactly.
    rise = measure.trig_targ(
        measure.EventSpec.crossing(out, 0.1 * V_SUPPLY, edge=measure.Edge.RISING),
        measure.EventSpec.crossing(out, 0.9 * V_SUPPLY, edge=measure.Edge.RISING),
    )
    ok &= check("10%-90% rise [trig_targ]", rise, TAU * math.log(9.0), 1e-8, "analytic")
    # The same rise time via FRAC_MAX, which derives the 10%/90% levels from the trace's own
    # peak instead of being told them -- a different code path. It is deliberately NOT compared
    # to tau*ln(9): the run stops at 5*tau, so the peak is v(5*tau) = 4.966 V, not the
    # asymptote 5 V, and the two levels shift down with it. Asserting tau*ln(9) here would be
    # asserting the wrong number and papering over it with a loose tolerance; the right
    # closed form for levels 0.1*peak and 0.9*peak is tau*ln((V-0.1*peak)/(V-0.9*peak)).
    peak = measure.max_(out).value
    ok &= check(
        "10%-90% rise [frac_max]",
        measure.trig_targ(
            measure.EventSpec.frac_max(out, 0.1, edge=measure.Edge.RISING),
            measure.EventSpec.frac_max(out, 0.9, edge=measure.Edge.RISING),
        ),
        TAU * math.log((V_SUPPLY - 0.1 * peak) / (V_SUPPLY - 0.9 * peak)),
        1e-8,
        "analytic",
    )
    # The dt-weighted mean of V*(1-e^{-t/tau}) over [0, 5*tau] is V*(1 - (1-e^-5)/5).
    ok &= check(
        "mean over 5 tau [avg]",
        measure.avg(out),
        V_SUPPLY * (1.0 - (1.0 - math.exp(-5.0)) / 5.0),
        1e-3,
        "analytic",
    )

    # --- 2. against general-simulator's own kind=measure ------------------------------
    # The important half: the same samples, measured by this Python port and by the Rust
    # gs-waveform-measurements crate the simulator uses. These must agree to ~machine
    # precision, since only the summation order differs.
    print("\nagainst general-simulator's own kind=measure on the same run:")
    ok &= check("find_at(tau)", measure.find_at(out, TAU), sim["MEAS_VTAU"], 1e-12, "kind=measure")
    ok &= check("trig_targ rise", rise, sim["MEAS_RISE"], 1e-12, "kind=measure")
    ok &= check("avg", measure.avg(out), sim["MEAS_AVG"], 1e-9, "kind=measure")
    ok &= check("rms", measure.rms(out), sim["MEAS_RMS"], 1e-9, "kind=measure")
    ok &= check("max", measure.max_(out).value, sim["MEAS_MAX"], 1e-12, "kind=measure")

    # --- 3. a measurement with no kind=measure counterpart ----------------------------
    # average_power is composed in Python from the ported primitives, so there is no
    # kind=measure to compare it to -- but it does have an analytic answer. I(Vin) is the
    # current into the source's positive terminal, so the source *delivering* power reads
    # negative; the energy the resistor dissipates over [0, 5*tau] is
    #   integral of (V^2/R) e^{-2t/tau} dt = (V^2 tau)/(2R) * (1 - e^-10),
    # and dividing by the 5*tau window gives the mean.
    print("\nmeasurements composed in Python (no kind=measure counterpart):")
    expected_mean_r_power = (
        (V_SUPPLY**2 * TAU) / (2 * R) * (1 - math.exp(-10.0)) / T_FINAL
    )
    v_r = [
        measure.Sample(a.t, a.v - b.v)
        for a, b in zip(run.trace.samples("V(in)"), out)
    ]
    i_r = [measure.Sample(s.t, -s.v) for s in current]  # into the resistor, not into Vin
    ok &= check(
        "mean resistor power",
        measure.average_power(v_r, i_r),
        expected_mean_r_power,
        1e-3,
        "analytic",
    )

    print("\nPASS -- schematic -> netlist -> simulator -> results -> measure, end to end."
          if ok else "\nFAIL -- see the lines marked FAIL above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
