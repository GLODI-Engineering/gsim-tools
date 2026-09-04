"""Cross-validation: every Python measurement against ``general-simulator``'s own ``kind=measure``.

:mod:`gsim_tools.measure` is a port of the Rust ``gs-waveform-measurements`` crate, and a port
is only worth anything if it produces the same numbers as the original. So for each fixture
this reads the CSV trace and the ``.log`` file that *one* real ``general-simulator`` invocation
produced (see ``conftest.py`` for how the pair is generated), recomputes each measurement in
Python from the CSV, and asserts it matches the log's value -- the simulator's own Rust answer
for the same samples.

The ``kind=measure`` declarations being replicated here are in each fixture's own ``.cir``, and
each ``.cir``'s header comment carries the hand-derivation of the expected value, so a failure
here has three independent things to check against: the Python result, the Rust result, and the
closed-form answer.

Tolerances are relative and tight (1e-9 by default). They are not zero because the two
implementations sum the same terms in a different order -- Python uses ``math.fsum`` where the
Rust uses a plain iterator ``sum`` -- so the last bit or two can differ; anything larger than
that would mean a genuine divergence in method.
"""

from __future__ import annotations

import math

import pytest

from conftest import fixture_text
from gsim_tools import logs, measure, results


def load(name: str):
    """The ``(trace, measurement_values)`` pair for one fixture, from one real run."""
    trace = results.read_csv(fixture_text(f"{name}.csv"))
    values = logs.parse_measurement_log(fixture_text(f"{name}.log"))
    return trace, values


def assert_matches(actual: float, expected: float, label: str, rel: float = 1e-9) -> None:
    assert actual == pytest.approx(expected, rel=rel, abs=1e-12), (
        f"{label}: gsim_tools.measure got {actual!r}, "
        f"general-simulator's kind=measure got {expected!r}"
    )


# ---------------------------------------------------------------------------
# ramp.cir -- max, min, max_at, min_at, pp, avg, rms, integ, deriv, find, when, trig_targ
# ---------------------------------------------------------------------------


def test_ramp_scalar_and_crossing_measurements_match_kind_measure():
    trace, expected = load("ramp")
    va = trace.samples("V(a)")
    vb = trace.samples("V(b)")

    assert_matches(measure.max_(va).value, expected["MEAS_MAX"], "max")
    assert_matches(measure.min_(va).value, expected["MEAS_MIN"], "min")
    assert_matches(measure.max_(va).time, expected["MEAS_MAXAT"], "max_at")
    assert_matches(measure.min_(va).time, expected["MEAS_MINAT"], "min_at")
    assert_matches(measure.peak_to_peak(va), expected["MEAS_PP"], "pp")
    assert_matches(measure.avg(va), expected["MEAS_AVG"], "avg")
    assert_matches(measure.rms(va), expected["MEAS_RMS"], "rms")
    assert_matches(measure.integ(va), expected["MEAS_INTEG"], "integ")

    # `at=5` forms.
    assert_matches(measure.deriv_at(va, 5.0), expected["MEAS_DERIVAT"], "deriv at=")
    assert_matches(measure.find_at(va, 5.0), expected["MEAS_FINDAT"], "find at=")

    # `when_val=10 rise=1` forms.
    rise1 = dict(edge=measure.Edge.RISING, occurrence=measure.Occurrence.nth(1))
    assert_matches(measure.when(va, 10.0, **rise1), expected["MEAS_WHEN"], "when")
    assert_matches(
        measure.find_when(vb, va, 10.0, **rise1), expected["MEAS_FINDWHEN"], "find when="
    )
    assert_matches(
        measure.deriv_when(va, va, 10.0, **rise1), expected["MEAS_DERIVWHEN"], "deriv when="
    )

    # trig_at=1, targ V(a)=15 rising.
    assert_matches(
        measure.trig_targ(
            measure.EventSpec.at(1.0),
            measure.EventSpec.crossing(va, 15.0, **rise1),
        ),
        expected["MEAS_TRIGTARG"],
        "trig_targ",
    )


def test_dt_weighting_is_what_makes_avg_right_on_a_non_uniform_trace():
    """The dt-weighting is not a detail, and the log says which answer is correct.

    Measured on the *adaptive* run, where the step size genuinely varies. ``MEAS_FOUR_dc`` is
    the simulator's own dt-weighted mean of that signal (``four``'s order-0 term is exactly
    ``integ / duration``), so it is directly comparable to :func:`gsim_tools.measure.avg`.

    On a sine over one whole period the true mean is ~0, and ``avg`` agrees with the simulator
    to within 1e-12 -- while a plain ``sum(values) / len(values)`` over the same samples comes
    out near 0.042, six orders of magnitude wrong, because the solver clustered its small steps
    where the waveform moves fastest. That skew is precisely what dt-weighting removes, and it
    is invisible on a fixed-step run.
    """
    trace, expected = load("sine_adaptive")
    sig = trace.samples("SIG")
    assert not trace.is_uniformly_stepped()

    assert_matches(measure.avg(sig), expected["MEAS_FOUR_dc"], "avg vs kind=measure four dc")

    naive = math.fsum(s.v for s in sig) / len(sig)
    assert abs(naive) > 1e4 * abs(expected["MEAS_FOUR_dc"]), (
        f"the naive mean ({naive}) was expected to be badly wrong here; if it is not, this "
        "trace is no longer non-uniform enough to demonstrate anything"
    )


# ---------------------------------------------------------------------------
# square.cir -- freq, on_time, off_time
# ---------------------------------------------------------------------------


def test_square_cycle_measurements_match_kind_measure():
    trace, expected = load("square")
    va = trace.samples("V(a)")
    # on=0.5 off=0.5 -- the fixture's own thresholds (no hysteresis).
    assert_matches(measure.freq(va, on=0.5, off=0.5), expected["MEAS_FREQ"], "freq")
    assert_matches(measure.on_time(va, on=0.5, off=0.5), expected["MEAS_ONTIME"], "on_time")
    assert_matches(measure.off_time(va, on=0.5, off=0.5), expected["MEAS_OFFTIME"], "off_time")


# ---------------------------------------------------------------------------
# sine.cir / two_tone.cir / sine_adaptive.cir -- four (magnitude, phase, THD)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fixture", "signal", "fundamental", "harmonics"),
    [
        # fundamental= and harmonics= come from each .cir's own kind=measure line.
        ("sine", "V(a)", 5.0, 5),
        ("two_tone", "V(a)", 10.0, 4),
        # The adaptive run: a genuinely non-uniformly stepped trace, measured on the sine
        # block's own output rather than a node voltage.
        ("sine_adaptive", "SIG", 5.0, 5),
    ],
)
def test_four_matches_kind_measure(fixture, signal, fundamental, harmonics):
    trace, expected = load(fixture)
    result = measure.four(
        trace.samples(signal), fundamental_freq=fundamental, num_harmonics=harmonics
    )

    assert_matches(result.dc, expected["MEAS_FOUR_dc"], f"{fixture} dc")
    assert_matches(
        result.thd_percent, expected["MEAS_FOUR_thd_percent"], f"{fixture} thd_percent"
    )
    for n in range(1, harmonics):
        assert_matches(
            result.harmonics[n].magnitude,
            expected[f"MEAS_FOUR_h{n}_mag"],
            f"{fixture} h{n} magnitude",
        )
        assert_matches(
            result.harmonics[n].phase_degrees,
            expected[f"MEAS_FOUR_h{n}_phase_deg"],
            f"{fixture} h{n} phase",
        )


def test_sine_adaptive_trace_really_is_non_uniformly_stepped():
    """Guards what the parity test above actually proves.

    If this run were uniformly stepped, agreeing with the Rust would say nothing about whether
    either implementation handles variable time steps -- which is the entire reason both use
    exact-segment integration. Same check, and same rationale, as
    ``general-simulator/tests/measure/test_sine_adaptive.py``.
    """
    trace, _ = load("sine_adaptive")
    steps = [b - a for a, b in zip(trace.time, trace.time[1:])]
    assert len(trace) > 20
    assert max(steps) / min(steps) > 5.0, (
        f"expected a genuinely non-uniform step size, got a max/min dt ratio of "
        f"{max(steps) / min(steps)}"
    )
    assert not trace.is_uniformly_stepped()


def test_thd_convenience_agrees_with_the_full_four_result():
    trace, expected = load("two_tone")
    va = trace.samples("V(a)")
    assert_matches(
        measure.thd(va, fundamental_freq=10.0, num_harmonics=4),
        expected["MEAS_FOUR_thd_percent"],
        "thd()",
    )


# ---------------------------------------------------------------------------
# err.cir -- err1, err2, error(l1/l2/infnorm)
# ---------------------------------------------------------------------------


def test_error_metrics_match_kind_measure():
    trace, expected = load("err")
    va, vb, vc = (trace.samples(n) for n in ("V(a)", "V(b)", "V(c)"))

    # The fixture gives no minval/ymin/ymax, so the CLI's own defaults apply -- which are this
    # module's defaults too (minval=1e-12, unbounded ymin/ymax).
    assert_matches(measure.err1(va, vb), expected["MEAS_ERR1"], "err1")
    assert_matches(measure.err2(va, vb), expected["MEAS_ERR2"], "err2")

    assert_matches(measure.error_norm(vc, vb, measure.Norm.L1), expected["MEAS_L1"], "l1")
    assert_matches(measure.error_norm(vc, vb, measure.Norm.L2), expected["MEAS_L2"], "l2")
    assert_matches(
        measure.error_norm(vc, vb, measure.Norm.INFNORM), expected["MEAS_INF"], "infnorm"
    )


# ---------------------------------------------------------------------------
# Cross-checks against the fixtures' own hand-derivations
# ---------------------------------------------------------------------------


def test_python_and_rust_both_land_on_the_hand_derived_answers():
    """Both implementations agreeing on a *wrong* number would still pass every test above.

    So this asserts the closed-form answers the fixtures' own header comments derive by hand:
    sine.cir is amplitude 2 at 5 Hz (h1_mag=2.0, phase=-90deg, ~0 THD), two_tone.cir adds a
    second harmonic at a tenth the amplitude (THD = 10% exactly), and square.cir is a 1 Hz,
    50%-duty square wave.
    """
    sine, _ = load("sine")
    result = measure.four(sine.samples("V(a)"), fundamental_freq=5.0, num_harmonics=5)
    assert result.fundamental.magnitude == pytest.approx(2.0, abs=5e-3)
    assert result.fundamental.phase_degrees == pytest.approx(-90.0, abs=0.5)
    assert result.thd_percent < 0.1

    two_tone, _ = load("two_tone")
    tt = measure.four(two_tone.samples("V(a)"), fundamental_freq=10.0, num_harmonics=4)
    assert tt.harmonics[1].magnitude == pytest.approx(1.0, abs=0.01)
    assert tt.harmonics[2].magnitude == pytest.approx(0.1, abs=0.005)
    assert tt.thd_percent == pytest.approx(10.0, abs=0.2)

    square, _ = load("square")
    sq = square.samples("V(a)")
    assert measure.freq(sq, on=0.5, off=0.5) == pytest.approx(1.0, abs=0.01)
    assert measure.duty_cycle(sq, on=0.5, off=0.5) == pytest.approx(0.5, abs=0.01)
