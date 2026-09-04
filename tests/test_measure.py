"""Closed-form checks on :mod:`gsim_tools.measure`.

These are the hand-derivable cases -- a triangle's area, a sine's RMS, a ramp's crossing time --
where the right answer is known analytically, so a passing test means the port is correct and
not merely self-consistent. They are the Python counterparts of the ``#[test]`` functions in
``gs-waveform-measurements/src/*.rs``, carried over with their derivations; the *other* half of
the proof, that this port agrees with the simulator's own Rust implementation on real traces,
lives in ``test_measure_parity.py``.
"""

from __future__ import annotations

import math

import pytest

from gsim_tools.measure import (
    Edge,
    EventSpec,
    Norm,
    Occurrence,
    Sample,
    Window,
    avg,
    average_power,
    clip,
    deriv_at,
    duty_cycle,
    err1,
    err2,
    error_norm,
    find_at,
    four,
    freq,
    integ,
    interpolate,
    max_,
    min_,
    off_time,
    on_time,
    peak,
    peak_to_peak,
    rms,
    thd,
    trig_targ,
    when,
)
from gsim_tools.measure import (
    EmptySeriesError,
    InvalidParameterError,
    NotFoundError,
)


def series(pairs):
    return [Sample(t, v) for t, v in pairs]


# --- sample.rs primitives ---------------------------------------------------


def test_interpolate_midpoint_and_outside_domain():
    s = series([(0.0, 0.0), (2.0, 10.0)])
    assert interpolate(s, 1.0) == 5.0
    assert interpolate(s, -1.0) is None
    assert interpolate(s, 3.0) is None


def test_clip_inserts_interpolated_boundaries():
    # Window [0.5, 1.5] over a 0 -> 10 -> 0 triangle: both boundaries fall mid-segment, so
    # both become new samples at v=5.0 by hand.
    s = series([(0.0, 0.0), (1.0, 10.0), (2.0, 0.0)])
    c = clip(s, Window(0.5, 1.5))
    assert c == [Sample(0.5, 5.0), Sample(1.0, 10.0), Sample(1.5, 5.0)]


def test_non_monotonic_series_is_rejected():
    with pytest.raises(InvalidParameterError):
        integ(series([(0.0, 1.0), (0.0, 2.0)]))


def test_empty_series_is_rejected():
    with pytest.raises(EmptySeriesError):
        max_([])


# --- scalar.rs --------------------------------------------------------------


def test_integ_of_ramp_is_the_triangle_area():
    # (0,0) -> (1,1): area of a right triangle = 1/2, by hand.
    assert integ(series([(0.0, 0.0), (1.0, 1.0)])) == pytest.approx(0.5, abs=1e-12)


def test_avg_is_dt_weighted_not_a_plain_mean():
    # 0,0 -> 1,10 -> 4,10 (dt=1 then dt=3). Trapezoid areas by hand: 5 and 30, total 35 over a
    # duration of 4 => 8.75. A naive sum(values)/count would give 6.667 -- this is the single
    # most important property of the whole module.
    s = series([(0.0, 0.0), (1.0, 10.0), (4.0, 10.0)])
    assert avg(s) == pytest.approx(8.75, abs=1e-12)
    assert abs(avg(s) - 20.0 / 3.0) > 0.1


def test_rms_of_constant_is_exact_regardless_of_spacing():
    s = series([(0.0, 5.0), (0.1, 5.0), (0.15, 5.0), (1.0, 5.0)])
    assert rms(s) == pytest.approx(5.0, abs=1e-12)


def test_rms_of_sine_is_amplitude_over_sqrt2_under_nonuniform_sampling():
    # Deliberately irregular dt, to prove the dt-weighting rather than relying on a convenient
    # uniform grid. Analytic answer over exactly one period: A / sqrt(2).
    amplitude, freq_hz = 3.0, 1.0
    period = 1.0 / freq_hz
    ts, t, i = [0.0], 0.0, 0
    while t < period:
        t = min(t + 0.0005 + 0.0004 * abs(math.sin(i * 0.7)), period)
        ts.append(t)
        i += 1
    s = [Sample(t, amplitude * math.sin(2 * math.pi * freq_hz * t)) for t in ts]
    assert rms(s) == pytest.approx(amplitude / math.sqrt(2), abs=1e-3)


def test_max_min_pp_and_peak():
    s = series([(0.0, 0.0), (1.0, 10.0), (2.0, -14.0), (3.0, 2.0)])
    assert (max_(s).value, max_(s).time) == (10.0, 1.0)
    assert (min_(s).value, min_(s).time) == (-14.0, 2.0)
    assert peak_to_peak(s) == pytest.approx(24.0)
    # peak() is the largest *absolute* excursion, so the negative extreme wins here.
    assert peak(s) == pytest.approx(14.0)


def test_window_restricts_extrema_to_the_interpolated_boundary():
    # Window starts at t=1.5, where linear interpolation between (1,10) and (2,-4) gives 3.0 --
    # itself the maximum over the clipped window, proving max_ sees the inserted boundary.
    s = series([(0.0, 0.0), (1.0, 10.0), (2.0, -4.0), (3.0, 2.0)])
    m = max_(s, Window(1.5, 3.0))
    assert m.value == pytest.approx(3.0, abs=1e-12)
    assert m.time == pytest.approx(1.5, abs=1e-12)


# --- crossing.rs ------------------------------------------------------------


def test_when_on_a_linear_ramp_is_exact():
    s = series([(0.0, 0.0), (10.0, 10.0)])
    assert when(s, 4.0, edge=Edge.RISING) == pytest.approx(4.0, abs=1e-12)


def test_when_respects_nonuniform_spacing():
    # Crossing of 1.2 falls inside the [0.15, 1.0] segment; hand-solved below.
    s = series([(0.0, 0.0), (0.1, 1.0), (0.15, 1.5), (1.0, 10.0)])
    expected = 0.15 + (1.2 - 1.5) / (10.0 - 1.5) * (1.0 - 0.15)
    assert when(s, 1.2, edge=Edge.RISING) == pytest.approx(expected, abs=1e-12)


def test_rise_and_fall_selection_on_a_triangle():
    s = series([(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)])
    assert when(s, 0.5, edge=Edge.RISING) == pytest.approx(0.5, abs=1e-12)
    assert when(s, 0.5, edge=Edge.FALLING) == pytest.approx(1.5, abs=1e-12)


def test_occurrence_selects_the_nth_and_the_last_crossing():
    # A two-cycle triangle: rising crossings of 0.5 at t=0.5 and t=2.5, by hand.
    s = series([(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0), (4.0, 0.0)])
    assert when(s, 0.5, edge=Edge.RISING, occurrence=Occurrence.nth(2)) == pytest.approx(2.5)
    assert when(s, 0.5, edge=Edge.RISING, occurrence=Occurrence.LAST) == pytest.approx(2.5)
    with pytest.raises(NotFoundError):
        when(s, 0.5, edge=Edge.RISING, occurrence=Occurrence.nth(3))


def test_crossing_between_two_variables_on_different_grids():
    # var1 ramps 0->10 over [0,10]; var2 is constant 5 on its own coarser grid. Their
    # difference crosses zero at t=5, independent of either grid.
    var1 = series([(0.0, 0.0), (10.0, 10.0)])
    var2 = series([(0.0, 5.0), (3.0, 5.0), (10.0, 5.0)])
    assert when(var1, var2, edge=Edge.RISING) == pytest.approx(5.0, abs=1e-9)


def test_find_at_and_deriv_at():
    assert find_at(series([(0.0, 0.0), (10.0, 20.0)]), 3.0) == pytest.approx(6.0, abs=1e-12)
    # Strictly inside a segment of slope 3: exact, not a finite difference.
    assert deriv_at(series([(0.0, 0.0), (2.0, 6.0)]), 1.0) == pytest.approx(3.0, abs=1e-12)
    # On the boundary between a slope-1 and a slope-3 segment: the central difference, 2.
    s = series([(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)])
    assert deriv_at(s, 1.0) == pytest.approx(2.0, abs=1e-12)


def test_trig_targ_fixed_and_crossing():
    s = series([(0.0, 0.0), (10.0, 10.0)])
    dt = trig_targ(
        EventSpec.at(1.0),
        EventSpec.crossing(s, 5.0, edge=Edge.RISING),
    )
    assert dt == pytest.approx(4.0, abs=1e-12)


def test_frac_max_rise_time():
    # A v=t ramp with peak 10: the 10%-90% rise time is 9 - 1 = 8, by hand.
    s = series([(0.0, 0.0), (10.0, 10.0)])
    dt = trig_targ(
        EventSpec.frac_max(s, 0.1, edge=Edge.RISING),
        EventSpec.frac_max(s, 0.9, edge=Edge.RISING),
    )
    assert dt == pytest.approx(8.0, abs=1e-9)


# --- cycle.rs ---------------------------------------------------------------


def square_wave(periods: int, high_fraction: float = 0.5):
    """An exactly-PWL square wave of period 1 with near-vertical (dt=1e-9) transitions."""
    eps, t = 1e-9, 0.0
    s = [Sample(t, 0.0)]
    for _ in range(periods):
        s.append(Sample(t + eps, 1.0))
        t += high_fraction
        s.append(Sample(t - eps, 1.0))
        s.append(Sample(t + eps, 0.0))
        t += 1.0 - high_fraction
        s.append(Sample(t - eps, 0.0))
    s.append(Sample(t, 0.0))
    return s


def test_freq_and_on_off_time_of_a_square_wave():
    s = square_wave(4)
    assert freq(s, on=0.5, off=0.5) == pytest.approx(1.0, abs=1e-4)
    assert on_time(s, on=0.5, off=0.5) == pytest.approx(0.5, abs=1e-4)
    assert off_time(s, on=0.5, off=0.5) == pytest.approx(0.5, abs=1e-4)
    assert duty_cycle(s, on=0.5, off=0.5) == pytest.approx(0.5, abs=1e-4)


def test_asymmetric_duty_cycle():
    s = square_wave(3, high_fraction=0.25)
    assert on_time(s, on=0.5, off=0.5) == pytest.approx(0.25, abs=1e-4)
    assert off_time(s, on=0.5, off=0.5) == pytest.approx(0.75, abs=1e-4)
    assert duty_cycle(s, on=0.5, off=0.5) == pytest.approx(0.25, abs=1e-4)


# --- fourier.rs -------------------------------------------------------------


def test_pure_sine_has_expected_fundamental_and_low_thd():
    amplitude, f = 2.0, 50.0
    period, n = 1.0 / f, 2000
    s = [
        Sample(period * i / n, amplitude * math.sin(2 * math.pi * f * period * i / n))
        for i in range(n + 1)
    ]
    result = four(s, fundamental_freq=f, num_harmonics=5)
    assert result.fundamental.magnitude == pytest.approx(amplitude, abs=1e-3)
    # v = A sin(wt) = A cos(wt - 90deg), so the phase must be -90 degrees.
    assert result.fundamental.phase_degrees == pytest.approx(-90.0, abs=0.5)
    assert result.thd_percent < 0.1
    assert result.dc == pytest.approx(0.0, abs=1e-6)


def test_four_on_a_triangle_wave_matches_independent_quadrature():
    """Ground truth from a *different* numerical method than the analytic segment formula.

    A midpoint Riemann sum over the triangle wave's own closed-form definition, at 200k steps,
    coded here independently -- so this checks the ported analytic integral against brute-force
    integration, not against the module's own internal consistency.
    """

    def triangle(t):
        if t <= 0.25:
            return 4.0 * t
        if t <= 0.75:
            return 1.0 - 4.0 * (t - 0.25)
        return -1.0 + 4.0 * (t - 0.75)

    def quadrature(n, steps=200_000):
        omega, dt, acc = 2 * math.pi * n, 1.0 / steps, 0j
        for i in range(steps):
            t = (i + 0.5) * dt
            v = triangle(t)
            acc += complex(v * math.cos(omega * t), -v * math.sin(omega * t)) * dt
        return acc

    s = series([(0.0, 0.0), (0.25, 1.0), (0.75, -1.0), (1.0, 0.0)])
    result = four(s, fundamental_freq=1.0, num_harmonics=4)
    for n in (1, 3):
        truth = quadrature(n)
        assert result.harmonics[n].magnitude == pytest.approx(2.0 * abs(truth), abs=1e-4)
        expected_phase = math.degrees(math.atan2(truth.imag, truth.real))
        assert result.harmonics[n].phase_degrees == pytest.approx(expected_phase, abs=0.1)
    # A symmetric triangle wave has no even harmonics.
    assert result.harmonics[2].magnitude < 1e-9


def test_thd_of_a_known_two_tone_signal():
    # sin(wt) + 0.1*sin(2wt): THD = 10% exactly, by hand.
    f, n = 10.0, 4000
    period = 1.0 / f
    s = []
    for i in range(n + 1):
        t = period * i / n
        w = 2 * math.pi * f * t
        s.append(Sample(t, math.sin(w) + 0.1 * math.sin(2 * w)))
    assert thd(s, fundamental_freq=f, num_harmonics=4) == pytest.approx(10.0, abs=0.2)


def test_four_rejects_bad_parameters():
    s = series([(0.0, 0.0), (1.0, 1.0)])
    with pytest.raises(InvalidParameterError):
        four(s, fundamental_freq=0.0, num_harmonics=4)
    with pytest.raises(InvalidParameterError):
        four(s, fundamental_freq=1.0, num_harmonics=0)
    # Fewer than DC + fundamental + one overtone: THD is undefined, reported as nan.
    assert math.isnan(four(s, fundamental_freq=1.0, num_harmonics=2).thd_percent)


# --- error_metrics.rs -------------------------------------------------------


def test_err1_err2_hand_computed():
    # comparison is always exactly half of measured, so each term is (M-C)/M = 0.5.
    m = series([(0.0, 2.0), (1.0, 4.0)])
    c = series([(0.0, 1.0), (1.0, 2.0)])
    assert err2(m, c) == pytest.approx(0.5, abs=1e-12)
    assert err1(m, c) == pytest.approx(0.5, abs=1e-12)
    # Identical series: exactly zero.
    assert abs(err1(m, m)) < 1e-12
    assert abs(err2(m, m)) < 1e-12


def test_error_norm_constant_offset_and_single_outlier():
    ref = series([(0.0, 0.0), (1.0, 10.0), (2.0, 0.0)])
    meas = series([(0.0, 3.0), (1.0, 13.0), (2.0, 3.0)])
    # A constant +3 offset makes all three norms coincide at 3.
    for norm in (Norm.L1, Norm.L2, Norm.INFNORM):
        assert error_norm(meas, ref, norm) == pytest.approx(3.0, abs=1e-9)

    # One point off by 10 among four: L1 = 10/4, L2 = sqrt(100/4), Inf = 10, by hand.
    ref = series([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)])
    meas = series([(0.0, 0.0), (1.0, 10.0), (2.0, 0.0), (3.0, 0.0)])
    assert error_norm(meas, ref, Norm.L1) == pytest.approx(2.5, abs=1e-9)
    assert error_norm(meas, ref, Norm.L2) == pytest.approx(5.0, abs=1e-9)
    assert error_norm(meas, ref, Norm.INFNORM) == pytest.approx(10.0, abs=1e-9)


# --- power (composed on top of the ported primitives) -----------------------


def test_average_power_of_a_resistive_load_is_vrms_times_irms():
    # v = 10 sin(wt) across 5 ohm => i = 2 sin(wt), in phase. Real power = Vrms*Irms
    # = (10/sqrt2)(2/sqrt2) = 10 W, and average_power must agree since cos(phi) = 1.
    f, n = 50.0, 4000
    period = 1.0 / f
    ts = [period * i / n for i in range(n + 1)]
    v = [Sample(t, 10.0 * math.sin(2 * math.pi * f * t)) for t in ts]
    i = [Sample(t, 2.0 * math.sin(2 * math.pi * f * t)) for t in ts]
    assert average_power(v, i) == pytest.approx(10.0, rel=1e-4)


def test_average_power_of_a_purely_reactive_load_is_zero():
    # 90 degrees out of phase: real power is zero even though both RMS values are nonzero.
    # This is why average_power is avg(v*i) and not rms(v)*rms(i).
    f, n = 50.0, 4000
    period = 1.0 / f
    ts = [period * i / n for i in range(n + 1)]
    v = [Sample(t, 10.0 * math.sin(2 * math.pi * f * t)) for t in ts]
    i = [Sample(t, 2.0 * math.cos(2 * math.pi * f * t)) for t in ts]
    assert average_power(v, i) == pytest.approx(0.0, abs=1e-6)
