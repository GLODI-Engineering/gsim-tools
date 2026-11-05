"""Post-simulation waveform measurements over a ``(time, value)`` time series.

This is a pure-Python port of the ``gs-waveform-measurements`` Rust crate (sibling repo
``gs-waveform-measurements/src/{sample,scalar,crossing,cycle,fourier,error_metrics}.rs``),
which is the implementation ``general-simulator``'s own ``kind=measure`` netlist blocks are
built on. The numerical methods here are the crate's, not reinvented ones; each function's
docstring names the Rust function it was ported from. Keeping the two in step matters: a
Python-side measurement is only trustworthy if it agrees with the simulator's own
``kind=measure`` answer for the same trace, and ``tests/test_measure_parity.py`` asserts
exactly that against committed ``.log`` fixtures.

Why these particular methods
----------------------------
A variable-timestep transient solver records the solution only at the instants it actually
stepped to, so its output is non-uniformly spaced in time. The one wrong thing to implement
is ``sum(values) / len(values)`` for anything called an "average" -- that silently misweights
variable step sizes. Every measurement here instead treats the input as **piecewise-linear
between real samples** and:

1. **Scalar reductions** (``rms``/``avg``/``integ``) are dt-weighted -- trapezoidal
   integration over the real samples. ``rms`` goes further and integrates each segment's
   square in closed form, so it is *exact* for piecewise-linear data.
2. **Crossing measurements** (``when``, ``find_when``, ``trig_targ``, ``freq``) interpolate
   strictly between the two adjacent real samples that straddle the threshold -- never by
   resampling the whole trace.
3. **Fourier analysis** (``four``) integrates each linear segment against the harmonic kernel
   analytically and sums the closed forms -- not a resampled FFT.

Nothing here imports numpy, ``gsim_py``, or any simulator: it takes ``Sample`` sequences in
and returns plain floats out.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Union

__all__ = [
    "Sample",
    "Window",
    "Extremum",
    "Harmonic",
    "FourierResult",
    "Edge",
    "Occurrence",
    "Norm",
    "Threshold",
    "EventSpec",
    "MeasureError",
    "EmptySeriesError",
    "EmptyWindowError",
    "NotFoundError",
    "InvalidParameterError",
    "interpolate",
    "check_monotonic",
    "clip",
    "max_",
    "min_",
    "peak_to_peak",
    "peak",
    "integ",
    "avg",
    "mean",
    "rms",
    "find_at",
    "deriv_at",
    "when",
    "find_when",
    "deriv_when",
    "resolve_event",
    "trig_targ",
    "freq",
    "on_time",
    "off_time",
    "duty_cycle",
    "four",
    "thd",
    "err1",
    "err2",
    "error_norm",
    "product",
    "average_power",
    "rms_power",
]


# ---------------------------------------------------------------------------
# Errors (ported from sample.rs's MeasureError enum)
# ---------------------------------------------------------------------------


class MeasureError(Exception):
    """Base class for every measurement failure.

    Ported from ``sample.rs``'s ``MeasureError``. Like the Rust crate, nothing here silently
    substitutes a fallback value the way ngspice/Xyce's ``DEFAULT_VAL=`` does -- a caller who
    wants that behavior wraps the call in ``try``/``except``.
    """


class EmptySeriesError(MeasureError):
    """The input series had fewer than two samples, so no interval quantity is defined."""


class EmptyWindowError(MeasureError):
    """The window (after intersecting from/to/td with the series' domain) has < 2 samples."""


class NotFoundError(MeasureError):
    """A crossing, threshold, or fixed time was requested but never occurs in the window."""


class InvalidParameterError(MeasureError):
    """A parameter combination is invalid independent of the data."""


# ---------------------------------------------------------------------------
# Core types (ported from sample.rs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """A single ``(time, value)`` observation. Ported from ``sample.rs``'s ``Sample``.

    ``t`` is simulation time in this library's intended use, but nothing here assumes that --
    any monotonically increasing coordinate works.
    """

    t: float
    v: float


@dataclass(frozen=True)
class Window:
    """A measurement window, mirroring ngspice/Xyce's ``FROM=``/``TO=``/``TD=`` qualifiers.

    Ported from ``sample.rs``'s ``Window``. The effective window is the intersection
    ``[max(from, td), to]`` with the series' own domain; any bound left as ``None`` is
    unconstrained on that side. ``Window()`` is the full, unconstrained window.
    """

    from_: Optional[float] = None
    to: Optional[float] = None
    td: Optional[float] = None

    def effective_from(self) -> float:
        if self.from_ is not None and self.td is not None:
            return max(self.from_, self.td)
        if self.from_ is not None:
            return self.from_
        if self.td is not None:
            return self.td
        return -math.inf

    def effective_to(self) -> float:
        return math.inf if self.to is None else self.to


FULL = Window()
"""The unconstrained window (``Window::FULL`` in the Rust crate)."""


def _win(window: Optional[Window]) -> Window:
    return FULL if window is None else window


def interpolate(samples: Sequence[Sample], t: float) -> Optional[float]:
    """Linearly interpolate ``samples`` (sorted by non-decreasing ``t``) at time ``t``.

    Returns ``None`` if ``t`` falls outside the series' domain. Ported from ``sample.rs``'s
    ``interpolate`` -- the single primitive every "value at an arbitrary time" measurement is
    built from.
    """
    if not samples:
        return None
    first, last = samples[0], samples[-1]
    if t < first.t or t > last.t:
        return None
    if t == first.t:
        return first.v
    if t == last.t:
        return last.v
    # `key=` rather than materialising the times: `bisect_left([s.t for s in
    # samples], t)` rebuilt the whole list of times on *every* call, making
    # this O(n) instead of O(log n) and defeating the bisect entirely. Since
    # every value-at-an-arbitrary-time measurement is built on this primitive,
    # anything that interpolated once per sample became quadratic —
    # `average_power` over a 20 ms run at 0.1 us took about fifteen minutes.
    idx = bisect.bisect_left(samples, t, key=lambda s: s.t)
    if idx < len(samples) and samples[idx].t == t:
        return samples[idx].v
    a, b = samples[idx - 1], samples[idx]
    frac = (t - a.t) / (b.t - a.t)
    return a.v + frac * (b.v - a.v)


def check_monotonic(samples: Sequence[Sample]) -> None:
    """Raise ``InvalidParameterError`` unless ``samples`` is strictly increasing in ``t``.

    Ported from ``sample.rs``'s ``check_monotonic``.
    """
    for a, b in zip(samples, samples[1:]):
        if b.t <= a.t:
            raise InvalidParameterError(
                "samples must be sorted by strictly increasing t "
                f"(t={b.t} follows t={a.t})"
            )


def clip(samples: Sequence[Sample], window: Optional[Window] = None) -> list[Sample]:
    """Clip ``samples`` to the effective window, inserting interpolated boundary samples.

    A boundary falling strictly between two real samples becomes a new interpolated
    ``Sample``, so integration over the result is exact up to the window edges rather than
    snapping to the nearest real sample. Ported from ``sample.rs``'s ``clip``.
    """
    window = _win(window)
    if not samples:
        raise EmptySeriesError("series has fewer than two samples")
    check_monotonic(samples)

    lo = max(window.effective_from(), samples[0].t)
    hi = min(window.effective_to(), samples[-1].t)
    if lo >= hi:
        raise EmptyWindowError("measurement window contains fewer than two samples")

    out: list[Sample] = []
    v_lo = interpolate(samples, lo)
    if v_lo is not None:
        out.append(Sample(lo, v_lo))
    out.extend(s for s in samples if lo < s.t < hi)
    v_hi = interpolate(samples, hi)
    if v_hi is not None:
        out.append(Sample(hi, v_hi))

    if len(out) < 2:
        raise EmptyWindowError("measurement window contains fewer than two samples")
    return out


# ---------------------------------------------------------------------------
# Scalar reductions (ported from scalar.rs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Extremum:
    """The value and time of an extremum, covering both ``MAX``/``MIN`` and ``MAX_AT``/``MIN_AT``.

    Ported from ``scalar.rs``'s ``Extremum``. On a flat plateau, ``time`` is the *first*
    sample (scanning forward) at which the extreme value occurs.
    """

    value: float
    time: float


def max_(samples: Sequence[Sample], window: Optional[Window] = None) -> Extremum:
    """``MAX``/``MAX_AT``: the maximum value within ``window``, and when it occurs.

    Ported from ``scalar.rs``'s ``max``. Named with a trailing underscore so it does not
    shadow the builtin.
    """
    best: Optional[Extremum] = None
    for s in clip(samples, window):
        if best is None or s.v > best.value:
            best = Extremum(s.v, s.t)
    assert best is not None  # clip() guarantees at least two samples
    return best


def min_(samples: Sequence[Sample], window: Optional[Window] = None) -> Extremum:
    """``MIN``/``MIN_AT``: the minimum value within ``window``, and when it occurs.

    Ported from ``scalar.rs``'s ``min``.
    """
    best: Optional[Extremum] = None
    for s in clip(samples, window):
        if best is None or s.v < best.value:
            best = Extremum(s.v, s.t)
    assert best is not None
    return best


def peak_to_peak(samples: Sequence[Sample], window: Optional[Window] = None) -> float:
    """``PP``: peak-to-peak amplitude, ``max(v) - min(v)``. Ported from ``scalar.rs``."""
    return max_(samples, window).value - min_(samples, window).value


def peak(samples: Sequence[Sample], window: Optional[Window] = None) -> float:
    """The largest absolute excursion, ``max(|v|)`` -- the usual meaning of "peak" for a
    bipolar waveform, where ``MAX`` alone would report only the positive extreme.

    Built from the ported ``max_``/``min_`` rather than being a separate scan.
    """
    hi = max_(samples, window).value
    lo = min_(samples, window).value
    return max(abs(hi), abs(lo))


def _trapezoid(samples: Sequence[Sample]) -> float:
    return math.fsum(
        0.5 * (a.v + b.v) * (b.t - a.t) for a, b in zip(samples, samples[1:])
    )


def integ(samples: Sequence[Sample], window: Optional[Window] = None) -> float:
    """``INTEG``: the definite integral of v dt over ``window``, by the trapezoidal rule.

    Exact for piecewise-linear data, and correct under variable time steps because each
    trapezoid uses that segment's own dt. Ported from ``scalar.rs``'s ``integ``.
    """
    return _trapezoid(clip(samples, window))


def avg(samples: Sequence[Sample], window: Optional[Window] = None) -> float:
    """``AVG``: the dt-weighted mean, i.e. ``integ / duration``.

    Never a plain arithmetic mean of sample values, which would misweight irregular spacing.
    Ported from ``scalar.rs``'s ``avg``.
    """
    clipped = clip(samples, window)
    return _trapezoid(clipped) / (clipped[-1].t - clipped[0].t)


#: ``mean`` is an alias for :func:`avg` -- the dt-weighted mean, never ``sum(v)/len(v)``.
mean = avg


def rms(samples: Sequence[Sample], window: Optional[Window] = None) -> float:
    """``RMS``: the dt-weighted root-mean-square over ``window``.

    Because v is linear within each segment, v^2 is quadratic there and its exact integral is
    ``dt/3 * (v0^2 + v0*v1 + v1^2)``. Summing that closed form over every segment makes this
    RMS *exact* for piecewise-linear data -- not an approximation from trapezoidally
    integrating the squared sample values. Ported from ``scalar.rs``'s ``rms``.
    """
    clipped = clip(samples, window)
    duration = clipped[-1].t - clipped[0].t
    integral_v2 = math.fsum(
        (b.t - a.t) / 3.0 * (a.v * a.v + a.v * b.v + b.v * b.v)
        for a, b in zip(clipped, clipped[1:])
    )
    return math.sqrt(integral_v2 / duration)


# ---------------------------------------------------------------------------
# Threshold crossings (ported from crossing.rs)
# ---------------------------------------------------------------------------


class Edge(Enum):
    """Which crossing direction to accept -- ngspice/Xyce's ``RISE=``/``FALL=``/``CROSS=``."""

    RISING = "rising"
    FALLING = "falling"
    EITHER = "either"


class Occurrence:
    """Which occurrence of a matching crossing to use.

    ``Occurrence.nth(n)`` is the 1-based n-th crossing (``RISE=1`` is the *first* rising
    crossing); ``Occurrence.LAST`` is the last one found in the window. Ported from
    ``crossing.rs``'s ``Occurrence`` enum.
    """

    __slots__ = ("n",)

    def __init__(self, n: Optional[int]) -> None:
        self.n = n

    @classmethod
    def nth(cls, n: int) -> "Occurrence":
        return cls(n)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "Occurrence.LAST" if self.n is None else f"Occurrence.nth({self.n})"


Occurrence.LAST = Occurrence(None)  # type: ignore[attr-defined]
Occurrence.FIRST = Occurrence(1)  # type: ignore[attr-defined]


#: The right-hand side of a ``WHEN <var> = <value>|<var2>`` clause: either a fixed level, or
#: another series whose difference with the first is watched for a sign change. Ported from
#: ``crossing.rs``'s ``Threshold`` enum.
Threshold = Union[float, Sequence[Sample]]


def find_at(samples: Sequence[Sample], t: float) -> float:
    """``FIND <var> AT=<time>``: the value at an arbitrary time, by linear interpolation.

    Ported from ``crossing.rs``'s ``find_at``.
    """
    check_monotonic(samples)
    v = interpolate(samples, t)
    if v is None:
        raise NotFoundError(f"t={t} is outside the series' domain")
    return v


def deriv_at(samples: Sequence[Sample], t: float) -> float:
    """``DERIV <var> AT=<time>``: the derivative at ``t``.

    Because the data is piecewise-linear, the derivative is *constant* within a segment -- it
    is exactly that segment's slope, not a finite-difference approximation. On an interior
    sample (a segment boundary), the natural generalization is the central difference: the
    average of the slopes on either side. Ported from ``crossing.rs``'s ``deriv_at``.
    """
    check_monotonic(samples)
    if len(samples) < 2:
        raise EmptySeriesError("series has fewer than two samples")
    if t < samples[0].t or t > samples[-1].t:
        raise NotFoundError(f"t={t} is outside the series' domain")

    def slope(a: Sample, b: Sample) -> float:
        return (b.v - a.v) / (b.t - a.t)

    times = [s.t for s in samples]
    idx = bisect.bisect_left(times, t)
    if idx < len(samples) and samples[idx].t == t:
        left = slope(samples[idx - 1], samples[idx]) if idx > 0 else None
        right = slope(samples[idx], samples[idx + 1]) if idx + 1 < len(samples) else None
        if left is not None and right is not None:
            return 0.5 * (left + right)
        if left is not None:
            return left
        if right is not None:
            return right
        raise EmptySeriesError("series has fewer than two samples")
    return slope(samples[idx - 1], samples[idx])


@dataclass(frozen=True)
class _CrossingEvent:
    t: float
    edge: Edge


def _diff_series(var: Sequence[Sample], threshold: Threshold) -> list[Sample]:
    """The "difference from threshold" series, on the union of both inputs' sample times.

    Ported from ``crossing.rs``'s ``diff_series``: a crossing between two series with
    different sample grids is found by interpolating each onto the other's *real* sample
    times, never by resampling either onto a synthetic uniform grid.
    """
    check_monotonic(var)
    if not var:
        raise EmptySeriesError("series has fewer than two samples")

    if isinstance(threshold, (int, float)):
        level = float(threshold)
        return [Sample(s.t, s.v - level) for s in var]

    other = threshold
    check_monotonic(other)
    if not other:
        raise EmptySeriesError("series has fewer than two samples")
    lo = max(var[0].t, other[0].t)
    hi = min(var[-1].t, other[-1].t)
    if lo >= hi:
        raise EmptyWindowError("the two series' domains do not overlap")

    times = sorted(s.t for s in list(var) + list(other) if lo <= s.t <= hi)
    deduped: list[float] = []
    for t in times:
        if not deduped or abs(t - deduped[-1]) >= 1e-15:
            deduped.append(t)

    diffs: list[Sample] = []
    for t in deduped:
        a = interpolate(var, t)
        b = interpolate(other, t)
        if a is not None and b is not None:
            diffs.append(Sample(t, a - b))
    if len(diffs) < 2:
        raise EmptyWindowError("fewer than two overlapping samples")
    return diffs


def _find_crossings(diff: Sequence[Sample]) -> list[_CrossingEvent]:
    """Every zero-crossing of a difference series.

    Uses the same level-crossing pseudocode ngspice/Xyce document, ported from
    ``crossing.rs``'s ``find_crossings``: rising if ``current >= 0 and previous < 0``,
    falling if ``current <= 0 and previous > 0``.
    """
    out: list[_CrossingEvent] = []
    for a, b in zip(diff, diff[1:]):
        d0, d1 = a.v, b.v
        if d0 == 0.0 and d1 == 0.0:
            continue
        t = a.t if d1 == d0 else a.t + (0.0 - d0) * (b.t - a.t) / (d1 - d0)
        if d1 >= 0.0 and d0 < 0.0:
            out.append(_CrossingEvent(t, Edge.RISING))
        elif d1 <= 0.0 and d0 > 0.0:
            out.append(_CrossingEvent(t, Edge.FALLING))
    return out


def _select_crossing(
    events: Sequence[_CrossingEvent], edge: Edge, occurrence: Occurrence
) -> float:
    matching = [e for e in events if edge is Edge.EITHER or e.edge is edge]
    if occurrence.n is None:
        if not matching:
            raise NotFoundError("no matching crossing found in the window")
        return matching[-1].t
    if occurrence.n == 0:
        raise InvalidParameterError(
            "RISE/FALL/CROSS occurrence is 1-based; 0 is not valid"
        )
    if occurrence.n > len(matching):
        raise NotFoundError(
            f"only {len(matching)} matching crossing(s) found, wanted #{occurrence.n}"
        )
    return matching[occurrence.n - 1].t


def when(
    var: Sequence[Sample],
    threshold: Threshold,
    window: Optional[Window] = None,
    edge: Edge = Edge.EITHER,
    occurrence: Optional[Occurrence] = None,
) -> float:
    """``WHEN <var> = <value>|<var2>``: the *time* at which ``var`` crosses ``threshold``.

    Ported from ``crossing.rs``'s ``when``.
    """
    occurrence = occurrence or Occurrence.nth(1)
    diff = clip(_diff_series(var, threshold), window)
    return _select_crossing(_find_crossings(diff), edge, occurrence)


def find_when(
    find_var: Sequence[Sample],
    when_var: Sequence[Sample],
    threshold: Threshold,
    window: Optional[Window] = None,
    edge: Edge = Edge.EITHER,
    occurrence: Optional[Occurrence] = None,
) -> float:
    """``FIND <a> WHEN <b> = <value>``: the value of ``find_var`` when ``when_var`` crosses.

    Ported from ``crossing.rs``'s ``find_when``.
    """
    t = when(when_var, threshold, window, edge, occurrence)
    v = interpolate(find_var, t)
    if v is None:
        raise NotFoundError(f"crossing at t={t} is outside find_var's domain")
    return v


def deriv_when(
    variable: Sequence[Sample],
    when_var: Sequence[Sample],
    threshold: Threshold,
    window: Optional[Window] = None,
    edge: Edge = Edge.EITHER,
    occurrence: Optional[Occurrence] = None,
) -> float:
    """``DERIV <a> WHEN <b> = <value>``. Ported from ``crossing.rs``'s ``deriv_when``."""
    t = when(when_var, threshold, window, edge, occurrence)
    return deriv_at(variable, t)


@dataclass(frozen=True)
class EventSpec:
    """One side of a ``TRIG``/``TARG`` pair. Ported from ``crossing.rs``'s ``EventSpec``.

    Exactly one of the three constructors is used:

    * :meth:`at` -- ``TRIG AT=<time>``: a fixed, already-known time (no search).
    * :meth:`crossing` -- ``TRIG <var>=<value>``: a threshold crossing with its own
      ``RISE=``/``FALL=``/``CROSS=`` selection and window.
    * :meth:`frac_max` -- ``TRIG <var> FRAC_MAX=<frac>``: a crossing of ``frac * max(var)``
      rather than an absolute level, for rise-time style measurements where the peak is not
      known in advance.
    """

    time: Optional[float] = None
    var: Optional[Sequence[Sample]] = None
    threshold: Optional[Threshold] = None
    frac: Optional[float] = None
    window: Optional[Window] = None
    edge: Edge = Edge.EITHER
    occurrence: Optional[Occurrence] = None

    @classmethod
    def at(cls, time: float) -> "EventSpec":
        return cls(time=time)

    @classmethod
    def crossing(
        cls,
        var: Sequence[Sample],
        threshold: Threshold,
        window: Optional[Window] = None,
        edge: Edge = Edge.EITHER,
        occurrence: Optional[Occurrence] = None,
    ) -> "EventSpec":
        return cls(
            var=var,
            threshold=threshold,
            window=window,
            edge=edge,
            occurrence=occurrence,
        )

    @classmethod
    def frac_max(
        cls,
        var: Sequence[Sample],
        frac: float,
        window: Optional[Window] = None,
        edge: Edge = Edge.EITHER,
        occurrence: Optional[Occurrence] = None,
    ) -> "EventSpec":
        return cls(
            var=var, frac=frac, window=window, edge=edge, occurrence=occurrence
        )


def resolve_event(spec: EventSpec) -> float:
    """Resolve an :class:`EventSpec` to a concrete time. Ported from ``resolve_event``."""
    if spec.time is not None:
        return spec.time
    if spec.var is None:
        raise InvalidParameterError("EventSpec has neither a time nor a variable")
    occurrence = spec.occurrence or Occurrence.nth(1)
    if spec.frac is not None:
        level: Threshold = spec.frac * max_(spec.var, spec.window).value
    else:
        if spec.threshold is None:
            raise InvalidParameterError("EventSpec crossing has no threshold")
        level = spec.threshold
    diff = clip(_diff_series(spec.var, level), spec.window)
    return _select_crossing(_find_crossings(diff), spec.edge, occurrence)


def trig_targ(trig: EventSpec, targ: EventSpec) -> float:
    """``TRIG ... TARG ...``: ``t_targ - t_trig``. Ported from ``crossing.rs``'s ``trig_targ``."""
    return resolve_event(targ) - resolve_event(trig)


# ---------------------------------------------------------------------------
# Cycle counting (ported from cycle.rs)
# ---------------------------------------------------------------------------


@dataclass
class _CycleWalk:
    on_crossings: list[float]
    on_intervals: list[tuple[float, float]]
    off_intervals: list[tuple[float, float]]


def _walk(samples: Sequence[Sample], on: float, off: float) -> _CycleWalk:
    """Two-level (Schmitt-trigger) state walk. Ported from ``cycle.rs``'s ``walk``."""
    on_crossings: list[float] = []
    on_intervals: list[tuple[float, float]] = []
    off_intervals: list[tuple[float, float]] = []

    if samples[0].v >= on:
        state = "on"
    elif samples[0].v <= off:
        state = "off"
    else:
        state = "unknown"
    state_start = samples[0].t
    if state == "on":
        on_crossings.append(samples[0].t)

    for a, b in zip(samples, samples[1:]):
        if state != "on" and b.v >= on and a.v < on:
            t = a.t + (on - a.v) * (b.t - a.t) / (b.v - a.v)
            if state == "off":
                off_intervals.append((state_start, t))
            on_crossings.append(t)
            state, state_start = "on", t
        elif state != "off" and b.v <= off and a.v > off:
            t = a.t + (off - a.v) * (b.t - a.t) / (b.v - a.v)
            if state == "on":
                on_intervals.append((state_start, t))
            state, state_start = "off", t

    last_t = samples[-1].t
    if state == "on":
        on_intervals.append((state_start, last_t))
    elif state == "off":
        off_intervals.append((state_start, last_t))

    return _CycleWalk(on_crossings, on_intervals, off_intervals)


def _cycle_count_and_bounds(walk: _CycleWalk) -> tuple[int, float, float]:
    if len(walk.on_crossings) < 2:
        raise NotFoundError(
            "fewer than two on-crossings found -- no complete cycle in the window"
        )
    return len(walk.on_crossings) - 1, walk.on_crossings[0], walk.on_crossings[-1]


def _sum_intervals_clipped(
    intervals: Sequence[tuple[float, float]], lo: float, hi: float
) -> float:
    return math.fsum(max(min(b, hi) - max(a, lo), 0.0) for a, b in intervals)


def freq(
    samples: Sequence[Sample],
    window: Optional[Window] = None,
    on: float = 0.0,
    off: float = 0.0,
) -> float:
    """``FREQ``: frequency by counting complete on/off cycles between the two thresholds.

    ``on`` and ``off`` are the Schmitt-trigger levels (``on >= off``; the gap between them is
    the hysteresis band, which for a clean square wave can be zero, both set to the midpoint).
    Result is ``(number of complete cycles) / (time spanned by those cycles)``. Ported from
    ``cycle.rs``'s ``freq``.
    """
    walk = _walk(clip(samples, window), on, off)
    n, lo, hi = _cycle_count_and_bounds(walk)
    span = hi - lo
    if span <= 0.0:
        raise NotFoundError("the counted cycles span zero time")
    return n / span


def on_time(
    samples: Sequence[Sample],
    window: Optional[Window] = None,
    on: float = 0.0,
    off: float = 0.0,
) -> float:
    """``ON_TIME``: total time at or above ``on``, normalized by the number of complete cycles.

    Restricted to the span covered by complete cycles, so a partial pulse straddling the very
    start or end of the data doesn't distort the per-cycle average. Ported from ``cycle.rs``.
    """
    walk = _walk(clip(samples, window), on, off)
    n, lo, hi = _cycle_count_and_bounds(walk)
    return _sum_intervals_clipped(walk.on_intervals, lo, hi) / n


def off_time(
    samples: Sequence[Sample],
    window: Optional[Window] = None,
    on: float = 0.0,
    off: float = 0.0,
) -> float:
    """``OFF_TIME``: total time at or below ``off``, normalized by complete-cycle count.

    Ported from ``cycle.rs``'s ``off_time``.
    """
    walk = _walk(clip(samples, window), on, off)
    n, lo, hi = _cycle_count_and_bounds(walk)
    return _sum_intervals_clipped(walk.off_intervals, lo, hi) / n


def duty_cycle(
    samples: Sequence[Sample],
    window: Optional[Window] = None,
    on: float = 0.0,
    off: float = 0.0,
) -> float:
    """Fraction of each cycle spent on, ``on_time / (on_time + off_time)``.

    Not a measure the Rust crate exposes directly, but the obvious composition of the two it
    does -- and the number people actually want from a PWM waveform.
    """
    t_on = on_time(samples, window, on, off)
    t_off = off_time(samples, window, on, off)
    total = t_on + t_off
    if total <= 0.0:
        raise NotFoundError("cycles span zero on- and off-time")
    return t_on / total


# ---------------------------------------------------------------------------
# Fourier analysis (ported from fourier.rs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Harmonic:
    """One harmonic of a :func:`four` result. Ported from ``fourier.rs``'s ``Harmonic``.

    For ``order == 0`` the magnitude is the signed DC (average) value; for ``order >= 1`` it
    is the peak amplitude ``2 * |C_n|``, always non-negative. ``phase_degrees`` follows
    ngspice/Xyce's convention of degrees, such that
    ``v(t) = mag_0 + sum_n mag_n * cos(n*w0*t + phase_n)``.
    """

    order: int
    frequency: float
    magnitude: float
    phase_degrees: float


@dataclass(frozen=True)
class FourierResult:
    """The full result of a :func:`four` call. Ported from ``fourier.rs``.

    ``harmonics[0]`` is the DC term. ``thd_percent`` is
    ``100 * sqrt(sum_{n>=2} A_n^2) / A_1``; it is ``nan`` unless at least a fundamental and
    one overtone were requested (``num_harmonics >= 3``).
    """

    harmonics: list[Harmonic]
    thd_percent: float

    @property
    def dc(self) -> float:
        """The DC (average) term, ``harmonics[0].magnitude``."""
        return self.harmonics[0].magnitude

    @property
    def fundamental(self) -> Harmonic:
        """The n=1 harmonic."""
        if len(self.harmonics) < 2:
            raise InvalidParameterError("no fundamental was requested")
        return self.harmonics[1]


def _segment_harmonic_integral(
    t0: float, v0: float, v1: float, dt: float, omega: float
) -> complex:
    """Exact analytic integral of one linear segment against ``e^{-j*omega*t}``.

    Ported from ``fourier.rs``'s ``segment_harmonic_integral``, specialized (as there) to a
    purely imaginary ``a = j*omega``, which is the only case ``FOUR`` ever needs.
    """
    m = (v1 - v0) / dt
    s, c = math.sin(omega * dt), math.cos(omega * dt)

    # I0 = (1 - e^{-j*omega*dt}) / (j*omega)
    i0 = complex(s / omega, -(1.0 - c) / omega)
    # I1 = (1 - e^{-j*omega*dt}*(1 + j*omega*dt)) / (j*omega)^2
    omega2 = omega * omega
    i1 = complex(
        (c + omega * dt * s - 1.0) / omega2,
        (omega * dt * c - s) / omega2,
    )

    bracket = i0 * v0 + i1 * m
    s0, c0 = math.sin(omega * t0), math.cos(omega * t0)
    return complex(c0, -s0) * bracket


def four(
    samples: Sequence[Sample],
    fundamental_freq: float,
    num_harmonics: int = 10,
    window: Optional[Window] = None,
) -> FourierResult:
    """``FOUR``: bounded-harmonic Fourier analysis at ``fundamental_freq`` (Hz).

    ``num_harmonics`` counts the DC term, matching ngspice/Xyce's ``NUMFREQ`` convention of
    "DC plus NUMFREQ-1 harmonics", so ``num_harmonics=N`` yields orders ``0..N-1``.

    The window's duration is used as the integration period, so the caller is responsible for
    choosing a window spanning a whole number of fundamental periods -- exactly as in
    ngspice/Xyce; this has no notion of "the" period of an arbitrary signal.

    Ported from ``fourier.rs``'s ``four``. Each coefficient is the sum of the closed-form
    integral of every linear segment against the harmonic kernel -- zero interpolation error
    on genuinely non-uniform data, at ``O(N * num_harmonics)`` rather than an FFT's
    ``O(N log N)`` plus a resampling step's own error.
    """
    if num_harmonics < 1:
        raise InvalidParameterError("num_harmonics must be at least 1 (the DC term)")
    if fundamental_freq <= 0.0:
        raise InvalidParameterError("fundamental_freq must be positive")

    clipped = clip(samples, window)
    duration = clipped[-1].t - clipped[0].t
    pairs = list(zip(clipped, clipped[1:]))

    harmonics: list[Harmonic] = []
    for n in range(num_harmonics):
        if n == 0:
            dc = _trapezoid(clipped) / duration
            harmonics.append(Harmonic(0, 0.0, dc, 0.0))
            continue
        omega = 2.0 * math.pi * n * fundamental_freq
        total = 0j
        for a, b in pairs:
            total += _segment_harmonic_integral(a.t, a.v, b.v, b.t - a.t, omega)
        c_n = total / duration
        harmonics.append(
            Harmonic(
                order=n,
                frequency=n * fundamental_freq,
                magnitude=2.0 * abs(c_n),
                phase_degrees=math.degrees(math.atan2(c_n.imag, c_n.real)),
            )
        )

    if num_harmonics >= 3:
        fundamental = harmonics[1].magnitude
        overtones = math.fsum(h.magnitude**2 for h in harmonics[2:])
        thd_percent = 100.0 * math.sqrt(overtones) / fundamental
    else:
        thd_percent = math.nan

    return FourierResult(harmonics, thd_percent)


def thd(
    samples: Sequence[Sample],
    fundamental_freq: float,
    num_harmonics: int = 10,
    window: Optional[Window] = None,
) -> float:
    """Total harmonic distortion as a percentage -- :func:`four`'s ``thd_percent``.

    A convenience over the full :func:`four` result, since THD is by far its most-asked-for
    field. ``num_harmonics`` must be at least 3 (DC + fundamental + one overtone).
    """
    return four(samples, fundamental_freq, num_harmonics, window).thd_percent


# ---------------------------------------------------------------------------
# Waveform comparison (ported from error_metrics.rs)
# ---------------------------------------------------------------------------


def _err_terms(
    measured: Sequence[Sample],
    comparison: Sequence[Sample],
    minval: float,
    ymin: float,
    ymax: float,
) -> list[float]:
    if not measured or not comparison:
        raise EmptySeriesError("series has fewer than two samples")
    terms: list[float] = []
    for m in measured:
        if not (ymin <= abs(m.v) <= ymax):
            continue
        c = interpolate(comparison, m.t)
        if c is None:
            continue
        terms.append((m.v - c) / max(minval, abs(m.v)))
    if not terms:
        raise EmptyWindowError("no sample pairs survived the ymin/ymax filter")
    return terms


def err1(
    measured: Sequence[Sample],
    comparison: Sequence[Sample],
    minval: float = 1e-12,
    ymin: float = 0.0,
    ymax: float = math.inf,
) -> float:
    """``ERR1``: RMS of the per-sample relative difference between two series.

    ``comparison`` is linearly interpolated onto ``measured``'s own sample times, so the two
    need not share a time grid; a sample is skipped if ``|M_i|`` falls outside
    ``[ymin, ymax]``. Ported from ``error_metrics.rs``'s ``err1``; the defaults here are the
    ones ``general-simulator``'s own ``kind=measure type=err1`` applies when the netlist omits
    them (``minval=1e-12``, no ymin/ymax bound), so a Python-side call and the simulator's own
    answer agree by default.
    """
    terms = _err_terms(measured, comparison, minval, ymin, ymax)
    return math.sqrt(math.fsum(t * t for t in terms) / len(terms))


def err2(
    measured: Sequence[Sample],
    comparison: Sequence[Sample],
    minval: float = 1e-12,
    ymin: float = 0.0,
    ymax: float = math.inf,
) -> float:
    """``ERR2``: mean of the same per-sample relative differences :func:`err1` RMSes.

    Ported from ``error_metrics.rs``'s ``err2``.
    """
    terms = _err_terms(measured, comparison, minval, ymin, ymax)
    return math.fsum(terms) / len(terms)


class Norm(Enum):
    """Which norm :func:`error_norm` computes -- Xyce's ``ERROR`` measure norms."""

    L1 = "l1"
    L2 = "l2"
    INFNORM = "infnorm"


def error_norm(
    measured: Sequence[Sample],
    reference: Sequence[Sample],
    norm: Norm = Norm.L2,
) -> float:
    """``ERROR``: the norm between ``measured`` and a ``reference`` waveform.

    Evaluated at ``measured``'s own sample times restricted to the overlap of both domains,
    with ``reference`` linearly interpolated onto that grid. Xyce is careful to note this is
    *not* the functional equivalent of ``ERR1``/``ERR2``. Ported from
    ``error_metrics.rs``'s ``error_norm``.
    """
    if not measured or not reference:
        raise EmptySeriesError("series has fewer than two samples")
    lo, hi = reference[0].t, reference[-1].t
    diffs: list[float] = []
    for m in measured:
        if not (lo <= m.t <= hi):
            continue
        r = interpolate(reference, m.t)
        if r is not None:
            diffs.append(m.v - r)
    if not diffs:
        raise EmptyWindowError("the two series' domains do not overlap")
    n = len(diffs)
    if norm is Norm.L1:
        return math.fsum(abs(d) for d in diffs) / n
    if norm is Norm.L2:
        return math.sqrt(math.fsum(d * d for d in diffs) / n)
    return max(abs(d) for d in diffs)


# ---------------------------------------------------------------------------
# Power (composed from the ported primitives)
# ---------------------------------------------------------------------------


def product(a: Sequence[Sample], b: Sequence[Sample]) -> list[Sample]:
    """The pointwise product of two series, on the union of their real sample times.

    Built the same way ``crossing.rs``'s ``diff_series`` builds a difference: each series is
    interpolated onto the other's real sample times over their overlapping domain, never onto
    a synthetic uniform grid. Note that a product of two piecewise-linear signals is only
    piecewise-*quadratic*, so the result is a PWL approximation of it -- exact only where the
    two share sample times densely, which is the normal case for two columns of one
    simulator run.
    """
    check_monotonic(a)
    check_monotonic(b)
    if not a or not b:
        raise EmptySeriesError("series has fewer than two samples")
    lo = max(a[0].t, b[0].t)
    hi = min(a[-1].t, b[-1].t)
    if lo >= hi:
        raise EmptyWindowError("the two series' domains do not overlap")

    times = sorted(s.t for s in list(a) + list(b) if lo <= s.t <= hi)
    deduped: list[float] = []
    for t in times:
        if not deduped or abs(t - deduped[-1]) >= 1e-15:
            deduped.append(t)

    out: list[Sample] = []
    for t in deduped:
        va, vb = interpolate(a, t), interpolate(b, t)
        if va is not None and vb is not None:
            out.append(Sample(t, va * vb))
    if len(out) < 2:
        raise EmptyWindowError("fewer than two overlapping samples")
    return out


def average_power(
    voltage: Sequence[Sample],
    current: Sequence[Sample],
    window: Optional[Window] = None,
) -> float:
    """Real (average) power, the dt-weighted mean of the instantaneous product ``v * i``.

    This is the honest definition for an arbitrary waveform: ``avg(v*i)``, *not*
    ``rms(v) * rms(i)`` (which is apparent power, and equals real power only at unity power
    factor). Composed from :func:`product` and the ported :func:`avg`.

    Note the sign convention is whatever the two input series carry -- a SPICE branch current
    ``I(Vsrc)`` flows into the source's positive terminal, so ``average_power(v, i)`` for a
    source comes out negative when the source is delivering power.
    """
    return avg(product(voltage, current), window)


def rms_power(
    voltage: Sequence[Sample],
    current: Sequence[Sample],
    window: Optional[Window] = None,
) -> float:
    """Apparent power, ``rms(v) * rms(i)``.

    Equals :func:`average_power` only at unity power factor; the ratio of the two is the
    power factor itself.
    """
    return rms(voltage, window) * rms(current, window)
