"""The zero-dependency half of this library: read a CSV, measure it, no gsim_py, no Rust.

This is the use case ``gsim_py`` was made optional for -- someone with a simulation result and
a notebook, who wants RMS and THD out of it and has no interest in building a Rust extension.
Nothing imported here needs a compiler, and it deliberately never touches
:mod:`gsim_tools.schematic`.

The input is ``tests/fixtures/two_tone.csv``, a real ``general-simulator`` run of a signal that
is a 10 Hz fundamental plus a second harmonic at a tenth the amplitude -- so the answers are
known by hand: fundamental magnitude 1.0, second-harmonic magnitude 0.1, THD exactly 10%.
``tests/fixtures/two_tone.log`` is what the simulator's own ``kind=measure`` computed for the
same run, printed alongside for comparison.

Run it::

    python examples/measure_a_csv_without_gsim_py.py
"""

import pathlib

from gsim_tools import logs, measure, results

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"
FUNDAMENTAL = 10.0  # Hz, from two_tone.cir's own kind=measure line


def main() -> int:
    trace = results.read_csv_file(FIXTURES / "two_tone.csv")
    print(f"read {len(trace)} points, signals: {', '.join(trace.names)}")
    print(f"uniformly stepped: {trace.is_uniformly_stepped()}")

    v = trace.samples("V(a)")

    print("\nmeasurements:")
    print(f"  peak            {measure.peak(v):.9g}")
    print(f"  peak-to-peak    {measure.peak_to_peak(v):.9g}")
    print(f"  mean (dt-wtd)   {measure.avg(v):.9g}")
    print(f"  rms             {measure.rms(v):.9g}")
    print(f"  thd             {measure.thd(v, FUNDAMENTAL, num_harmonics=4):.9g} %")

    spectrum = measure.four(v, FUNDAMENTAL, num_harmonics=4)
    print("\nharmonics:")
    for h in spectrum.harmonics:
        label = "dc" if h.order == 0 else f"h{h.order} @ {h.frequency:g} Hz"
        print(f"  {label:<16} magnitude {h.magnitude:>14.9g}   phase {h.phase_degrees:>10.4f} deg")

    # The same run's kind=measure log -- the simulator's own answer, for comparison.
    expected = logs.parse_measurement_log((FIXTURES / "two_tone.log").read_text())
    print("\nvs general-simulator's own kind=measure on this run:")
    for label, got, key in [
        ("h1 magnitude", spectrum.harmonics[1].magnitude, "MEAS_FOUR_h1_mag"),
        ("h2 magnitude", spectrum.harmonics[2].magnitude, "MEAS_FOUR_h2_mag"),
        ("thd_percent", spectrum.thd_percent, "MEAS_FOUR_thd_percent"),
    ]:
        print(f"  {label:<16} python {got:>16.12g}   simulator {expected[key]:>16.12g}")

    # And against the values two_tone.cir's own header comment derives by hand.
    assert abs(spectrum.harmonics[1].magnitude - 1.0) < 0.01
    assert abs(spectrum.harmonics[2].magnitude - 0.1) < 0.005
    assert abs(spectrum.thd_percent - 10.0) < 0.2
    print("\nmatches the hand-derived answers (h1=1.0, h2=0.1, THD=10%).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
