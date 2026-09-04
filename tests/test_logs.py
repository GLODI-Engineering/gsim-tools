"""Tests for :mod:`gsim_tools.logs`.

The stderr messages asserted here are real ``general-simulator`` output, captured by running
the release binary against deliberately broken inputs, not invented text. The measurement-log
tests run against the committed ``.log`` fixtures the real binary wrote.
"""

from __future__ import annotations

import pytest

from conftest import fixture_text
from gsim_tools.logs import (
    ErrorKind,
    MeasurementFailure,
    parse_error_line,
    parse_measurement_failures,
    parse_measurement_log,
    parse_measurement_report,
    parse_stderr,
)


# ---------------------------------------------------------------------------
# stderr diagnostics
# ---------------------------------------------------------------------------


def test_io_error_names_the_path():
    errors = parse_stderr(
        "error: reading /tmp/nonexistent.cir: No such file or directory (os error 2)\n"
    )
    assert len(errors) == 1
    err = errors[0]
    assert err.kind is ErrorKind.IO
    assert err.fields["verb"] == "reading"
    assert err.fields["path"] == "/tmp/nonexistent.cir"
    assert err.fields["detail"] == "No such file or directory (os error 2)"
    # The full original message is never rewritten or truncated.
    assert err.message.startswith("reading /tmp/nonexistent.cir:")


def test_parse_error_extracts_the_netlist_line_number():
    err = parse_error_line(
        "parsing kind=measure declarations: line 1: device TransmissionLine requires at "
        "least 4 nodes, got 0 tokens"
    )
    assert err.kind is ErrorKind.PARSE
    assert err.line == 1
    assert err.fields["what"] == "kind=measure declarations"
    assert err.fields["detail"].startswith("device TransmissionLine")


def test_parse_error_without_a_line_number():
    err = parse_error_line("parsing netlist: something went wrong")
    assert err.kind is ErrorKind.PARSE
    assert err.line is None
    assert err.fields["detail"] == "something went wrong"


@pytest.mark.parametrize(
    ("message", "flag", "value", "expected"),
    [
        ("unknown --mode 'foo' (expected 'dc' or 'transient')", "--mode", "foo",
         "'dc' or 'transient'"),
        ("unknown --format 'qqq' (expected 'csv' or 'raw')", "--format", "qqq",
         "'csv' or 'raw'"),
    ],
)
def test_bad_flag_value_is_broken_into_flag_value_and_expectation(message, flag, value, expected):
    err = parse_error_line(message)
    assert err.kind is ErrorKind.BAD_ARGUMENT
    assert err.fields == {"flag": flag, "value": value, "expected": expected}


def test_mutually_exclusive_step_flags():
    err = parse_error_line(
        "--dt (fixed step) and --dt-max/--dt-min/--dt-init/--reltol/--abstol (adaptive step) "
        "are mutually exclusive — pick one"
    )
    assert err.kind is ErrorKind.BAD_ARGUMENT


def test_unrecognized_argument_absorbs_the_usage_dump_that_follows():
    stderr = (
        "error: unrecognized argument '--frobnicate'\n"
        "usage: general-simulator <netlist> [--devices <file>] [--mode dc|transient] "
        "[--tfinal T]\n"
        "     --dt fixes the step size every step\n"
    )
    errors = parse_stderr(stderr)
    # One failure, not three: the usage lines belong to the message, they are not errors.
    assert len(errors) == 1
    assert errors[0].kind is ErrorKind.USAGE
    assert errors[0].fields["argument"] == "--frobnicate"
    assert "usage: general-simulator" in errors[0].message


def test_device_error_names_the_device():
    err = parse_error_line(
        "device 'SW1': every gate is block-driven (gate=block), which needs --mode transient, "
        "not 'dc' (a DC operating point has no notion of a block's time-stepped state)"
    )
    assert err.kind is ErrorKind.DEVICE
    assert err.fields["device"] == "SW1"
    assert err.fields["detail"].startswith("every gate is block-driven")


def test_dae_error_debug_dump_yields_the_variant_name():
    # Most DaeErrors reach stderr as Rust's `{e:?}`, so the variant name leads the message.
    err = parse_error_line(
        'Sig2PhysUsedAsCircuitNode { block: "SIGV", element: "V1", node: "a" }: element '
        "'V1' wires node 'a', but 'SIGV' is a kind=sig2phys converter"
    )
    assert err.kind is ErrorKind.SOLVER
    assert err.fields["variant"] == "Sig2PhysUsedAsCircuitNode"

    assert parse_error_line("SingularMatrix").kind is ErrorKind.SOLVER
    assert parse_error_line("NonConvergence(42)").fields["variant"] == "NonConvergence"


def test_unclassifiable_message_is_kept_intact_not_dropped():
    err = parse_error_line("something nobody anticipated")
    assert err.kind is ErrorKind.UNKNOWN
    assert err.message == "something nobody anticipated"


def test_clean_stderr_yields_no_errors():
    assert parse_stderr("") == []
    assert parse_stderr("some unrelated chatter\n") == []


def test_str_of_an_error_is_readable():
    err = parse_error_line("parsing netlist: line 7: bad token")
    assert str(err) == "[parse] (netlist line 7) parsing netlist: line 7: bad token"


# ---------------------------------------------------------------------------
# kind=measure log files
# ---------------------------------------------------------------------------


def test_parse_measurement_log_on_a_real_fixture():
    values = parse_measurement_log(fixture_text("ramp.log"))
    assert values["MEAS_MAX"] == pytest.approx(20.0, abs=1e-6)
    assert values["MEAS_INTEG"] == pytest.approx(100.0, abs=0.01)
    assert set(values) == {
        "MEAS_MAX", "MEAS_MIN", "MEAS_MAXAT", "MEAS_MINAT", "MEAS_PP", "MEAS_AVG",
        "MEAS_RMS", "MEAS_INTEG", "MEAS_DERIVAT", "MEAS_FINDAT", "MEAS_WHEN",
        "MEAS_FINDWHEN", "MEAS_DERIVWHEN", "MEAS_TRIGTARG",
    }


def test_four_sub_results_are_ordinary_keys():
    # A `four` measurement expands into <name>_dc / _h<n>_mag / _h<n>_phase_deg /
    # _thd_percent lines; nothing special is done with them, they are keys like any other.
    values = parse_measurement_log(fixture_text("sine.log"))
    assert values["MEAS_FOUR_h1_mag"] == pytest.approx(2.0, abs=5e-3)
    assert values["MEAS_FOUR_h1_phase_deg"] == pytest.approx(-90.0, abs=0.5)
    assert values["MEAS_FOUR_thd_percent"] < 0.1


def test_failed_measurements_are_surfaced_not_silently_missing():
    """The behavior this module deliberately adds over measure/_lib.py's parse_measurements.

    A failed measurement exits zero and writes only this line, so a parser that skips it makes
    "my measurement failed" indistinguishable from "I typoed the name".
    """
    log = (
        "MEAS_OK = 1.5\n"
        "measurement 'MEAS_BAD' failed: measurement window contains fewer than two samples\n"
        "MEAS_ALSO_OK = 2.5\n"
    )
    report = parse_measurement_report(log)
    assert report.values == {"MEAS_OK": 1.5, "MEAS_ALSO_OK": 2.5}
    assert report.failures == [
        MeasurementFailure(
            "MEAS_BAD", "measurement window contains fewer than two samples"
        )
    ]
    assert parse_measurement_failures(log) == report.failures
    # The narrow helper keeps the old shape exactly.
    assert parse_measurement_log(log) == report.values

    # Looking up a failed measurement says *why*, rather than "no such key".
    assert report["MEAS_OK"] == 1.5
    with pytest.raises(KeyError, match="failed: measurement window"):
        report["MEAS_BAD"]
    with pytest.raises(KeyError, match="MEAS_OK"):
        report["MEAS_TYPO"]


def test_empty_log_is_an_empty_report():
    # A netlist with no kind=measure lines makes the CLI write no log file at all, so an
    # empty string is a normal input.
    report = parse_measurement_report("")
    assert report.values == {}
    assert report.failures == []


def test_non_numeric_value_lines_are_ignored():
    assert parse_measurement_log("MEAS_X = not-a-number\nMEAS_Y = 3.0\n") == {"MEAS_Y": 3.0}
