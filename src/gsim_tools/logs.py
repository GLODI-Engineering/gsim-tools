"""Structured parsing of the text ``general-simulator`` writes about a run.

Two separate streams, deliberately kept separate by the CLI itself and therefore here too:

* **stderr diagnostics.** Every failure exits via one ``eprintln!("error: {message}")`` in
  ``general-simulator-cli/src/main.rs``'s ``main``, so stderr is either empty or carries one
  ``error: ...`` line (a usage dump may follow it). :func:`parse_stderr` turns that line into a
  :class:`SimulatorError` with a :class:`ErrorKind` classification and whatever structured
  fields the message carries -- the offending netlist line number, the rejected flag value, the
  device or block name -- so calling code can branch on *which* failure happened without
  string-matching in every call site.

* **the measurement log.** ``kind=measure`` results are written to ``<netlist stem>.log`` as
  ``name = value`` lines (``write_measurements_log`` in the same file), matching real SPICE
  tools' ``.measure`` logging convention -- a log file, not stdout/stderr, so stdout stays pure
  CSV and stderr stays free of measurement text. A measurement that could not be evaluated is
  written as ``measurement '<name>' failed: <reason>`` instead, on its own, without failing the
  run: one bad measurement must never abort the others. :func:`parse_measurement_log` returns
  the numeric results; :func:`parse_measurement_failures` returns the failed ones; and
  :func:`parse_measurement_report` returns both together.

``parse_measurement_log`` is a direct port of ``general-simulator/tests/measure/_lib.py``'s
``parse_measurements``, extended only to also surface the failure lines that function silently
skipped (it returned ``{name: float}`` and dropped anything that didn't parse as a float --
fine for a test harness asserting on known-good fixtures, not fine for a library where a
silently-missing key is the worst possible way to learn a measurement failed).

Nothing in this module imports ``gsim_py``, numpy, or runs a subprocess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

__all__ = [
    "ErrorKind",
    "SimulatorError",
    "MeasurementFailure",
    "MeasurementReport",
    "parse_stderr",
    "parse_error_line",
    "parse_measurement_log",
    "parse_measurement_failures",
    "parse_measurement_report",
]


class ErrorKind(Enum):
    """What kind of failure a ``general-simulator`` ``error:`` line reports.

    The classifications follow the distinct ``Err(...)`` sites in
    ``general-simulator-cli/src/main.rs``'s ``run()``. ``UNKNOWN`` is not a defect -- it is the
    honest answer for a message this parser has no specific pattern for, and the full text is
    always preserved in :attr:`SimulatorError.message` either way.
    """

    USAGE = "usage"
    """No netlist argument, or an unrecognized flag; a usage dump follows."""

    BAD_ARGUMENT = "bad-argument"
    """A flag was given an unacceptable value (``unknown --mode``, ``unknown --format``,
    ``bad --dt``, or the mutually-exclusive fixed/adaptive step flags together)."""

    IO = "io"
    """A file could not be read or written (``reading <path>: ...``)."""

    PARSE = "parse"
    """The netlist (or its ``kind=measure`` declarations) did not parse; usually carries a
    ``line <n>:`` position."""

    DEVICE = "device"
    """A specific named device is unusable in the requested mode (``device '<name>': ...``)."""

    SOLVER = "solver"
    """The DAE runtime or DC solver rejected the model -- these arrive as the Rust
    ``DaeError`` variant's ``Debug`` form, so the variant name is the first token."""

    UNKNOWN = "unknown"
    """An ``error:`` line this parser has no more specific pattern for."""


@dataclass(frozen=True)
class SimulatorError:
    """One ``error: ...`` line from ``general-simulator``'s stderr, classified.

    ``message`` is always the full text after ``error: ``, never truncated or rewritten.
    ``fields`` holds whatever the pattern for this :attr:`kind` could extract -- for example
    ``{"path": "/tmp/x.cir"}`` for an :attr:`ErrorKind.IO`, ``{"flag": "--mode",
    "value": "foo", "expected": "'dc' or 'transient'"}`` for a :attr:`ErrorKind.BAD_ARGUMENT`,
    or ``{"variant": "Sig2PhysUsedAsCircuitNode"}`` for a :attr:`ErrorKind.SOLVER`.
    """

    kind: ErrorKind
    message: str
    line: Optional[int] = None
    fields: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        where = f" (netlist line {self.line})" if self.line is not None else ""
        return f"[{self.kind.value}]{where} {self.message}"


# One `eprintln!("error: {message}")` in main.rs's main() -- every failure exits through it.
_ERROR_LINE_RE = re.compile(r"^error:\s*(?P<message>.*)$")

# `parsing netlist: ...` / `parsing kind=measure declarations: ...`, whose inner message
# usually starts `line <n>: `.
_PARSING_RE = re.compile(r"^parsing (?P<what>[^:]+):\s*(?P<detail>.*)$", re.DOTALL)
_LINE_RE = re.compile(r"^line (?P<line>\d+):\s*(?P<detail>.*)$", re.DOTALL)

_READING_RE = re.compile(r"^(?P<verb>reading|creating|writing) (?P<path>[^:]+):\s*(?P<detail>.*)$")
_UNKNOWN_FLAG_VALUE_RE = re.compile(
    r"^unknown (?P<flag>--\S+) '(?P<value>[^']*)'(?: \(expected (?P<expected>.*)\))?$"
)
_UNRECOGNIZED_RE = re.compile(r"^unrecognized argument '(?P<value>[^']*)'")
_DEVICE_RE = re.compile(r"^device '(?P<device>[^']*)':\s*(?P<detail>.*)$", re.DOTALL)
# A Rust `{e:?}` dump: the variant name, then either ` { ... }` or `(...)`.
_DEBUG_VARIANT_RE = re.compile(r"^(?P<variant>[A-Z][A-Za-z0-9]*)\s*[{(]")


def parse_error_line(message: str) -> SimulatorError:
    """Classify one already-stripped ``error:`` message body into a :class:`SimulatorError`.

    Exposed separately from :func:`parse_stderr` so a caller who already has the message text
    (from a raised exception, a captured log, a CI annotation) can classify it without
    reconstructing the ``error: `` prefix.
    """
    message = message.strip()

    m = _PARSING_RE.match(message)
    if m:
        detail = m.group("detail").strip()
        fields = {"what": m.group("what").strip()}
        line_m = _LINE_RE.match(detail)
        if line_m:
            fields["detail"] = line_m.group("detail").strip()
            return SimulatorError(
                ErrorKind.PARSE, message, line=int(line_m.group("line")), fields=fields
            )
        fields["detail"] = detail
        return SimulatorError(ErrorKind.PARSE, message, fields=fields)

    m = _READING_RE.match(message)
    if m:
        return SimulatorError(
            ErrorKind.IO,
            message,
            fields={
                "verb": m.group("verb"),
                "path": m.group("path").strip(),
                "detail": m.group("detail").strip(),
            },
        )

    m = _UNKNOWN_FLAG_VALUE_RE.match(message)
    if m:
        fields = {"flag": m.group("flag"), "value": m.group("value")}
        if m.group("expected"):
            fields["expected"] = m.group("expected")
        return SimulatorError(ErrorKind.BAD_ARGUMENT, message, fields=fields)

    m = _UNRECOGNIZED_RE.match(message)
    if m:
        return SimulatorError(
            ErrorKind.USAGE, message, fields={"argument": m.group("value")}
        )

    if message.startswith("usage: general-simulator"):
        return SimulatorError(ErrorKind.USAGE, message)

    if message.startswith("--dt ") or message.startswith("bad --"):
        return SimulatorError(ErrorKind.BAD_ARGUMENT, message)
    if re.match(r"^--\S+ needs a value$", message):
        return SimulatorError(ErrorKind.BAD_ARGUMENT, message)

    m = _DEVICE_RE.match(message)
    if m:
        return SimulatorError(
            ErrorKind.DEVICE,
            message,
            fields={"device": m.group("device"), "detail": m.group("detail").strip()},
        )

    m = _DEBUG_VARIANT_RE.match(message)
    if m:
        return SimulatorError(
            ErrorKind.SOLVER, message, fields={"variant": m.group("variant")}
        )
    # A bare variant name with no payload is still a DaeError Debug dump.
    if re.fullmatch(r"[A-Z][A-Za-z0-9]*", message):
        return SimulatorError(ErrorKind.SOLVER, message, fields={"variant": message})

    return SimulatorError(ErrorKind.UNKNOWN, message)


def parse_stderr(stderr: str) -> list[SimulatorError]:
    """Parse every ``error: ...`` line out of a ``general-simulator`` stderr capture.

    Normally at most one, since ``main()`` prints one message and exits, but a list is
    returned rather than an ``Optional`` so a caller aggregating several runs' stderr (a batch
    script, a CI log) needs no special case. Continuation lines belonging to a message -- the
    usage dump that follows ``unrecognized argument``, or a multi-line explanation -- are
    appended to that message rather than being reported as separate errors.
    """
    errors: list[SimulatorError] = []
    pending: Optional[list[str]] = None

    def flush() -> None:
        if pending:
            errors.append(parse_error_line("\n".join(pending).strip()))

    for raw in stderr.splitlines():
        m = _ERROR_LINE_RE.match(raw)
        if m:
            flush()
            pending = [m.group("message")]
        elif pending is not None:
            pending.append(raw)
    flush()
    return errors


# ---------------------------------------------------------------------------
# The kind=measure log file
# ---------------------------------------------------------------------------

# `writeln!(writer, "{label} = {value}")` in write_measurements_log.
_MEASUREMENT_RE = re.compile(r"^(?P<name>\S+) = (?P<value>.+)$")
# `writeln!(writer, "measurement '{name}' failed: {e}")` in the same function.
_MEASUREMENT_FAILED_RE = re.compile(r"^measurement '(?P<name>[^']*)' failed:\s*(?P<reason>.*)$")


@dataclass(frozen=True)
class MeasurementFailure:
    """One ``measurement '<name>' failed: <reason>`` line from the measurement log.

    A failed measurement is *not* a failed run -- the CLI still exits zero, because one bad
    measurement must not abort the others. That makes these lines the only signal that a
    measurement you asked for produced nothing, which is why they are surfaced as data rather
    than skipped.
    """

    name: str
    reason: str


@dataclass(frozen=True)
class MeasurementReport:
    """Everything a ``<netlist stem>.log`` file said: the values and the failures."""

    values: dict[str, float]
    failures: list[MeasurementFailure]

    def __getitem__(self, name: str) -> float:
        try:
            return self.values[name]
        except KeyError:
            for failure in self.failures:
                if failure.name == name:
                    raise KeyError(
                        f"measurement {name!r} failed: {failure.reason}"
                    ) from None
            raise KeyError(
                f"no measurement {name!r} in this log; present: "
                f"{', '.join(sorted(self.values))}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self.values


def parse_measurement_report(log_text: str) -> MeasurementReport:
    """Parse a ``kind=measure`` log file into its values and its failures.

    Ported from ``general-simulator/tests/measure/_lib.py``'s ``parse_measurements``, with the
    failure lines that function dropped now captured. A ``four`` measurement's harmonic
    sub-results (``<name>_h1_mag``, ``<name>_h1_phase_deg``, ``<name>_dc``,
    ``<name>_thd_percent``) come back as ordinary extra keys, exactly as printed.
    """
    values: dict[str, float] = {}
    failures: list[MeasurementFailure] = []
    for raw in log_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        failed = _MEASUREMENT_FAILED_RE.match(line)
        if failed:
            failures.append(
                MeasurementFailure(failed.group("name"), failed.group("reason").strip())
            )
            continue
        m = _MEASUREMENT_RE.match(line)
        if not m:
            continue
        try:
            values[m.group("name")] = float(m.group("value"))
        except ValueError:
            continue
    return MeasurementReport(values, failures)


def parse_measurement_log(log_text: str) -> dict[str, float]:
    """The ``{name: value}`` numeric results from a ``kind=measure`` log file.

    The exact behavior of ``measure/_lib.py``'s ``parse_measurements``, kept as the common
    case. Use :func:`parse_measurement_report` when you also need to know whether a
    measurement *failed* rather than simply being absent.
    """
    return parse_measurement_report(log_text).values


def parse_measurement_failures(log_text: str) -> list[MeasurementFailure]:
    """Just the failed measurements from a ``kind=measure`` log file."""
    return parse_measurement_report(log_text).failures
