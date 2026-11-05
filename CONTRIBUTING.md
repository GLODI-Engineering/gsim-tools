# Contributing

Thanks for considering a contribution to `gsim-tools`, the pure-Python tooling over the simulator.

This repository is part of the `general-simulator` toolchain. The same
workflow applies across all of them, so that a change can be traced from the
problem that prompted it to the commit that closed it.

## The workflow

**1. A milestone, when the work is bigger than one change.**

Anything that will take more than a single pull request gets a milestone —
a release, a feature with several parts, a migration. The milestone is what
answers "is this finished?", and it is the only thing that can.

Small, self-contained fixes do not need one.

**2. An issue, always.**

Every change starts from an issue, including your own. Write what is wrong
or missing and how you would know it was fixed — not the solution you have
in mind. An issue that only says what to do cannot be checked against
reality later.

Attach it to its milestone if it has one.

**3. A branch off `master`.**

Name it for the work: `fix/wire-endpoint-validation`, not `patch-2`.

**4. Commits that say why.**

Conventional Commits, enforced where a `commit-msg` hook is installed:
`feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`,
`revert`, with `!` for a breaking change.

The body is for the reasoning: what was actually wrong, what you ruled out,
what you decided not to do. The diff already says what changed. Reference
the issue — `Refs #12` while work is in progress, `Closes #12` on the commit
that finishes it, so the issue closes with the commit that closed it rather
than by hand afterwards.

**5. A pull request, with CI green.**

CI runs the same gate you ran locally. **A red build is not reviewable** —
fix it or say why it is unrelated. Never merge around it.

**6. Close with the commit, not after it.**

`Closes #12` in the merge means the issue carries a permanent link to the
change that resolved it. Closing an issue by hand loses that, and six months
later nobody can tell which commit it was.

A milestone closes when its issues do. If an issue in it turns out not to be
needed, close it saying so — do not let it sit so the milestone never ends.

## The gate

Run this before opening a pull request. CI runs the same thing.

```sh
uv venv && uv pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

`gsim_tools.measure` is a port of the Rust `gs-waveform-measurements`
crate and is cross-validated against it in `tests/test_measure_parity.py`.
Changing a measurement's behaviour here without changing it there leaves the
two disagreeing, which the parity test exists to prevent — do not skip it to
make a change land.

Watch the complexity of anything built on `interpolate`. It is the primitive
every value-at-an-arbitrary-time measurement uses, and when it was
accidentally O(n) the whole library went quadratic: `average_power` over a
50 000-sample run took about a minute.

## Reporting a bug

Include what you ran, what happened and what you expected. For anything
involving a circuit, attach the smallest netlist or schematic that shows it —
a reproduction that takes a minute to run is worth more than a paragraph of
description.

If the simulator produced a number you believe is wrong, say what you
expected and why, with the analytic expression if there is one. "The output
is wrong" cannot be acted on; "V_out should be V_in·D = 14.4 V and I measure
9.8 V" can.
