# AGENTS.md — `gsim-tools`

Pure-Python tooling over the `gsim_py` bindings: building schematics programmatically,
parsing results, and measuring waveforms.

## Authority order

1. **Mechanical gates** — the commands under **Required gates** below. If a gate fails, fix the
   cause; never bypass.
2. **This file** — what cannot be mechanized.
3. **`CONTRIBUTING.md`** — the same workflow written for a human contributor, at more length.
   This file and that one must not disagree; if they do, that is a bug in one of them.

## Development cycle

The gates below say how code is written. This says how the work is recorded, which matters for
the same reason: someone reads it later, and what they can reconstruct is what was written down.

1. **A large instruction becomes a milestone**, named for the outcome rather than the task
   ("Rules and diagnostics", not "implement the rules spec").
2. **One issue per deliverable phase or decision**, each stating the problem rather than the
   patch. The instruction as a whole — its reasoning, what it leaves open — is an issue in that
   milestone too, and the rest reference it with `Part of #N`.
3. **A branch and a pull request per issue**, the description ending `Closes #N`, assigned to the
   milestone. Nothing lands on the default branch directly.
4. **The merge closes the issue.** Closing it by hand loses the link between the issue and the
   commit that answered it, which is the whole point of the exercise.
5. **Anything found but not done becomes its own issue**, not a sentence in a pull request body.
   A merged pull request's prose is invisible; an open issue is not. A deliberate omission with
   a reason is worth more than a half-finished corner of a change.
6. **Gates run before pushing**, never after. A red branch spends someone's CI and tells the
   reader the author did not look.

Finish a repository's open pull request before starting new work in it, so review has one thing
to hold at a time.


## Required gates

```sh
uv venv && uv pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

CI runs the same three. `gsim_tools.measure` is a port of the Rust `gs-waveform-measurements`
crate and is cross-validated against it in `tests/test_measure_parity.py`: changing a
measurement here without changing it there leaves the two disagreeing, which is exactly what
that test exists to catch, so it is not a test to skip to make a change land.

## Project boundaries

- `gsim_py` and `gsim-core` live in a sibling repository and are **read-only** from here. A
  defect in the bindings is an issue there, not a workaround here.
- This package is not on PyPI. Installation is from a checkout, and any instruction that tells a
  reader otherwise is wrong.
