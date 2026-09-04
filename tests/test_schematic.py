"""Tests for :mod:`gsim_tools.schematic`.

Split in two on purpose:

* The tests at the top need **no** ``gsim_py`` -- they cover the parts this module owns
  outright: the optional-dependency guard, paradigm/wire-kind inference, endpoint coercion, and
  the mapping from ``gsim_py``'s stable keyword prefixes to typed exceptions. These run
  everywhere, which is the point: importing this module on a machine with no Rust toolchain
  must work.
* The rest are marked ``gsim_py`` and skipped unless the extension is built, since they drive
  real ``gsim_py`` calls.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from gsim_tools import schematic as S

def has_gsim_py() -> bool:
    try:
        import gsim_py  # noqa: F401
    except ImportError:
        return False
    return True


requires_gsim_py = pytest.mark.skipif(
    not has_gsim_py(),
    reason="gsim_py not built -- see gsim_tools.schematic's docstring for maturin instructions",
)


# ---------------------------------------------------------------------------
# No gsim_py needed
# ---------------------------------------------------------------------------


def test_importing_the_module_never_requires_gsim_py():
    """The whole reason the gsim_py import is lazy.

    Checked in a subprocess with ``gsim_py`` forced unimportable, because on this machine it
    may well be built -- an in-process check would prove nothing. Importing
    ``gsim_tools`` itself and ``gsim_tools.schematic`` must both succeed, and so must a
    ``results``/``measure`` round trip, since keeping those usable with no Rust toolchain is
    the entire reason for the lazy import.
    """
    script = textwrap.dedent(
        """
        import sys
        sys.modules["gsim_py"] = None  # any import of it now raises ImportError
        import gsim_tools
        import gsim_tools.schematic as S
        from gsim_tools import measure, results
        trace = results.read_csv("t,V(a)\\n0,0\\n1,2\\n")
        assert measure.integ(trace.samples("V(a)")) == 1.0
        assert S.Schematic is not None
        print("ok")
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_missing_gsim_py_raises_one_actionable_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gsim_py":
            raise ImportError("no module named gsim_py")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(S.GsimPyNotAvailable) as excinfo:
        S.Schematic("x")
    message = str(excinfo.value)
    # The error has to say what to install and how, not just "ImportError".
    assert "maturin develop" in message
    assert "gsim-core" in message
    assert "gsim_tools.results" in message  # and that the rest of the package still works
    # It is also an ImportError, so an optional-feature probe can catch it generically.
    assert isinstance(excinfo.value, ImportError)


@pytest.mark.parametrize(
    ("type_", "expected"),
    [
        ("passive.resistor", "physical-network"),
        ("passive.capacitor", "physical-network"),
        ("thermal.mass", "physical-network"),
        ("routing.ground", "physical-network"),
        ("source.voltage.dc", "physical-network"),
        ("source.current.dc", "physical-network"),
        ("control.gain", "signal-flow"),
        ("power.pwm1", "signal-flow"),
        ("source.const", "signal-flow"),
        ("extensibility.pyblock", "signal-flow"),
        ("converter.sig2phys", "converter"),
        ("hierarchy.subsystem", "hierarchy"),
        ("annotation.note", "annotation"),
        # No safe default: better a clear error than a guess that fails downstream.
        ("something.unknown", None),
        ("source.mystery", None),
    ],
)
def test_paradigm_inference_matches_the_real_examples(type_, expected):
    assert S.infer_paradigm(type_) == expected


def test_endpoint_coercion_accepts_tuples_and_endpoints():
    ep = S.Endpoint("R1", "p1")
    assert S._as_endpoint(ep) is ep
    assert S._as_endpoint(("R1", "p1")) == ep
    assert S._as_endpoint(["R1", "p1"]) == ep
    assert repr(ep) == "R1.p1"
    with pytest.raises(TypeError, match="Endpoint"):
        S._as_endpoint("R1.p1")


@pytest.mark.parametrize(
    ("message", "cls"),
    [
        ("cross-paradigm: a signal wire cannot join a physical-electrical one", S.CrossParadigmError),
        ("port-kind-mismatch: port 'in' is signal, port 'p1' is physical-electrical",
         S.PortKindMismatchError),
        ("port-dim-mismatch: [3] vs [1]", S.PortDimMismatchError),
        ("port-fan-in-on-signal: 'in' already has a driver", S.PortFanInOnSignalError),
        ("port-reserved-kind: 'physical-magnetic' is reserved", S.PortReservedKindError),
        ("duplicate: block id 'R1' already exists", S.DuplicateIdError),
        ("dangling: no block 'R9'", S.DanglingWireError),
        ("unknown-block: R9", S.UnknownBlockError),
        ("invalid-block: bad id pattern", S.InvalidBlockError),
        ("cyclic-subckt-def: a -> b -> a", S.CyclicSubcktDefError),
        ("floating-port: C1.p2 is unwired", S.FloatingPortError),
        ("missing-param: R1 needs 'R'", S.MissingParamError),
        ("unknown-type: mystery.block", S.UnknownTypeError),
        ("unknown-target: spice3f5", S.UnknownTargetError),
        ("vector-output-port: out is [3]", S.VectorOutputPortError),
        ("ambiguous-from-tag: 'clk'", S.AmbiguousFromTagError),
        ("routing-kind-unsupported: physical-electrical", S.RoutingKindUnsupportedError),
    ],
)
def test_keyword_prefixes_map_to_typed_exceptions(message, cls):
    err = S._raise_typed(ValueError(message))
    assert isinstance(err, cls)
    assert isinstance(err, S.SchematicError)
    # The original message survives intact -- classification never replaces detail.
    assert err.message == message
    assert err.keyword == message.split(":")[0]


def test_an_unrecognized_keyword_still_yields_a_schematic_error():
    """A newly added gsim-core error variant must not become a silent success or a bare crash."""
    err = S._raise_typed(ValueError("brand-new-variant: something happened"))
    assert type(err) is S.SchematicError
    assert err.keyword == "brand-new-variant"
    assert err.message == "brand-new-variant: something happened"


def test_a_message_with_no_keyword_prefix_is_handled():
    err = S._raise_typed(ValueError("invalid model JSON: expected value at line 1"))
    assert type(err) is S.SchematicError
    assert err.keyword is None


# ---------------------------------------------------------------------------
# Real gsim_py calls
# ---------------------------------------------------------------------------


@requires_gsim_py
def test_build_and_compile_an_rc_lowpass():
    sch = S.Schematic("rc", "RC low-pass")
    vin = sch.add("source.voltage.dc", id="Vin", at=(0, 0), V=5.0)
    r1 = sch.add("passive.resistor", id="R1", at=(60, 0), R=1000.0)
    c1 = sch.add("passive.capacitor", id="C1", at=(120, 0), C=1e-6)
    gnd = sch.add("routing.ground", id="gnd", at=(60, 80))

    # Wire kind is inferred -- every one of these is physical-network.
    w = sch.connect(vin.p, r1.p1, net="in")
    assert w.kind == "physical-electrical"
    sch.connect(r1.p2, c1.p1, net="out")
    sch.connect(c1.p2, gnd.p)
    sch.connect(vin.n, gnd.p)

    assert len(sch.blocks) == 4
    assert len(sch.wires) == 4
    assert "R1" in sch and sch["R1"].type == "passive.resistor"

    netlist = sch.compile()
    # `net=` named the nodes, so these are the real emitted cards.
    assert "Vin in 0 5" in netlist
    assert "R1 in out 1000" in netlist
    assert "C1 out 0 1e-6" in netlist


@requires_gsim_py
def test_compile_full_returns_notes_and_write_helper(tmp_path):
    sch = S.Schematic("rc")
    r1 = sch.add("passive.resistor", id="R1", R=1000.0)
    gnd = sch.add("routing.ground", id="gnd")
    vin = sch.add("source.voltage.dc", id="Vin", V=1.0)
    sch.connect(vin.p, r1.p1)
    sch.connect(r1.p2, gnd.p)
    sch.connect(vin.n, gnd.p)

    result = sch.compile_full()
    assert isinstance(result.notes, list)
    assert str(result) == result.netlist
    path = tmp_path / "out.cir"
    result.write(path)
    assert path.read_text() == result.netlist


@requires_gsim_py
def test_ids_are_assigned_when_omitted():
    sch = S.Schematic("auto")
    a = sch.add("passive.resistor", R=1.0)
    b = sch.add("passive.resistor", R=2.0)
    assert a.id and b.id and a.id != b.id
    assert sch[a.id] is a


@requires_gsim_py
def test_duplicate_id_raises_the_typed_exception():
    sch = S.Schematic("dup")
    sch.add("passive.resistor", id="R1", R=1.0)
    with pytest.raises(S.DuplicateIdError) as excinfo:
        sch.add("passive.resistor", id="R1", R=2.0)
    assert excinfo.value.keyword == "duplicate"
    # Still a ValueError-free, specific type -- the whole point of the mapping.
    assert not isinstance(excinfo.value, ValueError)


@requires_gsim_py
def test_unknown_block_endpoint_raises_dangling_wire_error():
    sch = S.Schematic("dangle")
    r1 = sch.add("passive.resistor", id="R1", R=1.0)
    with pytest.raises(S.DanglingWireError) as excinfo:
        sch.connect(r1.p2, ("NO_SUCH_BLOCK", "p"), kind="physical-electrical")
    assert excinfo.value.keyword == "dangling"


@requires_gsim_py
def test_a_mistyped_port_name_surfaces_at_compile_as_a_floating_port():
    """Documents where a port typo actually gets caught, since it is not at connect() time.

    gsim_py validates the block id but accepts any port name on a known block, so
    ``r1.typo_port`` makes a wire to a port that does not exist and leaves the real ``p2``
    unwired. The compiler is what notices, and it names the port precisely.
    """
    sch = S.Schematic("typo")
    vin = sch.add("source.voltage.dc", id="V1", V=1.0)
    r1 = sch.add("passive.resistor", id="R1", R=1.0)
    gnd = sch.add("routing.ground", id="gnd")
    sch.connect(vin.p, r1.p1)
    sch.connect(r1.typo_port, gnd.p)  # accepted here...
    sch.connect(vin.n, gnd.p)

    with pytest.raises(S.FloatingPortError, match="R1.p2"):  # ...caught here
        sch.compile()


@requires_gsim_py
def test_cross_paradigm_wire_is_rejected_with_its_own_type():
    """A signal-flow output wired straight into a physical-network pin: §8.3's boundary rule."""
    sch = S.Schematic("boundary")
    k = sch.add("control.gain", id="K1", k=2.0)
    r = sch.add("passive.resistor", id="R1", R=1.0)
    assert k.paradigm == "signal-flow"
    assert r.paradigm == "physical-network"

    # Inference refuses to pick a side, because picking one would turn a real modelling
    # mistake into a confusing downstream error.
    with pytest.raises(S.SchematicError, match="different kinds"):
        sch.connect(k.out, r.p1)

    # Forcing a kind gets the authoritative rejection from gsim_py itself.
    with pytest.raises(S.CrossParadigmError):
        sch.connect(k.out, r.p1, kind="signal")


@requires_gsim_py
def test_signal_wires_are_inferred_between_signal_flow_blocks():
    sch = S.Schematic("sig")
    src = sch.add("source.const", id="U1", value=1.0)
    gain = sch.add("control.gain", id="K1", k=2.5)
    w = sch.connect(src.out, gain["in"])
    assert w.kind == "signal"


@requires_gsim_py
def test_one_known_endpoint_is_enough_to_infer_a_wire_kind():
    """A raw tuple endpoint is fine as long as the other side implies a kind.

    And when the tuple names a block that does not exist, the error comes from gsim_py's own
    check, not from a guess made here -- which is the more useful message of the two.
    """
    sch = S.Schematic("partial")
    vin = sch.add("source.voltage.dc", id="V1", V=1.0)
    r1 = sch.add("passive.resistor", id="R1", R=1.0)
    w = sch.connect(vin.p, ("R1", "p1"))
    assert w.kind == "physical-electrical"

    with pytest.raises(S.DanglingWireError):
        sch.connect(r1.p2, ("SOMEWHERE_ELSE", "p"))


def test_no_wire_kind_can_be_inferred_between_two_bridging_blocks():
    """Both sides bridging: nothing implies a kind, so this asks rather than guessing."""
    sch = S.Schematic.__new__(S.Schematic)
    sch._blocks = {
        "CV1": S.Block(sch, "CV1", "converter.sig2phys", "converter"),
        "G1": S.Block(sch, "G1", "routing.goto", "routing"),
    }
    # routing.* defaults to physical-network, so force the paradigm-agnostic case explicitly.
    sch._blocks["G1"].paradigm = "hierarchy"
    with pytest.raises(S.SchematicError, match="both blocks bridge paradigms"):
        sch._infer_wire_kind(S.Endpoint("CV1", "out"), S.Endpoint("G1", "in"))


@requires_gsim_py
def test_set_param_and_add_param_and_remove():
    sch = S.Schematic("mutate")
    r1 = sch.add("passive.resistor", id="R1", R=1000.0)
    gnd = sch.add("routing.ground", id="gnd")
    sch.connect(r1.p2, gnd.p)

    r1.set(R=2200.0)
    assert next(b for b in sch.model["blocks"] if b["id"] == "R1")["params"]["R"] == 2200.0

    sch.add_param("temp", 27.0)
    # Model-level params are a list of {"name", "value"} records in the document.
    assert {"name": "temp", "value": 27.0} in sch.model["params"]

    # Removing a block cascades to its wires, in the model and in the bookkeeping.
    sch.remove("R1")
    assert "R1" not in sch
    assert sch.wires == []
    assert all(b["id"] != "R1" for b in sch.model["blocks"])


@requires_gsim_py
def test_save_and_load_round_trip(tmp_path):
    sch = S.Schematic("roundtrip", "Round trip")
    vin = sch.add("source.voltage.dc", id="Vin", at=(0, 0), V=5.0)
    r1 = sch.add("passive.resistor", id="R1", at=(60, 0), R=1000.0)
    gnd = sch.add("routing.ground", id="gnd", at=(60, 80))
    sch.connect(vin.p, r1.p1)
    sch.connect(r1.p2, gnd.p)
    sch.connect(vin.n, gnd.p)
    original = sch.compile()

    path = tmp_path / "roundtrip.gsim_sch"
    sch.save(path)
    assert path.exists()

    loaded = S.Schematic.load(path)
    assert set(loaded.blocks) == {"Vin", "R1", "gnd"}
    assert loaded["R1"].type == "passive.resistor"
    assert len(loaded.wires) == 3
    # The real test: the reloaded document compiles to the same netlist.
    assert loaded.compile() == original


@requires_gsim_py
def test_comments_reach_the_netlist_as_comment_lines():
    sch = S.Schematic("commented")
    vin = sch.add("source.voltage.dc", id="Vin", V=5.0)
    r1 = sch.add(
        "passive.resistor",
        id="R1",
        R=1000.0,
        comment_before="series element",
        comment_after="tolerance not modeled",
    )
    gnd = sch.add("routing.ground", id="gnd")
    sch.connect(vin.p, r1.p1)
    sch.connect(r1.p2, gnd.p)
    sch.connect(vin.n, gnd.p)

    lines = sch.compile().splitlines()
    i = next(n for n, line in enumerate(lines) if line.startswith("R1 "))
    assert lines[i - 1] == "* series element"
    assert lines[i + 1] == "* tolerance not modeled"


@requires_gsim_py
def test_unknown_paradigm_type_asks_for_one_instead_of_guessing():
    sch = S.Schematic("guess")
    with pytest.raises(S.SchematicError, match="no default paradigm"):
        sch.add("mystery.thing")
    # An explicit paradigm gets past this layer -- gsim-core accepts any well-formed type at
    # add_block time and only rejects an unknown one when it has to emit a card for it, so
    # that is where the typed error shows up.
    sch.add("mystery.thing", id="M1", paradigm="signal-flow")
    with pytest.raises(S.UnknownTypeError):
        sch.compile()


@requires_gsim_py
def test_context_manager_and_model_escape_hatch():
    with S.Schematic("ctx") as sch:
        sch.add("passive.resistor", id="R1", R=1.0)
        text = sch.model_json
        assert '"R1"' in text
        # replace_model puts a model built by a raw gsim_py call back in.
        sch.replace_model(text)
        assert sch.model["id"] == "ctx"
