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
        # The rest of the compile registry's signal-flow families. machine.pmsm is
        # signal-flow despite modelling a physical machine: every one of its ports is
        # kind=signal (vd/vq/t_load in, id/iq/omega/theta_e out).
        ("machine.pmsm", "signal-flow"),
        ("math.fn3", "signal-flow"),
        ("math.table_lookup", "signal-flow"),
        ("logic.gate", "signal-flow"),
        ("logic.flip_flop", "signal-flow"),
        ("source.pulsewave", "signal-flow"),
        ("source.expwave", "signal-flow"),
        ("source.sffmwave", "signal-flow"),
        ("source.time", "signal-flow"),
        # A dependent source is an electrical device, like an independent one.
        ("source.dependent.vcvs", "physical-network"),
        ("source.dependent.cccs", "physical-network"),
        # goto/from are paradigm-agnostic tunnels, unlike routing.ground above.
        ("routing.goto", "routing"),
        ("routing.from", "routing"),
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


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _signal_chain():
    """A small multi-layer signal-flow graph with a fan-out and a feedback wire.

    ``src -> sum -> pi -> plant``, ``pi -> monitor`` (fan-out, so one column holds two
    blocks), and ``plant -> sum`` (the feedback that makes the graph cyclic, exactly as a real
    control loop is).
    """
    sch = S.Schematic("chain", "Chain", grid=10)
    src = sch.add("source.const", id="src", value=1.0)
    junction = sch.add("control.sum", id="sum", signs=["+", "-"])
    pi = sch.add("control.gain", id="pi", k=2.0)
    plant = sch.add("control.gain", id="plant", k=0.5)
    monitor = sch.add("control.gain", id="monitor", k=1.0)
    sch.connect(src.out, junction["in1"])
    sch.connect(junction.out, pi["in"])
    sch.connect(pi.out, plant["in"])
    sch.connect(pi.out, monitor["in"])
    feedback = sch.connect(plant.out, junction["in2"])
    return sch, feedback


@requires_gsim_py
def test_layered_layout_orders_blocks_along_the_signal_flow():
    sch, feedback = _signal_chain()
    pos = sch.layered_layout(skip_wires=[feedback.id])

    assert set(pos) == {"src", "sum", "pi", "plant", "monitor"}
    # Each stage strictly further along +x than the one feeding it.
    assert pos["src"][0] < pos["sum"][0] < pos["pi"][0] < pos["plant"][0]
    # The fan-out pair shares a column and is separated vertically instead.
    assert pos["plant"][0] == pos["monitor"][0]
    assert pos["plant"][1] != pos["monitor"][1]
    # And the model itself was rewritten, not just the return value.
    by_id = {b["id"]: tuple(b["position"]) for b in sch.model["blocks"]}
    assert by_id == pos


@requires_gsim_py
def test_layered_layout_leaves_no_two_blocks_overlapping():
    sch, feedback = _signal_chain()
    size = (100, 80)
    pos = sch.layered_layout(skip_wires=[feedback.id], default_size=size)

    rects = [(x, y, x + size[0], y + size[1]) for x, y in pos.values()]
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])


@requires_gsim_py
def test_layered_layout_snaps_to_the_grid_and_honours_the_origin():
    sch, feedback = _signal_chain()
    pos = sch.layered_layout(skip_wires=[feedback.id], origin=(400, 200))
    assert all(x % 10 == 0 and y % 10 == 0 for x, y in pos.values())
    assert min(x for x, _ in pos.values()) == 400
    assert min(y for _, y in pos.values()) == 200


@requires_gsim_py
def test_layered_layout_can_compute_without_applying():
    sch, feedback = _signal_chain()
    before = {b["id"]: tuple(b["position"]) for b in sch.model["blocks"]}
    pos = sch.layered_layout(skip_wires=[feedback.id], apply=False)
    after = {b["id"]: tuple(b["position"]) for b in sch.model["blocks"]}
    assert after == before
    assert pos != before  # it really did compute something


@requires_gsim_py
def test_layered_layout_accepts_an_explicit_edge_list_and_per_block_sizes():
    """An edge naming a block that is not in the model is ignored, not an error."""
    sch, _ = _signal_chain()
    pos = sch.layered_layout(
        edges=[("src", "plant"), ("plant", "GHOST")],
        bboxes={"src": (40, 40), "plant": (200, 40)},
    )
    assert pos["src"][0] < pos["plant"][0]
    # Only the one real edge constrains anything; everything else lands in the first column.
    assert pos["src"][0] == pos["sum"][0] == pos["pi"][0] == pos["monitor"][0]


def _mixed_paradigm_sheet():
    """A power stage plus the control chain around it -- the normal power-electronics shape.

    ``src -> gain -> converter -> switch`` is the control chain; the switch, the source, the
    resistor and the ground are a *physical network*, joined conductively with no signal
    direction at all and with the ground node joined to two of them.
    """
    sch = S.Schematic("mixed", "Mixed", grid=10)
    gnd = sch.add("routing.ground", id="gnd")
    vdc = sch.add("source.voltage.dc", id="Vdc", V=10.0)
    sw = sch.add(
        "passive.switch.ideal", id="SW", r_on=0.1, g_breakdown=0.0, v_breakdown=-100.0,
        g_off=1e-9, v_th=0.5, g_on=5.0,
    )
    rload = sch.add("passive.resistor", id="R", R=10.0)
    e = "physical-electrical"
    sch.connect(vdc.p, sw.d, e)
    sch.connect(sw.s, rload.p1, e)
    sch.connect(rload.p2, gnd.p, e)
    sch.connect(vdc.n, gnd.p, e)

    ref = sch.add("source.const", id="ref", value=0.5)
    duty = sch.add("control.gain", id="duty", k=1.0)
    gate = sch.add("converter.sig2phys", id="gate", params={"domain": "voltage"})
    sch.connect(ref.out, duty["in"])
    sch.connect(duty.out, gate["in"])
    sch.connect(gate.out, sw.g, e)  # the paradigm boundary
    return sch


@requires_gsim_py
def test_layered_layout_without_blocks_drags_the_power_stage_into_the_first_column():
    """Why ``blocks`` exists: every block takes part, so untouched ones pile into layer 0."""
    sch = _mixed_paradigm_sheet()
    pos = sch.layered_layout(apply=False)

    assert set(pos) == {"gnd", "Vdc", "SW", "R", "ref", "duty", "gate"}
    first_column = {b for b, (x, _) in pos.items() if x == min(x for x, _ in pos.values())}
    # The control chain's own head shares its column with three power-stage blocks.
    assert {"ref", "Vdc"} <= first_column


@requires_gsim_py
def test_layered_layout_over_a_subset_leaves_every_other_block_alone():
    sch = _mixed_paradigm_sheet()
    power = {"gnd", "Vdc", "SW", "R"}
    before = {b["id"]: tuple(b["position"]) for b in sch.model["blocks"]}
    sch.set_positions({b: (500, 500) for b in power})

    control = ["ref", "duty", "gate"]
    pos = sch.layered_layout({"ref": (40, 40)}, blocks=control, apply=True)

    # Only the named blocks come back, and only they moved.
    assert set(pos) == set(control)
    assert pos["ref"][0] < pos["duty"][0] < pos["gate"][0]
    after = {b["id"]: tuple(b["position"]) for b in sch.model["blocks"]}
    assert {b: after[b] for b in power} == {b: (500, 500) for b in power}
    assert before["ref"] == (0, 0)

    # The gate wire leaves the subset, so it constrains nothing and needed no skip_wires.
    assert "SW" not in pos


@requires_gsim_py
def test_layered_layout_blocks_ignores_an_id_the_model_does_not_have():
    sch = _mixed_paradigm_sheet()
    pos = sch.layered_layout(blocks=["ref", "duty", "GHOST"], apply=False)
    assert set(pos) == {"ref", "duty"}


@requires_gsim_py
def test_layered_layout_tolerates_a_pure_cycle_without_hanging():
    sch, _ = _signal_chain()
    pos = sch.layered_layout()  # feedback wire included: the graph is cyclic
    assert len(pos) == 5


@requires_gsim_py
def test_set_positions_applies_a_nudged_layout():
    """The compute-nudge-apply workflow: layout, move one block, put it back."""
    sch, feedback = _signal_chain()
    pos = sch.layered_layout(skip_wires=[feedback.id], apply=False)
    pos["monitor"] = (pos["monitor"][0], 999)
    sch.set_positions({"monitor": pos["monitor"], "UNKNOWN": (5, 5)})  # unknown id ignored

    by_id = {b["id"]: tuple(b["position"]) for b in sch.model["blocks"]}
    assert by_id["monitor"] == pos["monitor"]
    assert by_id["src"] == (0, 0)  # untouched blocks keep what they had


@requires_gsim_py
def test_place_without_overlap_steps_around_an_occupied_spot():
    sch = S.Schematic("place", grid=10)
    sch.add("control.gain", id="a", at=(0, 0), k=1.0)
    sch.add("control.gain", id="b", at=(500, 500), k=1.0)

    # Asking for exactly where 'a' sits pushes 'b' clear of it along x, and the model moves.
    x, y = sch.place_without_overlap("b", (0, 0), size=(100, 80), default_size=(100, 80))
    assert (x, y) != (0, 0)
    assert abs(x) >= 100
    assert y == 0
    assert (
        tuple(next(b for b in sch.model["blocks"] if b["id"] == "b")["position"]) == (x, y)
    )

    # A clear spot is returned untouched, and 'vertical' moves along y instead.
    assert sch.place_without_overlap("b", (600, 300), apply=False) == (600, 300)
    _, vy = sch.place_without_overlap("b", (0, 0), axis="vertical", apply=False)
    assert vy != 0


@requires_gsim_py
def test_place_without_overlap_rejects_an_unknown_block_and_a_bad_axis():
    sch = S.Schematic("place2", grid=10)
    sch.add("control.gain", id="a", k=1.0)
    with pytest.raises(S.UnknownBlockError):
        sch.place_without_overlap("nope", (0, 0))
    with pytest.raises(S.SchematicError, match="invalid axis"):
        sch.place_without_overlap("a", (0, 0), axis="diagonal")
