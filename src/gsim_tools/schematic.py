"""An ergonomic Python layer over ``gsim_py`` for building ``.gsim_sch`` schematics.

``gsim_py`` (the PyO3 bindings in the sibling ``gsim-core`` repo) is deliberately stateless and
string-in/string-out: every call takes JSON text, returns JSON text, and the caller threads the
model through by hand::

    model = gsim_py.new_model("rc", "RC", 10)
    r = json.loads(gsim_py.add_block(model, json.dumps({...})))
    model = json.dumps(r["model"])

That shape is right for a thin binding and wrong for a script author. This module keeps the
model JSON in a :class:`Schematic` object and hands back real Python values::

    from gsim_tools.schematic import Schematic

    sch = Schematic("rc_lowpass", "RC low-pass")
    vin = sch.add("source.voltage.dc", id="Vin", at=(0, 0), V=5.0)
    r1 = sch.add("passive.resistor", id="R1", at=(60, 0), R=1000.0)
    c1 = sch.add("passive.capacitor", id="C1", at=(120, 0), C=1e-6)
    gnd = sch.add("routing.ground", id="gnd", at=(60, 80))

    sch.connect(vin.p, r1.p1)
    sch.connect(r1.p2, c1.p1)
    sch.connect(c1.p2, gnd.p)
    sch.connect(vin.n, gnd.p)

    netlist = sch.compile()

**This is a wrapper, not a reimplementation.** Every document operation -- validating a block
spec, assigning ids, checking a wire's paradigm and port compatibility, laying blocks out,
compiling to a netlist -- is performed by ``gsim_py``, which performs it by calling
``gsim-core``. Nothing here hand-constructs ``.gsim_sch`` JSON beyond assembling the argument
objects each ``gsim_py`` function documents. Two conveniences are genuinely added on top, and
both are overridable defaults rather than rules: paradigm inference from a block's type prefix
(see :func:`infer_paradigm`) and wire-kind inference from the two endpoints' paradigms (see
:meth:`Schematic.connect`).

**Typed errors.** ``gsim_py`` raises ``ValueError`` whose message carries a *stable keyword
prefix* (``cross-paradigm: ...``, ``port-kind-mismatch: ...``, ``floating-port: ...``) exactly
so a caller can tell which failure happened. This module parses that prefix and raises the
matching :class:`SchematicError` subclass, so ``except PortKindMismatchError`` works and no
caller has to string-match. The original message is preserved verbatim on the exception.

**``gsim_py`` is optional.** It is a local Rust extension built with ``maturin``, not something
on PyPI, so it is not a hard dependency. Importing this module without it is fine; the first
call that actually needs it raises :class:`GsimPyNotAvailable` explaining how to build it. That
keeps :mod:`gsim_tools.results` and :mod:`gsim_tools.measure` usable with no Rust toolchain
anywhere on the system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Union

__all__ = [
    "Schematic",
    "Block",
    "Endpoint",
    "Wire",
    "CompileResult",
    "infer_paradigm",
    "SchematicError",
    "GsimPyNotAvailable",
    "DuplicateIdError",
    "DanglingWireError",
    "CrossParadigmError",
    "PortReservedKindError",
    "PortKindMismatchError",
    "PortFanInOnSignalError",
    "PortDimMismatchError",
    "UnknownBlockError",
    "InvalidBlockError",
    "UnknownSubcktDefError",
    "CyclicSubcktDefError",
    "InvalidSubcktError",
    "UnknownTypeError",
    "UnknownTargetError",
    "FloatingPortError",
    "UnconnectedRefError",
    "MissingParamError",
    "UnconnectedSubcktPortError",
    "UnknownVariantError",
    "VectorOutputPortError",
    "IndexedRefArityError",
    "UnresolvedFromTagError",
    "AmbiguousFromTagError",
    "CyclicFromTagError",
    "RoutingKindUnsupportedError",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SchematicError(Exception):
    """Base class for every failure this module reports.

    ``keyword`` is the stable prefix ``gsim_py`` emitted (``"cross-paradigm"``,
    ``"floating-port"``, ...), or ``None`` for a failure that carried no prefix. ``message`` is
    the full original text, unmodified -- this layer only classifies, it never rewrites or
    truncates what the Rust error said.
    """

    keyword: Optional[str] = None

    def __init__(self, message: str, keyword: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        if keyword is not None:
            self.keyword = keyword


class GsimPyNotAvailable(SchematicError, ImportError):
    """``gsim_py`` is not installed, and this operation needs it.

    Also an ``ImportError``, so ``except ImportError`` around an optional-feature probe works.
    """


# --- document-builder failures (gsim-py's builder_err_to_py keywords) ---


class DuplicateIdError(SchematicError):
    """A block, wire, or subckt-def id was already used."""


class DanglingWireError(SchematicError):
    """A wire endpoint names a block or port that does not exist."""


class CrossParadigmError(SchematicError):
    """A wire tried to join two different paradigms directly.

    A ``signal`` wire and a ``physical-electrical`` wire cannot connect without a
    ``converter.*`` block between them -- the ``.gsim_sch`` format's §8.3 cross-paradigm
    boundary rule.
    """


class PortReservedKindError(SchematicError):
    """A port's kind is reserved by a later schema version and cannot be wired yet."""


class PortKindMismatchError(SchematicError):
    """The two endpoints' port kinds are incompatible (§8.1)."""


class PortFanInOnSignalError(SchematicError):
    """More than one driver was wired into a single ``signal`` input port.

    A signal-flow input takes exactly one source; joining several needs an explicit
    ``routing.merge``.
    """


class PortDimMismatchError(SchematicError):
    """The two endpoints' port dimensions differ (§8.2)."""


class UnknownBlockError(SchematicError):
    """An operation named a block id that is not in the model."""


class InvalidBlockError(SchematicError):
    """A block spec was rejected -- an invalid id or type pattern, or an invalid rotation."""


class UnknownSubcktDefError(SchematicError):
    """A ``def_ref`` names no ``SubcktDef`` in the model."""


class CyclicSubcktDefError(SchematicError):
    """The ``.SUBCKT`` definitions reference each other in a cycle."""


class InvalidSubcktError(SchematicError):
    """A ``SubcktDef`` was rejected -- an invalid id pattern or port kind."""


# --- compiler failures (gsim-py's compile_err_to_py keywords) ---


class UnknownTypeError(SchematicError):
    """The compiler has no emitter for one of the block types used."""


class UnknownTargetError(SchematicError):
    """The requested compile target is not one the compiler knows."""


class FloatingPortError(SchematicError):
    """A port that must be wired was left unconnected."""


class UnconnectedRefError(SchematicError):
    """A block referenced another block's output that is not actually wired."""


class MissingParamError(SchematicError):
    """A block is missing a parameter its emitter requires."""


class UnconnectedSubcktPortError(SchematicError):
    """A ``.SUBCKT`` instance left one of the definition's ports unwired."""


class UnknownVariantError(SchematicError):
    """A parameter selected a variant the emitter does not implement."""


class VectorOutputPortError(SchematicError):
    """A vector-valued output port was used where a scalar is required."""


class IndexedRefArityError(SchematicError):
    """An indexed reference's arity does not match the referenced port's dimension."""


class UnresolvedFromTagError(SchematicError):
    """A ``routing.from`` tag has no matching ``routing.goto``."""


class AmbiguousFromTagError(SchematicError):
    """A ``routing.from`` tag matches more than one ``routing.goto``."""


class CyclicFromTagError(SchematicError):
    """``routing.goto``/``routing.from`` tags form a cycle."""


class RoutingKindUnsupportedError(SchematicError):
    """A routing block was used on a wire kind it does not support."""


#: Maps the stable keyword prefix ``gsim-py`` puts in front of every ``BuilderError`` and
#: ``CompileError`` message (see ``builder_err_to_py`` / ``compile_err_to_py`` in
#: ``gsim-core/crates/gsim-py/src/lib.rs``) to the exception class raised for it. A keyword not
#: in this table -- a newly added error variant, say -- falls back to a plain
#: :class:`SchematicError` carrying the keyword, never to a silent success.
_ERROR_CLASSES: dict[str, type[SchematicError]] = {
    "duplicate": DuplicateIdError,
    "dangling": DanglingWireError,
    "cross-paradigm": CrossParadigmError,
    "port-reserved-kind": PortReservedKindError,
    "port-kind-mismatch": PortKindMismatchError,
    "port-fan-in-on-signal": PortFanInOnSignalError,
    "port-dim-mismatch": PortDimMismatchError,
    "unknown-block": UnknownBlockError,
    "invalid-block": InvalidBlockError,
    "unknown-subckt-def": UnknownSubcktDefError,
    "cyclic-subckt-def": CyclicSubcktDefError,
    "invalid-subckt": InvalidSubcktError,
    "unknown-type": UnknownTypeError,
    "unknown-target": UnknownTargetError,
    "floating-port": FloatingPortError,
    "unconnected-ref": UnconnectedRefError,
    "missing-param": MissingParamError,
    "unconnected-subckt-port": UnconnectedSubcktPortError,
    "unknown-variant": UnknownVariantError,
    "vector-output-port": VectorOutputPortError,
    "indexed-ref-arity": IndexedRefArityError,
    "unresolved-from-tag": UnresolvedFromTagError,
    "ambiguous-from-tag": AmbiguousFromTagError,
    "cyclic-from-tag": CyclicFromTagError,
    "routing-kind-unsupported": RoutingKindUnsupportedError,
}


def _raise_typed(exc: Exception) -> "SchematicError":
    """Translate a ``gsim_py`` ``ValueError`` into the matching typed exception.

    ``gsim_py`` formats every builder/compiler failure as ``"<keyword>: <real message>"``
    precisely so this is possible from the message text alone. The keyword is matched
    longest-first, because several keywords share a prefix (``unknown-subckt-def`` vs
    ``unknown-subckt-def`` inside a longer phrase, ``port-kind-mismatch`` vs ``port-...``).
    """
    message = str(exc)
    keyword, _, _ = message.partition(":")
    keyword = keyword.strip()
    cls = _ERROR_CLASSES.get(keyword)
    if cls is not None:
        return cls(message, keyword)
    return SchematicError(message, keyword if keyword and " " not in keyword else None)


# ---------------------------------------------------------------------------
# gsim_py access
# ---------------------------------------------------------------------------

_GSIM_PY_HELP = """gsim_py is not installed, and gsim_tools.schematic needs it.

gsim_py is a local Rust extension module (PyO3 bindings for gsim-core), not a PyPI package, so
it has to be built from the sibling gsim-core repository:

    cd <path to>/gsim-core/crates/gsim-py
    python3 -m venv .venv && source .venv/bin/activate
    pip install maturin
    maturin develop --release

That installs gsim_py into the active virtualenv; install gsim-tools into the same one
(`pip install -e <path to>/gsim-tools`) and this module works.

gsim_tools.results, gsim_tools.measure and gsim_tools.logs do not need gsim_py at all -- only
schematic building does."""


def _gsim_py() -> Any:
    """Import ``gsim_py`` on first use, with one clear, actionable error if it is missing.

    Deliberately *not* a module-level import: ``import gsim_tools.schematic`` must not fail on
    a machine with no Rust toolchain, and neither must importing :mod:`gsim_tools` itself.
    """
    try:
        import gsim_py  # noqa: PLC0415 - optional dependency, imported lazily on purpose
    except ImportError as exc:
        raise GsimPyNotAvailable(_GSIM_PY_HELP) from exc
    return gsim_py


# ---------------------------------------------------------------------------
# Paradigm inference
# ---------------------------------------------------------------------------

#: Default paradigm per block-type prefix. These follow the assignments the real
#: ``gsim-core/examples/*.py`` scripts use for each type family; ``routing.*`` is mapped to
#: ``physical-network`` because that is what those examples pass for ``routing.ground`` (the
#: format does have a paradigm-agnostic ``routing`` paradigm, but a ground pin in a real
#: circuit is wired as a physical-network block). Anything not listed here has no default and
#: must be given ``paradigm=`` explicitly -- guessing wrong would produce a confusing
#: cross-paradigm rejection later instead of a clear error now.
_PARADIGM_BY_PREFIX: dict[str, str] = {
    "passive": "physical-network",
    "thermal": "physical-network",
    "routing": "physical-network",
    "control": "signal-flow",
    "power": "signal-flow",
    "extensibility": "signal-flow",
    # Every remaining signal-flow family in the compile registry. `machine.*` is listed here
    # even though a real machine is a physical device: `machine.pmsm`'s ports are all
    # `kind=signal` (vd/vq/t_load in, id/iq/omega/theta_e out), so it is a signal-flow block.
    "machine": "signal-flow",
    "math": "signal-flow",
    "logic": "signal-flow",
    "converter": "converter",
    "hierarchy": "hierarchy",
    "annotation": "annotation",
}

#: Types whose family default does not apply. ``source.*`` splits: an independent electrical
#: source is physical-network, a signal-flow constant/waveform generator is signal-flow.
#: ``routing.goto``/``routing.from`` are paradigm-agnostic tunnels (paradigm ``routing``),
#: unlike ``routing.ground``, which the prefix default correctly calls physical-network.
_PARADIGM_BY_TYPE: dict[str, str] = {
    "source.const": "signal-flow",
    "source.step": "signal-flow",
    "source.ramp": "signal-flow",
    "source.sine": "signal-flow",
    "source.sinwave": "signal-flow",
    "source.square": "signal-flow",
    "source.pulse": "signal-flow",
    "source.pulsewave": "signal-flow",
    "source.expwave": "signal-flow",
    "source.sffmwave": "signal-flow",
    "source.time": "signal-flow",
    "routing.goto": "routing",
    "routing.from": "routing",
}


def infer_paradigm(type_: str) -> Optional[str]:
    """The default paradigm for a block ``type``, or ``None`` if there is no safe default.

    A convenience so ordinary scripts need not repeat ``paradigm="physical-network"`` on every
    resistor. It is only ever a default: :meth:`Schematic.add` accepts an explicit
    ``paradigm=`` that always wins, and a type with no inferred paradigm raises a clear error
    asking for one rather than guessing.
    """
    if type_ in _PARADIGM_BY_TYPE:
        return _PARADIGM_BY_TYPE[type_]
    prefix = type_.split(".", 1)[0]
    if prefix == "source":
        # An electrical source (source.voltage.*, source.current.*, source.dependent.*) is
        # physical-network; anything else under source.* is a signal generator, handled by
        # _PARADIGM_BY_TYPE.
        parts = type_.split(".")
        if len(parts) >= 2 and parts[1] in ("voltage", "current", "dependent"):
            return "physical-network"
        return None
    return _PARADIGM_BY_PREFIX.get(prefix)


#: The wire kind used when both endpoints agree on a paradigm. ``converter``/``routing``/
#: ``hierarchy``/``annotation``/``spice-subckt`` blocks bridge paradigms (they have no
#: paradigm of their own for this purpose), so a wire touching one takes its kind from the
#: other endpoint.
_WIRE_KIND_BY_PARADIGM: dict[str, str] = {
    "physical-network": "physical-electrical",
    "signal-flow": "signal",
}

_BRIDGING_PARADIGMS = frozenset(
    {"converter", "routing", "hierarchy", "annotation", "spice-subckt"}
)


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Endpoint:
    """One end of a wire: a block id plus a port id.

    Normally obtained as ``block.p1`` or ``block["p1"]`` rather than constructed directly.
    """

    block: str
    port: str

    def as_json(self) -> str:
        return json.dumps({"block": self.block, "port": self.port})

    def __repr__(self) -> str:
        return f"{self.block}.{self.port}"


#: Anything :meth:`Schematic.connect` accepts as an endpoint: an :class:`Endpoint`, or a plain
#: ``("block_id", "port_id")`` tuple for callers who never made a :class:`Block` object.
EndpointLike = Union[Endpoint, "tuple[str, str]", "Sequence[str]"]


def _as_endpoint(value: EndpointLike) -> Endpoint:
    if isinstance(value, Endpoint):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return Endpoint(str(value[0]), str(value[1]))
    raise TypeError(
        f"expected an Endpoint (e.g. r1.p1) or a (block_id, port_id) pair, got {value!r}"
    )


class Block:
    """A block that has been added to a :class:`Schematic`.

    Access a port by attribute (``r1.p1``) or by item (``r1["p1"]``); both yield an
    :class:`Endpoint`. No port-name validation happens here -- port names are a property of the
    block's GSF contract, which this layer deliberately does not duplicate.

    Note what that means in practice: ``gsim_py``'s ``connect`` validates the *block* id (an
    unknown one is :class:`DanglingWireError`) but accepts any port name on a known block, so a
    mistyped port makes a wire to nowhere and the block's real port is then left unwired. That
    surfaces at :meth:`Schematic.compile` as :class:`FloatingPortError` naming the port that
    never got a node -- late, but specific, and from the authoritative check rather than a
    guess made here.
    """

    __slots__ = ("id", "type", "paradigm", "_schematic")

    def __init__(self, schematic: "Schematic", block_id: str, type_: str, paradigm: str):
        self.id = block_id
        self.type = type_
        self.paradigm = paradigm
        self._schematic = schematic

    def port(self, name: str) -> Endpoint:
        """This block's ``name`` port as an :class:`Endpoint`."""
        return Endpoint(self.id, name)

    def __getitem__(self, name: str) -> Endpoint:
        return Endpoint(self.id, name)

    def __getattr__(self, name: str) -> Endpoint:
        if name.startswith("_"):
            raise AttributeError(name)
        return Endpoint(self.id, name)

    def set(self, **params: Any) -> "Block":
        """Set one or more parameters on this block, in place. Returns ``self`` for chaining."""
        for name, value in params.items():
            self._schematic.set_param(self.id, name, value)
        return self

    def __repr__(self) -> str:
        return f"Block(id={self.id!r}, type={self.type!r}, paradigm={self.paradigm!r})"


@dataclass(frozen=True)
class Wire:
    """A wire that has been added to a :class:`Schematic`."""

    id: str
    from_: Endpoint
    to: Endpoint
    kind: str

    def __repr__(self) -> str:
        return f"Wire({self.id!r}: {self.from_!r} -> {self.to!r}, kind={self.kind!r})"


@dataclass(frozen=True)
class CompileResult:
    """The output of :meth:`Schematic.compile_full`.

    ``netlist`` is the emitted text; ``notes`` are the compiler's own info-level remarks about
    decisions it made. :meth:`Schematic.compile` returns just the netlist, since that is what
    almost every caller wants.
    """

    netlist: str
    notes: list[str]

    def __str__(self) -> str:
        return self.netlist

    def write(self, path: Any) -> None:
        """Write the netlist text to ``path``."""
        import pathlib  # noqa: PLC0415 - only needed on this path

        pathlib.Path(path).write_text(self.netlist)


# ---------------------------------------------------------------------------
# Schematic
# ---------------------------------------------------------------------------


class Schematic:
    """A ``.gsim_sch`` schematic being built up in Python.

    Holds the model JSON that ``gsim_py``'s stateless functions thread between calls, so a
    caller never has to. Every mutating method calls straight into ``gsim_py`` and replaces the
    held model with what came back; nothing is cached or shadowed, so the object's state is
    always exactly what ``gsim-core`` last produced.

    Usable as a context manager, which is only sugar for scoping -- there is nothing to close::

        with Schematic("rc", "RC low-pass") as sch:
            r = sch.add("passive.resistor", R=1e3)
            ...
    """

    def __init__(
        self,
        id: str,
        name: Optional[str] = None,
        grid: int = 10,
        *,
        contracts: Optional[dict[str, Any]] = None,
    ) -> None:
        """Create an empty schematic.

        ``contracts`` is an optional ``{block type: ComponentContract}`` map. Supplying it
        makes ``gsim_py``'s per-port compatibility check resolve each endpoint's real kind and
        dim from the block's own params instead of approximating from its paradigm -- which is
        what makes :class:`PortKindMismatchError` exact and :class:`PortDimMismatchError`
        possible at all. Omitting it changes nothing that used to work: an unregistered type
        keeps the coarser check. It is passed per ``connect`` call because ``gsim_py`` is
        stateless; this object holds it so callers only build it once.
        """
        gsim_py = _gsim_py()
        self.id = id
        self.name = name if name is not None else id
        self.contracts: dict[str, Any] = dict(contracts or {})
        self._model: str = gsim_py.new_model(id, self.name, grid)
        self._blocks: dict[str, Block] = {}
        self._wires: list[Wire] = []

    # -- context manager (scoping sugar only) --------------------------------

    def __enter__(self) -> "Schematic":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    # -- state ---------------------------------------------------------------

    @property
    def model_json(self) -> str:
        """The current model as ``gsim_py``-shaped JSON text.

        The escape hatch: anything this wrapper does not cover can be done by passing this to
        ``gsim_py`` directly and feeding the result back through :meth:`replace_model`.
        """
        return self._model

    @property
    def model(self) -> dict[str, Any]:
        """The current model as a parsed Python dict (a fresh copy on every access)."""
        return json.loads(self._model)

    def replace_model(self, model_json: str) -> None:
        """Replace the held model with ``model_json`` -- the counterpart to :attr:`model_json`.

        Used after calling a ``gsim_py`` function this wrapper does not expose. Block and wire
        bookkeeping is *not* re-derived, so ids this object never saw will not appear in
        :attr:`blocks`; the model itself is always authoritative.
        """
        self._model = model_json

    @property
    def blocks(self) -> "dict[str, Block]":
        """Blocks added through this object, by id."""
        return dict(self._blocks)

    @property
    def wires(self) -> list[Wire]:
        """Wires added through this object, in creation order."""
        return list(self._wires)

    def __getitem__(self, block_id: str) -> Block:
        try:
            return self._blocks[block_id]
        except KeyError:
            raise KeyError(
                f"no block {block_id!r} in this schematic; have: "
                f"{', '.join(sorted(self._blocks))}"
            ) from None

    def __contains__(self, block_id: str) -> bool:
        return block_id in self._blocks

    def __repr__(self) -> str:
        return (
            f"Schematic(id={self.id!r}, blocks={len(self._blocks)}, "
            f"wires={len(self._wires)})"
        )

    # -- building ------------------------------------------------------------

    def add(
        self,
        type: str,
        *,
        id: Optional[str] = None,
        at: "tuple[int, int]" = (0, 0),
        paradigm: Optional[str] = None,
        domain: str = "electrical",
        rotation: int = 0,
        mirror: Optional[str] = None,
        fidelity: Optional[str] = None,
        label: Optional[str] = None,
        ic: Optional[dict[str, Any]] = None,
        comment_before: Optional[Union[str, Sequence[str]]] = None,
        comment_after: Optional[Union[str, Sequence[str]]] = None,
        params: Optional[dict[str, Any]] = None,
        **param_kwargs: Any,
    ) -> Block:
        """Add a block and return it.

        ``type`` is a ``.gsim_sch`` block type such as ``"passive.resistor"``. Parameters may
        be given as keyword arguments (``sch.add("passive.resistor", R=1e3)``) or, when a
        parameter name would collide with one of this method's own keywords, in the explicit
        ``params=`` dict; the two are merged with ``params=`` taking precedence.

        ``at`` is the block's ``(x, y)`` position on the grid. Position matters for the saved
        document and for layout, not for the compiled netlist, so a script that only wants a
        netlist can leave every block at the default and call :meth:`auto_arrange` -- or not
        bother at all.

        ``id`` is optional; ``gsim_py`` assigns one when it is omitted, and the assigned id is
        what the returned :class:`Block` carries. ``paradigm`` defaults to
        :func:`infer_paradigm` on the type.

        ``ic`` is a ``{name: value}`` map of initial conditions (a ``.gsim_sch`` block's ``ic``
        field is a map, not a scalar -- ``ic={"V": 0.0}`` on a capacitor, not ``ic=0.0``).

        ``comment_before``/``comment_after`` are authoring comments emitted as ``*``-prefixed
        SPICE comment lines immediately above/below this block's card(s); a single string
        containing newlines becomes several comment lines.

        Raises :class:`DuplicateIdError`, :class:`InvalidBlockError`, or another
        :class:`SchematicError` subclass on rejection.
        """
        gsim_py = _gsim_py()

        resolved_paradigm = paradigm or infer_paradigm(type)
        if resolved_paradigm is None:
            raise SchematicError(
                f"no default paradigm known for block type {type!r} -- pass "
                "paradigm='physical-network' / 'signal-flow' / 'converter' / 'routing' / "
                "'hierarchy' / 'annotation' / 'spice-subckt' explicitly"
            )

        merged: dict[str, Any] = dict(param_kwargs)
        merged.update(params or {})

        spec: dict[str, Any] = {
            "type": type,
            "paradigm": resolved_paradigm,
            "domain": domain,
            "position": list(at),
            "rotation": rotation,
            "params": merged,
        }
        if id is not None:
            spec["id"] = id
        for key, value in (
            ("mirror", mirror),
            ("fidelity", fidelity),
            ("label", label),
            ("ic", ic),
            ("comment_before", comment_before),
            ("comment_after", comment_after),
        ):
            if value is not None:
                spec[key] = value

        try:
            raw = gsim_py.add_block(self._model, json.dumps(spec))
        except ValueError as exc:
            raise _raise_typed(exc) from exc

        result = json.loads(raw)
        self._model = json.dumps(result["model"])
        block = Block(self, result["block_id"], type, resolved_paradigm)
        self._blocks[block.id] = block
        return block

    def connect(
        self,
        from_: EndpointLike,
        to: EndpointLike,
        kind: Optional[str] = None,
        *,
        id: Optional[str] = None,
        net: Optional[str] = None,
        label: Optional[str] = None,
        pinned: Optional[bool] = None,
        waypoints: Optional[Sequence[Sequence[int]]] = None,
    ) -> Wire:
        """Wire two endpoints together and return the resulting :class:`Wire`.

        Endpoints may be :class:`Endpoint` objects (``r1.p2``) or ``("R1", "p2")`` pairs.

        ``kind`` defaults to the wire kind implied by the two blocks' paradigms:
        ``physical-electrical`` between two ``physical-network`` blocks, ``signal`` between two
        ``signal-flow`` blocks, and the non-bridging side's kind when one endpoint is a
        bridging block (``converter.*``, ``routing.*``, a subckt instance). If the two sides
        disagree, no kind is inferred and one must be given -- inferring here would only pick a
        side and turn a real cross-paradigm mistake into a confusing downstream error.

        Raises :class:`DanglingWireError` for an endpoint that does not exist,
        :class:`CrossParadigmError` for a paradigm boundary crossed without a converter,
        :class:`PortKindMismatchError` / :class:`PortDimMismatchError` /
        :class:`PortFanInOnSignalError` for the per-port §8.1/§8.2 rules.
        """
        gsim_py = _gsim_py()
        a, b = _as_endpoint(from_), _as_endpoint(to)
        resolved_kind = kind or self._infer_wire_kind(a, b)

        opts: dict[str, Any] = {}
        if id is not None:
            opts["id"] = id
        if net is not None:
            opts["net"] = net
        if label is not None:
            opts["label"] = label
        if pinned is not None:
            opts["pinned"] = pinned
        opts["waypoints"] = [list(p) for p in (waypoints or [])]

        try:
            raw = gsim_py.connect(
                self._model,
                a.as_json(),
                b.as_json(),
                resolved_kind,
                json.dumps(opts),
                json.dumps(self.contracts) if self.contracts else None,
            )
        except ValueError as exc:
            raise _raise_typed(exc) from exc

        result = json.loads(raw)
        self._model = json.dumps(result["model"])
        wire = Wire(result["wire_id"], a, b, resolved_kind)
        self._wires.append(wire)
        return wire

    def connect_all(
        self, pairs: Iterable["tuple[EndpointLike, EndpointLike]"], kind: Optional[str] = None
    ) -> list[Wire]:
        """Make several wires in one call. Sugar over :meth:`connect`, same semantics."""
        return [self.connect(a, b, kind) for a, b in pairs]

    def _infer_wire_kind(self, a: Endpoint, b: Endpoint) -> str:
        """The wire kind implied by two endpoints' blocks' paradigms.

        A bridging block (converter/routing/hierarchy/annotation/spice-subckt) contributes no
        kind of its own, so the other end decides -- and so does a block this object never
        added, since there is nothing to read a paradigm from. When one side still implies a
        kind, that is used and ``gsim_py`` gets the final say (an endpoint naming a block that
        does not exist comes back as :class:`DanglingWireError` from the authoritative check,
        which is a better error than anything invented here). Only when *no* side implies a
        kind, or the two disagree, is a "pass kind=" error raised.
        """
        kinds: set[str] = set()
        unknown: list[str] = []
        for endpoint in (a, b):
            block = self._blocks.get(endpoint.block)
            if block is None:
                unknown.append(endpoint.block)
                continue
            if block.paradigm in _BRIDGING_PARADIGMS:
                continue
            wire_kind = _WIRE_KIND_BY_PARADIGM.get(block.paradigm)
            if wire_kind is not None:
                kinds.add(wire_kind)

        if len(kinds) == 1:
            return next(iter(kinds))
        if unknown:
            raise SchematicError(
                f"cannot infer a wire kind: block(s) {', '.join(sorted(unknown))} were not "
                "added through this Schematic -- pass kind='physical-electrical' or "
                "kind='signal' explicitly"
            )
        if not kinds:
            raise SchematicError(
                f"cannot infer a wire kind between {a!r} and {b!r}: both blocks bridge "
                "paradigms, so neither implies one -- pass kind= explicitly"
            )
        raise SchematicError(
            f"cannot infer a wire kind between {a!r} and {b!r}: they imply different kinds "
            f"({', '.join(sorted(kinds))}) -- a paradigm boundary needs a converter.* block, "
            "or pass kind= explicitly if that is really what you mean"
        )

    def set_param(self, block_id: str, name: str, value: Any) -> "Schematic":
        """Set one parameter on one block. Returns ``self`` for chaining."""
        gsim_py = _gsim_py()
        try:
            self._model = gsim_py.set_param(
                self._model, block_id, name, json.dumps(value)
            )
        except ValueError as exc:
            raise _raise_typed(exc) from exc
        return self

    def add_param(self, name: str, value: Any) -> "Schematic":
        """Add a model-level (schematic-wide) parameter. Returns ``self`` for chaining."""
        gsim_py = _gsim_py()
        try:
            self._model = gsim_py.add_model_param(self._model, name, json.dumps(value))
        except ValueError as exc:
            raise _raise_typed(exc) from exc
        return self

    def remove(self, block_id: str) -> "Schematic":
        """Remove a block, cascading to every wire touching it. Returns ``self``."""
        gsim_py = _gsim_py()
        self._model = gsim_py.remove_block(self._model, block_id)
        self._blocks.pop(block_id, None)
        self._wires = [
            w for w in self._wires if block_id not in (w.from_.block, w.to.block)
        ]
        return self

    # -- subcircuits ---------------------------------------------------------

    def add_subckt_def(self, definition: dict[str, Any]) -> str:
        """Add a reusable ``.SUBCKT`` definition; returns its assigned id.

        ``definition`` is a ``SubcktDef`` dict -- ``{"id", "name", "ports": [{"id", "kind"}],
        "params"?, "blocks"?, "wires"?, "instances"?}``, exactly the shape ``gsim_py``'s
        ``add_subckt_def`` documents. Passed through unchanged.
        """
        gsim_py = _gsim_py()
        try:
            raw = gsim_py.add_subckt_def(self._model, json.dumps(definition))
        except ValueError as exc:
            raise _raise_typed(exc) from exc
        result = json.loads(raw)
        self._model = json.dumps(result["model"])
        return result["def_id"]

    def instantiate_subckt(
        self,
        def_ref: str,
        *,
        id: Optional[str] = None,
        at: "tuple[int, int]" = (0, 0),
        rotation: int = 0,
        domain: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
        label: Optional[str] = None,
    ) -> Block:
        """Instantiate a ``SubcktDef`` as a top-level block; returns it.

        The created block's wireable ports are exactly the referenced definition's
        ``ports[].id`` list -- wire them with ordinary :meth:`connect` calls. Its paradigm is
        ``spice-subckt``, which bridges paradigms, so wires touching it take their kind from
        the other end (or from an explicit ``kind=``).
        """
        gsim_py = _gsim_py()
        spec: dict[str, Any] = {
            "def_ref": def_ref,
            "position": list(at),
            "rotation": rotation,
        }
        if id is not None:
            spec["id"] = id
        if domain is not None:
            spec["domain"] = domain
        if params is not None:
            spec["params"] = params
        if label is not None:
            spec["label"] = label
        try:
            raw = gsim_py.instantiate_subckt(self._model, json.dumps(spec))
        except ValueError as exc:
            raise _raise_typed(exc) from exc
        result = json.loads(raw)
        self._model = json.dumps(result["model"])
        block = Block(self, result["block_id"], f"subckt:{def_ref}", "spice-subckt")
        self._blocks[block.id] = block
        return block

    # -- layout --------------------------------------------------------------

    def auto_arrange(
        self,
        bboxes: Optional[dict[str, "tuple[int, int]"]] = None,
        column_gap: int = 40,
        row_gap: int = 40,
    ) -> "Schematic":
        """Rewrite every block's position by ``gsim_py``'s own auto-arrange. Returns ``self``.

        ``bboxes`` maps block id to ``(width, height)``; a block missing from the map is
        treated as zero-sized, as ``gsim_py.auto_arrange`` itself does. Sizes come from
        evaluating each block's GSF contract, which this layer does not do for you -- pass what
        you have, or nothing if approximate placement is fine.
        """
        gsim_py = _gsim_py()
        self._model = gsim_py.auto_arrange(
            self._model,
            json.dumps({k: list(v) for k, v in (bboxes or {}).items()}),
            column_gap,
            row_gap,
        )
        return self

    # -- output --------------------------------------------------------------

    def compile(self, target: str = "general-simulator") -> str:
        """Compile to netlist text for ``target``.

        Raises :class:`FloatingPortError`, :class:`MissingParamError`,
        :class:`UnknownTypeError` and the rest of the compiler's typed failures on rejection.
        Use :meth:`compile_full` when you also want the compiler's info-level notes.
        """
        return self.compile_full(target).netlist

    def compile_full(self, target: str = "general-simulator") -> CompileResult:
        """Compile to netlist text *and* the compiler's info-level notes."""
        gsim_py = _gsim_py()
        try:
            raw = gsim_py.compile(self._model, target)
        except ValueError as exc:
            raise _raise_typed(exc) from exc
        result = json.loads(raw)
        return CompileResult(result["netlist"], list(result.get("notes", [])))

    def save(self, path: Optional[Any] = None) -> str:
        """Serialize to a full ``.gsim_sch`` document; writes it to ``path`` when given.

        Returns the document JSON text either way.
        """
        gsim_py = _gsim_py()
        document = gsim_py.save_document(self._model)
        if path is not None:
            import pathlib  # noqa: PLC0415 - only needed on this path

            pathlib.Path(path).write_text(document)
        return document

    @classmethod
    def load(cls, path_or_json: Any) -> "Schematic":
        """Load a ``.gsim_sch`` document (a path, or the JSON text itself).

        The document is validated and canonically re-serialized by ``gsim_py.load_document``,
        so a malformed or non-v2 document is rejected here rather than surfacing later. Blocks
        loaded this way populate :attr:`blocks`, so wire-kind inference works on them too.
        """
        import pathlib  # noqa: PLC0415 - only needed on this path

        gsim_py = _gsim_py()
        text = str(path_or_json)
        if not text.lstrip().startswith("{"):
            text = pathlib.Path(text).read_text()
        try:
            document = gsim_py.load_document(text)
        except ValueError as exc:
            raise _raise_typed(exc) from exc

        parsed = json.loads(document)
        model = parsed.get("model", parsed)
        sch = cls.__new__(cls)
        sch.id = model.get("id", "")
        sch.name = model.get("name", sch.id)
        sch.contracts = {}
        sch._model = json.dumps(model)
        sch._blocks = {}
        sch._wires = []
        for raw_block in model.get("blocks", []):
            block = Block(
                sch,
                raw_block["id"],
                raw_block.get("type", ""),
                raw_block.get("paradigm", ""),
            )
            sch._blocks[block.id] = block
        for raw_wire in model.get("wires", []):
            sch._wires.append(
                Wire(
                    raw_wire.get("id", ""),
                    Endpoint(raw_wire["from"]["block"], raw_wire["from"]["port"]),
                    Endpoint(raw_wire["to"]["block"], raw_wire["to"]["port"]),
                    raw_wire.get("kind", ""),
                )
            )
        return sch
