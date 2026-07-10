"""Foundry unified plugin.

Consolidates RoseTTAFold3 (rf3_predict), RFdiffusion3 (rfd3_design), and
LigandMPNN (sequence_design) into a single plugin that runs in the
shared foundry pixi environment. Sub-engines load lazily on first use
to conserve GPU memory; checkpoint discovery happens eagerly at init
(fast glob, no model loading).

Catalog:

Ops (mutate state, return assembly bytes):

- ``rf3_predict(num_recycles)`` — STREAM. RF3. Reads the focused or
  session-wide assembly via ``init`` / ``update_assembly``.
  ``creates_entities=True``.
- ``rfd3_design(length, contig, num_designs, num_steps, step_scale,
  save_trajectories)`` — STREAM. RFD3. Same assembly source.
- ``mpnn_design(num_sequences, temperature)`` — INVOKE. LigandMPNN.
  Samples sequences for the focused chain and commits the best.
  ``creates_entities=True``.

Queries (read state, return query-defined data):

- ``sequence_design(num_sequences, temperature)`` — MPNN. Returns N
  candidate sequences for the focused protein chain (no entity
  mutation; user picks one and a separate ``apply_sequence`` op
  commits it). Selection-derived ``fixed_positions``: each
  ``ResidueRef`` in ``DispatchContext.selection`` is treated as a fixed
  residue.
"""

import os
import platform
import threading
import traceback
import warnings
from typing import Any

# set before torch is imported (CPU fallback for unimplemented MPS ops)
if platform.system() == "Darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

from foldit_plugin_sdk import (
    PluginInterface,
    PollOutcome,
    DispatchContext,
    make_param_value,
    weights,
)
from foldit_plugin_sdk.cache_utils import weights_dir_from_config
from foldit_plugin_sdk.checkpoint_utils import find_checkpoint
from foldit_plugin_sdk.logging_config import get_logger
from foldit_plugin_sdk.multiprocessing_utils import configure_multiprocessing
from foldit_plugin_sdk.proto import plugin_pb2

logger = get_logger(__name__)

# Configure multiprocessing early (fixes sys.executable for PyO3)
configure_multiprocessing()

# Models the download_weights op installs (via foundry_cli.install_model)
# into <cache_dir>/rc_foundry/. Unlike the URL-table plugins, foundry's
# weights come from rc-foundry's own installer, so its download is a
# per-model loop, not a WeightSpec list.
FOUNDRY_MODELS = ["rf3", "rfd3", "proteinmpnn", "ligandmpnn"]


# Module-level utilities (shared across all sub-engines). Stateless helpers.

def kabsch_align(mobile, target):
    """Align mobile coords to target via Kabsch (SVD). (N, 3) numpy arrays."""
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    mobile_centered = mobile - mobile_center
    target_centered = target - target_center

    H = mobile_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)

    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T

    aligned = (mobile_centered @ R.T) + target_center
    return aligned


def kabsch_transform(mobile_anchors, target_anchors):
    """Compute (R, t) mapping mobile_anchors → target_anchors."""
    mc = mobile_anchors.mean(axis=0)
    tc = target_anchors.mean(axis=0)
    Mc = mobile_anchors - mc
    Tc = target_anchors - tc
    H = Mc.T @ Tc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = tc - R @ mc
    return R, t


def apply_kabsch(coords, R, t):
    return (coords @ R.T) + t


def parse_contig_motif_residues(contig_str):
    import re
    motif = []
    for part in contig_str.split(","):
        part = part.strip()
        if not part or part == "/0":
            continue
        if not any(c.isalpha() for c in part):
            continue
        m = re.match(r'([A-Za-z])(\d+)(?:-[A-Za-z]?(\d+))?', part)
        if m:
            chain = m.group(1)
            start = int(m.group(2))
            end = int(m.group(3)) if m.group(3) else start
            for rid in range(start, end + 1):
                motif.append((chain, rid))
    return motif


def align_to_input_frame(output_aa, input_aa, contig=None, saved_transform=None):
    if contig is not None:
        return _align_via_motif_anchors(
            output_aa, input_aa, contig, saved_transform=saved_transform,
        )

    in_keys = set(zip(input_aa.chain_id.tolist(), input_aa.res_id.tolist()))
    out_keys = set(zip(output_aa.chain_id.tolist(), output_aa.res_id.tolist()))
    common = sorted(in_keys & out_keys)

    if not common:
        return output_aa

    in_cas, out_cas = [], []
    for chain, rid in common:
        in_mask = (
            (input_aa.chain_id == chain)
            & (input_aa.res_id == rid)
            & (np.char.strip(input_aa.atom_name) == "CA")
        )
        out_mask = (
            (output_aa.chain_id == chain)
            & (output_aa.res_id == rid)
            & (np.char.strip(output_aa.atom_name) == "CA")
        )
        if np.any(in_mask) and np.any(out_mask):
            in_cas.append(input_aa.coord[in_mask][0])
            out_cas.append(output_aa.coord[out_mask][0])

    if len(in_cas) < 1:
        return output_aa

    return _apply_alignment(output_aa, in_cas, out_cas)


def _align_via_motif_anchors(output_aa, input_aa, contig, saved_transform=None):
    if saved_transform is not None:
        R, t = saved_transform
        output_aa.coord = apply_kabsch(
            output_aa.coord.astype(np.float64), R, t
        ).astype(output_aa.coord.dtype)
        return output_aa

    motif_residues = parse_contig_motif_residues(contig)
    if not motif_residues:
        return output_aa

    in_cas = []
    for chain, rid in motif_residues:
        mask = (
            (input_aa.chain_id == chain)
            & (input_aa.res_id == rid)
            & (np.char.strip(input_aa.atom_name) == "CA")
        )
        if np.any(mask):
            in_cas.append(input_aa.coord[mask][0])
        else:
            in_cas.append(None)

    out_cas = None
    out_annots = output_aa.get_annotation_categories()
    if "src_component" in out_annots:
        out_cas = _extract_motif_cas_by_src_component(output_aa, motif_residues)

    if out_cas is None and "is_motif_atom_with_fixed_coord" in out_annots:
        out_cas = _extract_motif_cas_by_fixed_coord(output_aa, len(motif_residues))

    if out_cas is None:
        out_cas = _extract_motif_cas_by_position(output_aa, len(motif_residues))

    paired_in, paired_out = [], []
    for ic, oc in zip(in_cas, out_cas):
        if ic is not None and oc is not None:
            paired_in.append(ic)
            paired_out.append(oc)

    if not paired_in:
        return output_aa

    return _apply_alignment(output_aa, paired_in, paired_out)


def _extract_motif_cas_by_src_component(output_aa, motif_residues):
    out_cas = []
    for chain, rid in motif_residues:
        component = f"{chain}{rid}"
        ca_mask = np.char.strip(output_aa.atom_name) == "CA"
        src_match = output_aa.src_component == component
        mask = ca_mask & src_match
        if np.any(mask):
            out_cas.append(output_aa.coord[mask][0])
        else:
            out_cas.append(None)
    matched = sum(1 for c in out_cas if c is not None)
    return out_cas if matched > 0 else None


def _extract_motif_cas_by_fixed_coord(output_aa, n_motif):
    fixed = output_aa.is_motif_atom_with_fixed_coord.astype(bool)
    ca_mask = np.char.strip(output_aa.atom_name) == "CA"
    motif_ca_mask = fixed & ca_mask
    motif_ca_coords = output_aa.coord[motif_ca_mask]
    if len(motif_ca_coords) == 0:
        return None
    return [
        motif_ca_coords[i] if i < len(motif_ca_coords) else None
        for i in range(n_motif)
    ]


def _extract_motif_cas_by_position(output_aa, n_motif):
    out_cas = []
    seen_res = set()
    for idx in range(len(output_aa)):
        if len(out_cas) >= n_motif:
            break
        atom_name = np.char.strip(output_aa.atom_name[idx])
        res_key = (output_aa.chain_id[idx], int(output_aa.res_id[idx]))
        if atom_name == "CA" and res_key not in seen_res:
            seen_res.add(res_key)
            out_cas.append(output_aa.coord[idx])
    while len(out_cas) < n_motif:
        out_cas.append(None)
    return out_cas


def _apply_alignment(output_aa, in_cas, out_cas):
    in_cas = np.array(in_cas, dtype=np.float64)
    out_cas = np.array(out_cas, dtype=np.float64)

    if len(in_cas) < 3:
        t = in_cas.mean(axis=0) - out_cas.mean(axis=0)
        output_aa.coord = output_aa.coord + t.astype(output_aa.coord.dtype)
    else:
        R, t = kabsch_transform(out_cas, in_cas)
        output_aa.coord = apply_kabsch(
            output_aa.coord.astype(np.float64), R, t
        ).astype(output_aa.coord.dtype)
    return output_aa


# IUPAC one-letter → three-letter map for the 20 standard amino acids.
# Used by apply_sequence to map an MPNN-style sequence string onto
# biotite residue templates (`biotite.structure.info.residue`).
_AA_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def _build_template_residue(target_residue, new_three_letter, residue_template_fn):
    """Build a template-based replacement for `target_residue`.

    Fetches the standard residue template for `new_three_letter`,
    superposes it onto `target_residue`'s N/CA/C via Kabsch, and stamps
    the target's chain_id / res_id onto the result so the rebuilt
    residue slots back into the parent AtomArray cleanly.

    Sidechain rotamer is whatever the biotite template ships (typically
    the most-extended conformer); a downstream rotamer pack is expected
    to refine clashes with neighbors.
    """
    template = residue_template_fn(new_three_letter).copy()

    def backbone_coords(arr):
        names = arr.atom_name
        n_idx = next((i for i, n in enumerate(names) if n == "N"), None)
        ca_idx = next((i for i, n in enumerate(names) if n == "CA"), None)
        c_idx = next((i for i, n in enumerate(names) if n == "C"), None)
        return n_idx, ca_idx, c_idx

    t_n, t_ca, t_c = backbone_coords(target_residue)
    if t_n is None or t_ca is None or t_c is None:
        chain = (
            str(target_residue.chain_id[0])
            if len(target_residue) > 0
            else "?"
        )
        rid = (
            int(target_residue.res_id[0])
            if len(target_residue) > 0
            else -1
        )
        raise ValueError(
            f"apply_sequence: target residue (chain {chain!r} res_id "
            f"{rid}) missing backbone N/CA/C; cannot superpose template"
        )

    tpl_n, tpl_ca, tpl_c = backbone_coords(template)
    if tpl_n is None or tpl_ca is None or tpl_c is None:
        raise RuntimeError(
            f"biotite template {new_three_letter!r} lacks backbone N/CA/C"
        )

    target_anchors = np.asarray(
        [
            target_residue.coord[t_n],
            target_residue.coord[t_ca],
            target_residue.coord[t_c],
        ],
        dtype=np.float64,
    )
    mobile_anchors = np.asarray(
        [
            template.coord[tpl_n],
            template.coord[tpl_ca],
            template.coord[tpl_c],
        ],
        dtype=np.float64,
    )

    R, t = kabsch_transform(mobile_anchors, target_anchors)
    template.coord = apply_kabsch(
        template.coord.astype(np.float64), R, t
    ).astype(template.coord.dtype)

    # Stamp identity from the target residue (chain_id, res_id). res_name
    # is set by the template; hetero stays False for standard AAs.
    template.chain_id[:] = target_residue.chain_id[0]
    template.res_id[:] = int(target_residue.res_id[0])
    return template


def _entity_info_from_assembly(assembly_bytes, atom_array):
    """Walk the assembly's entities → list of {molecule_type, chain_id, atom_count}.

    `entity_id` is the molex-allocated EntityId the host addresses entities by;
    `molecule_type` is lower-cased ("protein", "ligand", "cofactor", ...) to
    match the comparisons the callers make; `chain_id` is the entity's chain
    id ("" for a non-polymer entity with no chain). `atom_count` lets callers
    slice the concatenated atom_array per entity.
    """
    import molex

    try:
        asm = molex.Assembly.from_assembly_bytes(assembly_bytes)
        entities = []
        for e in asm.entities():
            entities.append({
                "entity_id": e.id,
                "molecule_type": e.molecule_type.lower(),
                "chain_id": e.chain_id or "",
                "atom_count": e.atom_count,
            })
        return entities
    except Exception as e:
        logger.warning("Failed to read entity info from assembly: %s", e)
        return []


def _extract_ligand_names(atom_array, assembly_bytes):
    entity_info = _entity_info_from_assembly(assembly_bytes, atom_array)
    if not entity_info:
        return None

    ligand_res_names = set()
    atom_offset = 0
    for info in entity_info:
        atom_count = info.get("atom_count", 0)
        mol_type = info.get("molecule_type", "")
        if mol_type in ("ligand", "cofactor"):
            end = min(atom_offset + atom_count, len(atom_array))
            if atom_offset < end:
                names = set(atom_array.res_name[atom_offset:end])
                ligand_res_names.update(names)
        atom_offset += atom_count

    return ",".join(sorted(ligand_res_names)) if ligand_res_names else None


def _build_context_contig(atom_array, assembly_bytes, design_length_range):
    entity_info = _entity_info_from_assembly(assembly_bytes, atom_array)
    if not entity_info:
        return None

    chain_ranges: dict[str, tuple[int, int]] = {}
    atom_offset = 0
    for info in entity_info:
        atom_count = info.get("atom_count", 0)
        mol_type = info.get("molecule_type", "")
        chain_id = info.get("chain_id", "")
        if mol_type == "protein" and chain_id:
            end = min(atom_offset + atom_count, len(atom_array))
            if atom_offset < end:
                res_ids = atom_array.res_id[atom_offset:end]
                rmin = int(res_ids.min())
                rmax = int(res_ids.max())
                if chain_id not in chain_ranges:
                    chain_ranges[chain_id] = (rmin, rmax)
                else:
                    prev = chain_ranges[chain_id]
                    chain_ranges[chain_id] = (
                        min(prev[0], rmin), max(prev[1], rmax),
                    )
        atom_offset += atom_count

    if not chain_ranges:
        return None

    parts = [f"{cid}{rmin}-{rmax}" for cid, (rmin, rmax) in chain_ranges.items()]
    parts.append("/0")
    parts.append(str(design_length_range))
    return ",".join(parts)


def _splice_coords(target, source):
    """Write `source` coordinates and B-factors onto `target` by atom identity.

    RF3's pipeline returns a DIFFERENT atom set than it was handed (hydrogens
    stripped, terminal oxygens removed, ligands atomized, short polymers
    dropped), so its output cannot be committed over the session's entity lanes
    directly. Matching on ``(chain_id, res_id, atom_name)`` writes the predicted
    coordinates back onto the original array, preserving atom count, ordering,
    and the ``entity_id`` annotation the host matches entities by. Atoms RF3
    dropped keep their prior coordinates.

    Returns ``(spliced_atom_array, matched_atom_count)``.
    """
    out = target.copy()
    index = {}
    for j in range(len(source)):
        key = (
            str(source.chain_id[j]),
            int(source.res_id[j]),
            str(source.atom_name[j]),
        )
        index[key] = j

    source_b = getattr(source, "b_factor", None)
    out_b = getattr(out, "b_factor", None)
    matched = 0
    for i in range(len(out)):
        key = (str(out.chain_id[i]), int(out.res_id[i]), str(out.atom_name[i]))
        j = index.get(key)
        if j is None:
            continue
        out.coord[i] = source.coord[j]
        if source_b is not None and out_b is not None:
            out_b[i] = source_b[j]
        matched += 1
    return out, matched


def _entity_index_for_id(entity_info, entity_id):
    """Position of `entity_id` within the assembly's entity list, or None.

    The host addresses entities by their molex-allocated `EntityId`, which
    coincides with the list position only on a freshly loaded assembly and
    diverges after any entity is added or removed. Every caller that needs to
    slice `atom_array` per entity must go through this.
    """
    if entity_id is None:
        return None
    for idx, info in enumerate(entity_info):
        if info.get("entity_id") == entity_id:
            return idx
    return None


def _build_focused_target_contig(
    atom_array, assembly_bytes, focused_entity_id, design_length_range
):
    """Build a binder-design spec that targets one focused entity.

    `focused_entity_id` is the host's molex `EntityId`; it is resolved to a
    position in the assembly's entity list via `_entity_index_for_id`.

    Returns ``(contig, ligand)`` or ``None``:

    - **Protein focus** → ``(contig, None)`` where ``contig`` keeps the
      focused chain's residue range fixed as a motif and appends a
      ``/0,{design_length_range}`` designed segment (the binder).
    - **Ligand / cofactor focus** → ``(None, ligand_res_names)``; the
      caller designs a de-novo binder of ``design_length_range`` against
      that ligand alone.
    - Unresolvable focus (out of range, or an entity type we don't target)
      → ``None``; the caller falls back to whole-assembly context.
    """
    entity_info = _entity_info_from_assembly(assembly_bytes, atom_array)
    focused_idx = _entity_index_for_id(entity_info, focused_entity_id)
    if focused_idx is None:
        return None

    atom_start = sum(
        info.get("atom_count", 0) for info in entity_info[:focused_idx]
    )
    atom_end = atom_start + entity_info[focused_idx].get("atom_count", 0)
    atom_end = min(atom_end, len(atom_array))
    if atom_start >= atom_end:
        return None

    focused = entity_info[focused_idx]
    mol_type = focused.get("molecule_type", "")
    chain_id = focused.get("chain_id", "")

    if mol_type == "protein" and chain_id:
        res_ids = atom_array.res_id[atom_start:atom_end]
        rmin = int(res_ids.min())
        rmax = int(res_ids.max())
        contig = f"{chain_id}{rmin}-{rmax},/0,{design_length_range}"
        return contig, None

    if mol_type in ("ligand", "cofactor"):
        names = sorted(set(atom_array.res_name[atom_start:atom_end]))
        ligand = ",".join(names) if names else None
        return None, ligand

    return None


class Plugin(PluginInterface):
    """Foundry: RF3 + RFD3 + LigandMPNN behind plugin.proto."""

    def __init__(self, config: dict[str, Any]) -> None:
        # Per protocol §"Plugin self-configuration": weights are plugin
        # assets under <plugin_dir>/assets/weights/, resolved from the
        # plugin_dir the host hands us in config. No cache_dir on the wire.
        self.cache_dir = weights_dir_from_config(config)
        self.model_type = config.get("model_type", "ligand_mpnn")

        # Eager checkpoint discovery (fast glob only, no model loading).
        # Re-run lazily at op time + after download since the glob ran
        # before any download_weights op could have placed the files.
        self._rf3_checkpoint = None
        self._rfd3_checkpoint = None
        self._mpnn_checkpoint = None
        self._rediscover_checkpoints()

        # Lazy sub-engine handles.
        self._mpnn_engine = None

        from foldit_plugin_sdk.device_utils import log_device_info

        log_device_info(logger)

        available = []
        if self._rf3_checkpoint:
            available.append("predict (RF3)")
        if self._rfd3_checkpoint:
            available.append("design (RFD3)")
        if self._mpnn_checkpoint:
            available.append("sequence_design (MPNN)")
        logger.info(
            "Foundry engine ready — capabilities: %s",
            ", ".join(available) or "none",
        )

        # Stream state.
        self._streams: dict[int, PollOutcome] = {}
        self._cancel: dict[int, bool] = {}
        self._gpu_lock = threading.Lock()
        self._assembly: bytes | None = None
        # Guards against a second download_weights op racing the first.
        self._download_lock = threading.Lock()
        self._download_active = False

    def _rediscover_checkpoints(self) -> None:
        """(Re)glob the three sub-engine checkpoints from ``cache_dir``.

        Cheap (glob only). Run at __init__ and again after a download (or
        lazily at op time) so checkpoints placed by the download_weights op
        after construction are picked up without a respawn."""
        self._rf3_checkpoint = self._find_optional("rf3_*.ckpt", "RoseTTAFold3")
        self._rfd3_checkpoint = self._find_optional(
            "rfd3*.ckpt",
            "RFdiffusion3",
            search_dirs=["rc_foundry", "rfd3"],
        )
        self._mpnn_checkpoint = self._find_optional(
            "ligandmpnn_*.pt"
            if self.model_type == "ligand_mpnn"
            else "proteinmpnn_*.pt",
            f"LigandMPNN ({self.model_type})",
        )

    def _find_optional(self, pattern, name, search_dirs=None):
        try:
            return find_checkpoint(
                self.cache_dir,
                pattern=pattern,
                model_name=name,
                search_dirs=search_dirs,
            )
        except FileNotFoundError:
            logger.info("%s checkpoint not found — capability disabled", name)
            return None

    # Lifecycle

    def init(self, assembly_bytes: bytes) -> int:
        self._assembly = assembly_bytes
        return 1

    def update_assembly(
        self,
        session: int,
        payload_kind: int,
        bytes: bytes,
        from_gen: int,
        to_gen: int,
    ) -> None:
        del session, from_gen, to_gen
        if payload_kind == PluginInterface.PAYLOAD_KIND_DELTA:
            # Delta bytes patch against the current assembly. The host streams
            # these for live edits (e.g. a rosetta wiggle). They are NOT
            # assembly bytes, so storing them raw corrupts `_assembly` and the
            # next molex decode dies on the magic number. Decode + apply via
            # molex and re-emit a full assembly snapshot.
            if self._assembly is None:
                logger.warning(
                    "update_assembly: delta bytes before any full assembly; ignoring"
                )
                return
            try:
                import molex

                asm = molex.Assembly.from_assembly_bytes(self._assembly)
                asm.apply_delta(bytes)
                self._assembly = asm.to_assembly_bytes()
            except Exception as e:
                # Gen mismatch / malformed delta: keep the last good full
                # assembly rather than crashing the next op (slightly stale).
                logger.warning(
                    "update_assembly: delta apply failed (%s); keeping prior "
                    "assembly", e,
                )
        else:
            # PAYLOAD_KIND_FULL: fresh full assembly snapshot.
            self._assembly = bytes

    def drop(self, session: int) -> None:
        for rid in list(self._cancel.keys()):
            self._cancel[rid] = True
        self._streams.clear()
        self._cancel.clear()
        self._assembly = None

    # Registration

    def register(self) -> plugin_pb2.PluginRegistration:
        # Ops are declared unconditionally — independent of whether their
        # checkpoints are present at construction. Weights download on
        # first use (download_weights op); the op handlers re-discover the
        # checkpoint and surface a clear error if it is still absent.
        return plugin_pb2.PluginRegistration(
            id="foundry",
            version="0.1.0",
            operations=[
                self._predict_op_spec(),
                self._design_op_spec(),
                self._mpnn_design_op_spec(),
                self._apply_sequence_op_spec(),
                weights.download_weights_op_spec(),
            ],
            queries=[
                self._sequence_design_query_spec(),
                weights.weights_status_query_spec(),
            ],
        )

    @staticmethod
    def _predict_op_spec() -> plugin_pb2.PluginOp:
        return plugin_pb2.PluginOp(
            id="rf3_predict",
            display_name="Predict (RF3)",
            description="Structure prediction with RoseTTAFold3.",
            kind=plugin_pb2.OP_KIND_STREAM,
            # Never creates entities: the prediction is committed as new
            # coordinates on the existing lanes. `compatible_focus_types` with
            # no `requires_focus` makes focus OPTIONAL -- the host locks just
            # the focused entity (so only it commits) and falls back to a
            # global lock (whole-structure refold) when nothing is focused.
            creates_entities=False,
            compatible_focus_types=[
                plugin_pb2.ENTITY_TYPE_PROTEIN,
                plugin_pb2.ENTITY_TYPE_NUCLEIC_ACID,
            ],
            params=[
                plugin_pb2.ParamSpec(
                    name="num_recycles",
                    display_name="Num recycles",
                    type=plugin_pb2.PARAM_TYPE_INT,
                    default=make_param_value(4),
                    constraints=plugin_pb2.ParamConstraints(
                        int_range=plugin_pb2.IntRange(min=1, max=20),
                    ),
                ),
            ],
        )

    @staticmethod
    def _design_op_spec() -> plugin_pb2.PluginOp:
        return plugin_pb2.PluginOp(
            # Op-ids must be globally unique: dispatch routes op-id → plugin
            # through a flat registry (last-writer-wins). A bare "design"
            # collides with the dummy plugin's "design", so this is namespaced
            # to the model, as `rf3_predict` is. The `sequence_design` QUERY id
            # still collides with dummy's, but queries carry no buttons.
            id="rfd3_design",
            display_name="Design (RFD3)",
            description="De novo / motif-scaffold protein design.",
            kind=plugin_pb2.OP_KIND_STREAM,
            creates_entities=True,
            # Gate to a focused protein or ligand: the focused entity is used
            # as the fixed binder target (see `_build_focused_target_contig`).
            # Empty would mean global-scoped; declaring these scopes the op's
            # lock to the focused entity and disables the button when an
            # incompatible entity (e.g. nucleic acid) is focused.
            compatible_focus_types=[
                plugin_pb2.ENTITY_TYPE_PROTEIN,
                plugin_pb2.ENTITY_TYPE_SMALL_MOLECULE,
            ],
            # Binder design genuinely needs the focused entity as the fixed
            # target; refuse (button disabled) when nothing compatible is
            # focused, rather than falling back to a global run.
            requires_focus=True,
            params=[
                plugin_pb2.ParamSpec(
                    name="length",
                    display_name="Length",
                    description="\"min-max\" residue range, e.g. 70-100.",
                    type=plugin_pb2.PARAM_TYPE_STRING,
                    default=make_param_value("70-100"),
                ),
                plugin_pb2.ParamSpec(
                    name="contig",
                    display_name="Contig",
                    description="Optional contig for motif scaffolding.",
                    type=plugin_pb2.PARAM_TYPE_STRING,
                    default=make_param_value(""),
                ),
                plugin_pb2.ParamSpec(
                    name="num_designs",
                    display_name="Num designs",
                    type=plugin_pb2.PARAM_TYPE_INT,
                    default=make_param_value(1),
                    constraints=plugin_pb2.ParamConstraints(
                        int_range=plugin_pb2.IntRange(min=1, max=100),
                    ),
                ),
                plugin_pb2.ParamSpec(
                    name="num_steps",
                    display_name="Diffusion steps",
                    type=plugin_pb2.PARAM_TYPE_INT,
                    default=make_param_value(50),
                    constraints=plugin_pb2.ParamConstraints(
                        int_range=plugin_pb2.IntRange(min=10, max=1000),
                    ),
                ),
                plugin_pb2.ParamSpec(
                    name="step_scale",
                    display_name="Step scale",
                    type=plugin_pb2.PARAM_TYPE_FLOAT,
                    default=make_param_value(1.5),
                    constraints=plugin_pb2.ParamConstraints(
                        float_range=plugin_pb2.FloatRange(min=0.1, max=10.0),
                    ),
                ),
                plugin_pb2.ParamSpec(
                    name="save_trajectories",
                    display_name="Save trajectories",
                    description=(
                        "If true, encode all denoised frames in the final poll outcome. "
                        "Otherwise final assembly only — intermediates already arrived "
                        "as Pending snapshots."
                    ),
                    type=plugin_pb2.PARAM_TYPE_BOOL,
                    default=make_param_value(False),
                ),
            ],
        )

    @staticmethod
    def _mpnn_design_op_spec() -> plugin_pb2.PluginOp:
        """One-shot LigandMPNN redesign of the focused chain.

        The `sequence_design` QUERY returns N scored candidates for a panel to
        offer; this op is the button-shaped counterpart -- it runs the same
        compute and commits the best-scoring sequence via `apply_sequence`.
        """
        return plugin_pb2.PluginOp(
            id="mpnn_design",
            display_name="Design (LigandMPNN)",
            description=(
                "Redesign the focused protein chain's sequence with "
                "LigandMPNN and apply the best-scoring candidate. Selected "
                "residues are held fixed."
            ),
            kind=plugin_pb2.OP_KIND_INVOKE,
            creates_entities=False,
            compatible_focus_types=[plugin_pb2.ENTITY_TYPE_PROTEIN],
            requires_focus=True,
            params=[
                plugin_pb2.ParamSpec(
                    name="num_sequences",
                    display_name="Candidates",
                    description="How many sequences to sample; the best is applied.",
                    type=plugin_pb2.PARAM_TYPE_INT,
                    default=make_param_value(8),
                    constraints=plugin_pb2.ParamConstraints(
                        int_range=plugin_pb2.IntRange(min=1, max=32),
                    ),
                ),
                plugin_pb2.ParamSpec(
                    name="temperature",
                    display_name="Temperature",
                    description="Sampling temperature; lower is more conservative.",
                    type=plugin_pb2.PARAM_TYPE_FLOAT,
                    default=make_param_value(0.1),
                    constraints=plugin_pb2.ParamConstraints(
                        float_range=plugin_pb2.FloatRange(min=0.01, max=1.0),
                    ),
                ),
            ],
        )

    @staticmethod
    def _sequence_design_query_spec() -> plugin_pb2.PluginQuery:
        return plugin_pb2.PluginQuery(
            id="sequence_design",
            display_name="Sequence Design (MPNN)",
            description=(
                "Inverse folding via LigandMPNN. Returns candidate sequences "
                "for the focused protein chain; selected residues are kept "
                "fixed. User picks one; a separate apply_sequence op commits."
            ),
            params=[
                plugin_pb2.ParamSpec(
                    name="num_sequences",
                    display_name="Num sequences",
                    type=plugin_pb2.PARAM_TYPE_INT,
                    default=make_param_value(10),
                    constraints=plugin_pb2.ParamConstraints(
                        int_range=plugin_pb2.IntRange(min=1, max=100),
                    ),
                ),
                plugin_pb2.ParamSpec(
                    name="temperature",
                    display_name="Temperature",
                    type=plugin_pb2.PARAM_TYPE_FLOAT,
                    default=make_param_value(0.1),
                    constraints=plugin_pb2.ParamConstraints(
                        float_range=plugin_pb2.FloatRange(min=0.0, max=1.0),
                    ),
                ),
            ],
        )

    @staticmethod
    def _apply_sequence_op_spec() -> plugin_pb2.PluginOp:
        return plugin_pb2.PluginOp(
            id="apply_sequence",
            display_name="Apply Sequence",
            description=(
                "Commit a sequence onto the focused protein chain. Each "
                "residue is rebuilt from a biotite template superposed onto "
                "the existing N/CA/C backbone (sidechain rotamers are "
                "template defaults; a downstream rosetta repack is expected "
                "to refine clashes)."
            ),
            kind=plugin_pb2.OP_KIND_INVOKE,
            creates_entities=False,
            compatible_focus_types=[plugin_pb2.ENTITY_TYPE_PROTEIN],
            requires_focus=True,
            params=[
                plugin_pb2.ParamSpec(
                    name="sequence",
                    display_name="Sequence",
                    description=(
                        "One-letter amino acid string (length must equal the "
                        "focused chain's residue count). Standard 20 AAs only."
                    ),
                    type=plugin_pb2.PARAM_TYPE_STRING,
                    default=make_param_value(""),
                    constraints=plugin_pb2.ParamConstraints(
                        string_pattern=plugin_pb2.StringPattern(
                            pattern="^[ACDEFGHIKLMNPQRSTVWY]+$",
                        ),
                    ),
                ),
            ],
        )

    # Dispatch: streaming

    def start_stream(
        self,
        session: int,
        op: str,
        context: DispatchContext,
        params: dict[str, Any],
        request_id: int,
    ) -> None:
        if op == weights.DOWNLOAD_WEIGHTS_OP:
            self._start_download(request_id)
            return

        if op == "rf3_predict":
            if not self._rf3_checkpoint:
                self._rediscover_checkpoints()
            if not self._rf3_checkpoint:
                raise RuntimeError(
                    "RF3 checkpoint not available — run download_weights"
                )
            rid = request_id
            self._streams[rid] = PollOutcome.pending(progress=0.0, stage="queued")
            self._cancel[rid] = False
            threading.Thread(
                target=self._run_predict,
                args=(rid, context, params),
                name=f"foundry-predict-{rid}",
                daemon=True,
            ).start()
            return

        if op == "rfd3_design":
            if not self._rfd3_checkpoint:
                self._rediscover_checkpoints()
            if not self._rfd3_checkpoint:
                raise RuntimeError(
                    "RFD3 checkpoint not available — run download_weights"
                )
            rid = request_id
            self._streams[rid] = PollOutcome.pending(progress=0.0, stage="queued")
            self._cancel[rid] = False
            threading.Thread(
                target=self._run_design,
                args=(rid, context, params),
                name=f"foundry-design-{rid}",
                daemon=True,
            ).start()
            return

        raise ValueError(f"Unknown stream op: {op!r}")

    def poll_stream(self, request_id: int) -> PollOutcome:
        return self._streams.get(
            request_id,
            PollOutcome.error("NOT_FOUND", f"No stream {request_id}", {}),
        )

    def cancel_stream(self, request_id: int) -> None:
        if request_id in self._cancel:
            self._cancel[request_id] = True

    # Dispatch: query

    def query(
        self,
        session: int,
        query: str,
        context: DispatchContext,
        params: dict[str, Any],
    ) -> bytes:
        if query == weights.WEIGHTS_STATUS_QUERY:
            return self._weights_status()
        if query == "sequence_design":
            return self._run_sequence_design(context, params)
        raise ValueError(f"Unknown query: {query!r}")

    # Dispatch: invoke

    def invoke(
        self,
        session: int,
        op: str,
        context: DispatchContext,
        params: dict[str, Any],
    ) -> bytes:
        if op == "apply_sequence":
            return self._run_apply_sequence(context, params)
        if op == "mpnn_design":
            return self._run_mpnn_design(context, params)
        raise ValueError(f"Unknown invoke op: {op!r}")

    # Weight download (download_weights op)

    def _weights_status(self) -> bytes:
        """Report rf3 / rfd3 / mpnn checkpoint presence as the status JSON."""
        self._rediscover_checkpoints()
        present: list[str] = []
        missing: list[str] = []
        for label, ckpt in (
            ("rf3", self._rf3_checkpoint),
            ("rfd3", self._rfd3_checkpoint),
            ("mpnn", self._mpnn_checkpoint),
        ):
            (present if ckpt else missing).append(label)
        return weights.status_payload(present, missing)

    def _start_download(self, request_id: int) -> None:
        """Kick off a background checkpoint install under the host-assigned
        stream id.

        A second download_weights op while one runs is refused rather than
        racing the same files."""
        with self._download_lock:
            if self._download_active:
                raise RuntimeError("Weight download already in progress")
            self._download_active = True
        rid = request_id
        self._streams[rid] = PollOutcome.pending(progress=0.0, stage="starting")
        self._cancel[rid] = False
        threading.Thread(
            target=self._run_download,
            args=(rid,),
            name=f"foundry-download-{rid}",
            daemon=True,
        ).start()

    def _run_download(self, rid: int) -> None:
        # foundry's checkpoints are plain URL files, so fetch them through the
        # SDK's shared byte-progress downloader (the same path simplefold uses)
        # rather than rc-foundry's install_model, whose only progress sink is a
        # rich terminal bar. That yields real per-file progress: fraction is the
        # blended overall byte fraction, stage carries the current file and MB.
        # Files land in <cache_dir>/rc_foundry/ under each checkpoint's own
        # filename, where _rediscover_checkpoints globs them.
        try:
            if self._assembly is None:
                raise RuntimeError("download_weights: init must run first")

            from foundry.inference_engines.checkpoint_registry import (
                REGISTERED_CHECKPOINTS,
            )

            specs = [
                weights.WeightSpec(
                    url=REGISTERED_CHECKPOINTS[model].url,
                    subdir="rc_foundry",
                    name=REGISTERED_CHECKPOINTS[model].filename,
                )
                for model in FOUNDRY_MODELS
            ]

            def on_progress(frac: float, stage: str) -> None:
                if not self._cancel.get(rid):
                    self._streams[rid] = PollOutcome.pending(progress=frac, stage=stage)

            weights.download_specs(
                self.cache_dir,
                specs,
                on_progress=on_progress,
                should_cancel=lambda: self._cancel.get(rid, False),
            )
            # Pick up the freshly placed checkpoints for subsequent ops.
            self._rediscover_checkpoints()
            if self._cancel.get(rid):
                self._streams[rid] = PollOutcome.error(
                    "CANCELLED", "Cancelled by user", {}
                )
                return
            # No entity change: return the unchanged working assembly as
            # the stream terminal (the protocol requires assembly bytes).
            self._streams[rid] = PollOutcome.final_(self._assembly)
            logger.info("Foundry weights download rid=%d complete", rid)
        except weights.WeightDownloadCancelled:
            self._streams[rid] = PollOutcome.error(
                "CANCELLED", "Cancelled by user", {}
            )
        except Exception as e:
            traceback.print_exc()
            self._streams[rid] = PollOutcome.error("INTERNAL", str(e), {})
        finally:
            with self._download_lock:
                self._download_active = False

    def _set_pending(
        self,
        rid: int,
        latest_assembly: bytes | None,
        progress: float | None,
        stage: str | None,
    ) -> None:
        # Coalesce — overwrite, no queue.
        self._streams[rid] = PollOutcome.pending(
            latest_assembly=latest_assembly, progress=progress, stage=stage
        )

    # predict (RF3)

    def _run_predict(
        self, rid: int, context: DispatchContext, params: dict[str, Any]
    ) -> None:
        try:
            num_recycles = int(params.get("num_recycles", 4))
            # RF3 has no native subset prediction, so the whole assembly is
            # always folded (it needs the full context anyway). Scoping to the
            # focused entity is the host's job: an entity-scoped lock means
            # only that entity's lane accepts the committed coordinates.
            if context.focused_entity_id is not None:
                logger.info(
                    "Foundry rf3_predict rid=%d: focused entity %s; only its "
                    "lane will accept the result",
                    rid, context.focused_entity_id,
                )
            if not self._assembly:
                raise RuntimeError("No assembly — call init or update_assembly first")

            with self._gpu_lock:
                if self._cancel.get(rid):
                    return
                _, coords_bytes, confidence = self._predict_impl(
                    rid, self._assembly, num_recycles
                )

            if self._cancel.get(rid):
                self._streams[rid] = PollOutcome.error(
                    "CANCELLED", "Cancelled by user", {}
                )
                return

            # Confidence is embedded in assembly bytes (RF3 sets B-factors).
            self._streams[rid] = PollOutcome.final_(coords_bytes)
            logger.info(
                "Foundry predict rid=%d done, confidence=%.3f", rid, confidence
            )
        except Exception as e:
            traceback.print_exc()
            self._streams[rid] = PollOutcome.error(
                "INTERNAL", str(e), {}
            )

    def _predict_impl(
        self, rid: int, assembly_bytes: bytes, num_recycles: int
    ) -> tuple[None, bytes, float]:
        num_recycles = max(1, min(num_recycles, 20))

        self._set_pending(rid, None, 0.0, "loading-rf3")
        logger.info(
            "Loading RF3 inference engine (checkpoint=%s)...", self._rf3_checkpoint
        )

        from rf3.inference_engines.rf3 import RF3InferenceEngine

        inference_engine = RF3InferenceEngine(
            ckpt_path=self._rf3_checkpoint,
            n_recycles=num_recycles,
            diffusion_batch_size=5,
            num_steps=50,
            early_stopping_plddt_threshold=0.5,
            verbose=False,
        )

        from . import molex_io

        atom_array = molex_io.assembly_bytes_to_atom_array(assembly_bytes)
        logger.info(
            "Predicting from AtomArray (%d atoms, recycles=%d)...",
            len(atom_array), num_recycles,
        )

        rf3_step_callback = self._build_rf3_step_callback(rid, atom_array)

        try:
            results = inference_engine.run(
                inputs=atom_array, out_dir=None, step_callback=rf3_step_callback
            )
        except (AssertionError, RuntimeError) as e:
            # Atomworks transforms may fail on non-standard residues. Retry
            # with protein/nucleic-acid only.
            logger.warning(
                "Full assembly failed (%s), retrying with standard residues...", e,
            )
            from biotite.structure import filter_amino_acids, filter_nucleotides

            protein_mask = filter_amino_acids(atom_array) | filter_nucleotides(
                atom_array
            )
            atom_array_filtered = atom_array[protein_mask]
            rf3_step_callback = self._build_rf3_step_callback(
                rid, atom_array_filtered
            )
            results = inference_engine.run(
                inputs=atom_array_filtered,
                out_dir=None,
                step_callback=rf3_step_callback,
            )

        if not results:
            raise RuntimeError("RoseTTAFold3 returned no predictions")

        example_id = next(iter(results))
        rf3_outputs = results[example_id]
        if not rf3_outputs:
            raise RuntimeError(f"No RF3Output objects for example {example_id}")

        rf3_output = rf3_outputs[0]
        out_atom_array = rf3_output.atom_array
        confidence = rf3_output.summary_confidences.get("ptm", 0.0)
        if confidence == 0.0:
            confidence = max(
                0.0,
                rf3_output.summary_confidences.get("ranking_score", 0.0) / 100.0,
            )

        # RF3 returns a processed atom set, not the one it was given; splice
        # its coordinates back onto the input so the commit preserves atom
        # count, order and entity ids.
        spliced, matched = _splice_coords(atom_array, out_atom_array)
        if matched == 0:
            raise RuntimeError(
                "RF3 returned no atoms matching the input structure; refusing "
                "to commit a prediction that cannot be aligned"
            )
        logger.info(
            "RF3 spliced %d/%d predicted atoms onto the input structure",
            matched, len(atom_array),
        )
        result_bytes = molex_io.atom_array_to_assembly_bytes(spliced)
        return None, result_bytes, confidence

    def _build_rf3_step_callback(self, rid: int, atom_array):
        """Stream RF3 diffusion intermediates as backbone-only frames."""
        from biotite.structure import AtomArray, filter_amino_acids

        from . import molex_io

        aa_mask = filter_amino_acids(atom_array)
        residues = []
        residue_index = {}
        for i in range(len(atom_array)):
            if not aa_mask[i]:
                continue
            key = (str(atom_array.chain_id[i]), int(atom_array.res_id[i]))
            slot = residue_index.get(key)
            if slot is None:
                slot = (
                    str(atom_array.chain_id[i]),
                    int(atom_array.res_id[i]),
                    str(atom_array.res_name[i]),
                    {},
                )
                residue_index[key] = slot
                residues.append(slot)
            slot[3][str(atom_array.atom_name[i])] = i

        bb_indices = []
        bb_chain_ids = []
        bb_res_ids = []
        bb_res_names = []
        for chain_id, res_id, res_name, atoms_by_name in residues:
            if not all(name in atoms_by_name for name in ("N", "CA", "C", "O")):
                continue
            for name in ("N", "CA", "C", "O"):
                bb_indices.append(atoms_by_name[name])
            bb_chain_ids.extend([chain_id] * 4)
            bb_res_ids.extend([res_id] * 4)
            bb_res_names.extend([res_name] * 4)

        if not bb_indices:
            return None

        bb_indices_arr = np.asarray(bb_indices, dtype=np.int64)
        bb_chain_ids_arr = np.asarray(bb_chain_ids)
        bb_res_ids_arr = np.asarray(bb_res_ids, dtype=np.int32)
        bb_res_names_arr = np.asarray(bb_res_names)
        n_bb = len(bb_indices_arr)
        atom_names_arr = np.tile(np.array(["N", "CA", "C", "O"]), n_bb // 4)
        elements_arr = np.tile(np.array(["N", "C", "C", "O"]), n_bb // 4)

        cb_state = {"prev_coords": None}
        outer_self = self

        def callback(info):
            try:
                if outer_self._cancel.get(rid):
                    return
                step, total = info.step, info.total_steps
                all_coords = (
                    info.coords.detach().cpu()[0].numpy().astype(np.float64)
                )
                if all_coords.shape[0] != len(atom_array):
                    return
                coords = all_coords[bb_indices_arr]
                if (
                    cb_state["prev_coords"] is not None
                    and coords.shape == cb_state["prev_coords"].shape
                ):
                    coords = kabsch_align(coords, cb_state["prev_coords"])
                cb_state["prev_coords"] = coords.copy()

                aa = AtomArray(n_bb)
                aa.coord = np.ascontiguousarray(coords, dtype=np.float32)
                aa.atom_name = atom_names_arr
                aa.element = elements_arr
                aa.res_id = bb_res_ids_arr
                aa.res_name = bb_res_names_arr
                aa.chain_id = bb_chain_ids_arr

                frame_bytes = molex_io.atom_array_to_assembly_bytes(aa)
                outer_self._set_pending(
                    rid, frame_bytes, step / total, f"rf3-step-{step}/{total}"
                )
            except Exception as e:
                logger.warning("RF3 step callback error: %s\n%s", e, traceback.format_exc())

        return callback

    # design (RFD3)

    def _run_design(
        self, rid: int, context: DispatchContext, params: dict[str, Any]
    ) -> None:
        try:
            # A plain button click arrives with empty params (the GUI does
            # not apply ParamSpec defaults), so fall back to each spec default
            # here. `length` is the one that bites: empty → the op refuses.
            length = params.get("length", "") or "70-100"
            contig = params.get("contig", "")
            num_designs = int(params.get("num_designs", 1))
            num_steps = int(params.get("num_steps", 50))
            step_scale = float(params.get("step_scale", 1.5))
            save_trajectories = bool(params.get("save_trajectories", False))

            if not self._assembly:
                raise RuntimeError("No assembly — call init or update_assembly first")

            with self._gpu_lock:
                if self._cancel.get(rid):
                    return
                results = self._design_impl(
                    rid=rid,
                    length=length,
                    assembly_bytes=self._assembly,
                    contig=contig if contig else None,
                    num_designs=num_designs,
                    num_steps=num_steps,
                    step_scale=step_scale,
                    save_trajectories=save_trajectories,
                    context=context,
                )

            if self._cancel.get(rid):
                self._streams[rid] = PollOutcome.error(
                    "CANCELLED", "Cancelled by user", {}
                )
                return

            if not results:
                raise RuntimeError("No designs generated")

            from . import molex_io

            # First design's assembly is the canonical final assembly; only
            # it is surfaced in the final poll outcome.
            atom_array, confidence, _trajectory_stack = results[0]
            assembly_bytes = molex_io.atom_array_to_assembly_bytes(atom_array)

            # Confidence is embedded in assembly via b_factor.
            self._streams[rid] = PollOutcome.final_(assembly_bytes)
            logger.info(
                "Foundry design rid=%d done, %d designs, lead confidence=%.3f",
                rid, len(results), confidence,
            )
        except Exception as e:
            traceback.print_exc()
            self._streams[rid] = PollOutcome.error(
                "INTERNAL", str(e), {}
            )

    def _design_impl(
        self,
        rid: int,
        length: str,
        assembly_bytes: bytes,
        contig: str | None,
        num_designs: int,
        num_steps: int,
        step_scale: float,
        save_trajectories: bool,
        context: DispatchContext | None = None,
    ):
        if not length and not contig:
            raise ValueError("Must specify either 'length' or 'contig'")
        if num_designs < 1 or num_designs > 100:
            raise ValueError("num_designs must be between 1 and 100")
        if num_steps < 10 or num_steps > 1000:
            raise ValueError("num_steps must be between 10 and 1000")

        self._set_pending(rid, None, 0.0, "loading-rfd3")

        from rfd3.engine import RFD3InferenceEngine, RFD3InferenceConfig

        config = RFD3InferenceConfig(
            ckpt_path=self._rfd3_checkpoint,
            diffusion_batch_size=1,
            specification={},
            inference_sampler={
                "num_timesteps": num_steps,
                "step_scale": step_scale,
            },
            skip_existing=False,
            dump_trajectories=save_trajectories,
            dump_prediction_metadata_json=False,
            output_full_json=False,
            low_memory_mode=False,
        )
        engine = RFD3InferenceEngine(**config.__dict__)

        from rfd3.utils.inference import ensure_inference_sampler_matches_design_spec

        from . import molex_io

        atom_array = molex_io.assembly_bytes_to_atom_array(assembly_bytes)

        # When a protein or ligand entity is focused, use it as the fixed
        # binder target: derive the contig (protein → keep its residues as a
        # motif, design `length` new residues against it) or the ligand spec
        # (ligand → design a binder against that ligand alone) from the
        # focused entity rather than the whole assembly.
        target_ligand: str | None = None
        focused_id = (
            context.focused_entity_id
            if context is not None and context.focused_entity_id is not None
            else None
        )
        # Button / default path (no explicit contig): the focused entity IS
        # the binder target. Require a focused protein or ligand — never
        # silently design against the whole assembly. An explicitly supplied
        # `contig` (manual motif design) bypasses this gate and is honored.
        if not contig:
            if focused_id is None:
                raise ValueError(
                    "RFdiffusion3 needs a focused protein or ligand entity to "
                    "use as the binder target — focus one and try again."
                )
            focus_target = _build_focused_target_contig(
                atom_array, assembly_bytes, focused_id, length
            )
            if focus_target is None:
                raise ValueError(
                    "RFdiffusion3: the focused entity is not a protein or "
                    "ligand; focus a protein or ligand to use as the target."
                )
            focus_contig, target_ligand = focus_target
            if focus_contig:
                contig = focus_contig

        # Pre-compute motif CAs for Kabsch alignment at the streaming/final
        # boundary.
        n_motif_tokens = 0
        input_motif_cas = None
        if contig:
            motif_residues = parse_contig_motif_residues(contig)
            cas = []
            for chain, rid_motif in motif_residues:
                ca_mask = (
                    (atom_array.chain_id == chain)
                    & (atom_array.res_id == rid_motif)
                    & (np.char.strip(atom_array.atom_name) == "CA")
                )
                if np.any(ca_mask):
                    cas.append(atom_array.coord[ca_mask][0])
                    n_motif_tokens += 1
            if cas:
                input_motif_cas = np.array(cas, dtype=np.float64)

        input_ca_centroid = None
        all_ca_mask = np.char.strip(atom_array.atom_name) == "CA"
        all_cas = atom_array.coord[all_ca_mask]
        if len(all_cas) > 0:
            input_ca_centroid = all_cas.mean(axis=0).astype(np.float64)

        outer_self = self

        def step_callback(info):
            try:
                if outer_self._cancel.get(rid):
                    return
                step, total = info.step, info.total_steps
                all_coords = info.coords.detach().cpu()[0].numpy().astype(np.float32)
                n_total_atoms = all_coords.shape[0]
                n_residues = n_total_atoms // 14
                bb_idx = []
                for ri in range(n_residues):
                    base = ri * 14
                    bb_idx.extend([base, base + 1, base + 2, base + 3])
                coords = all_coords[bb_idx]

                motif_mask = getattr(info, "motif_mask", None)
                residue_motif = None
                if motif_mask is not None:
                    motif_mask_np = motif_mask.detach().cpu().numpy().astype(bool)
                    residue_motif = motif_mask_np[::14][:n_residues]

                # Centroid-align via motif (or fallback to all-CA centroid).
                if residue_motif is not None and input_motif_cas is not None:
                    all_cas_local = coords[1::4]
                    motif_cas = all_cas_local[residue_motif[: len(all_cas_local)]]
                    if len(motif_cas) == len(input_motif_cas):
                        offset = (
                            input_motif_cas.mean(axis=0)
                            - motif_cas.mean(axis=0).astype(np.float64)
                        )
                        coords = (
                            coords.astype(np.float64) + offset
                        ).astype(np.float32)
                    elif input_ca_centroid is not None:
                        offset = input_ca_centroid - coords.mean(axis=0).astype(
                            np.float64
                        )
                        coords = (
                            coords.astype(np.float64) + offset
                        ).astype(np.float32)
                elif input_motif_cas is not None and n_motif_tokens > 0:
                    stream_motif_cas = coords[1::4][:n_motif_tokens]
                    if len(stream_motif_cas) == len(input_motif_cas):
                        offset = (
                            input_motif_cas.mean(axis=0)
                            - stream_motif_cas.mean(axis=0).astype(np.float64)
                        )
                        coords = (
                            coords.astype(np.float64) + offset
                        ).astype(np.float32)
                elif input_ca_centroid is not None:
                    offset = input_ca_centroid - coords.mean(axis=0).astype(
                        np.float64
                    )
                    coords = (
                        coords.astype(np.float64) + offset
                    ).astype(np.float32)

                # Strip motif backbone (already visible in original entity).
                if residue_motif is not None:
                    designed_mask = ~residue_motif
                    n_designed = int(designed_mask.sum())
                    if n_designed > 0 and n_designed < n_residues:
                        bb_shaped = coords.reshape(n_residues, 4, 3)
                        coords = bb_shaped[designed_mask].reshape(-1, 3)
                        n_residues = n_designed
                elif n_motif_tokens > 0 and n_residues > n_motif_tokens:
                    motif_bb_count = n_motif_tokens * 4
                    coords = coords[motif_bb_count:]
                    n_residues = n_residues - n_motif_tokens

                from biotite.structure import AtomArray

                n_bb = coords.shape[0]
                aa = AtomArray(n_bb)
                aa.coord = np.ascontiguousarray(coords, dtype=np.float32)
                aa.atom_name = np.array(["N", "CA", "C", "O"] * n_residues)
                aa.element = np.array(["N", "C", "C", "O"] * n_residues)
                aa.res_id = np.repeat(np.arange(1, n_residues + 1), 4)
                aa.res_name = np.array(["ALA"] * n_bb)
                aa.chain_id = np.array(["A"] * n_bb)

                frame_bytes = molex_io.atom_array_to_assembly_bytes(aa)
                outer_self._set_pending(
                    rid, frame_bytes, step / total, f"rfd3-step-{step}/{total}"
                )
            except Exception as e:
                logger.warning(
                    "RFD3 step callback error: %s\n%s", e, traceback.format_exc()
                )

        # A focused ligand target wins; otherwise fall back to every ligand
        # present in the assembly as design context.
        ligand_names = target_ligand or _extract_ligand_names(
            atom_array, assembly_bytes
        )
        spec_dict: dict = {
            "extra": {},
            "input": None,
            "atom_array_input": atom_array,
        }

        if contig:
            parts = [p.strip() for p in contig.split(",") if p.strip()]
            has_designed = any(
                not any(c.isalpha() for c in p) and p != "/0" for p in parts
            )
            if length and not has_designed:
                spec_dict["contig"] = f"{contig},/0,{length}"
            else:
                spec_dict["contig"] = contig
        elif length:
            spec_dict["length"] = length
        if ligand_names:
            spec_dict["ligand"] = ligand_names

        engine._set_out_dir(None)
        inputs_dict = {"design_0": spec_dict}
        design_specifications = engine._multiply_specifications(
            inputs=inputs_dict, n_batches=num_designs,
        )
        ensure_inference_sampler_matches_design_spec(
            design_specifications, engine.inference_sampler_overrides,
        )
        engine.initialize()
        outputs_dict = engine._run_multi(
            design_specifications, step_callback=step_callback
        )

        results = []
        for _example_id, output_list in outputs_dict.items():
            for rfd3_output in output_list:
                out_atom_array = rfd3_output.atom_array
                # Final-output Kabsch alignment to input motif frame.
                if input_motif_cas is not None and n_motif_tokens > 0:
                    out_cas = _extract_motif_cas_by_position(
                        out_atom_array, n_motif_tokens
                    )
                    paired_in, paired_out = [], []
                    for ic, oc in zip(input_motif_cas, out_cas):
                        if oc is not None:
                            paired_in.append(ic)
                            paired_out.append(oc)
                    if len(paired_in) >= 3:
                        paired_in_arr = np.array(paired_in, dtype=np.float64)
                        paired_out_arr = np.array(paired_out, dtype=np.float64)
                        R, t = kabsch_transform(paired_out_arr, paired_in_arr)
                        out_atom_array.coord = apply_kabsch(
                            out_atom_array.coord.astype(np.float64), R, t
                        ).astype(out_atom_array.coord.dtype)
                    elif len(paired_in) >= 1:
                        paired_in_arr = np.array(paired_in, dtype=np.float64)
                        paired_out_arr = np.array(paired_out, dtype=np.float64)
                        t = paired_in_arr.mean(axis=0) - paired_out_arr.mean(axis=0)
                        out_atom_array.coord = (
                            out_atom_array.coord
                            + t.astype(out_atom_array.coord.dtype)
                        )

                # Strip motif atoms after alignment.
                if contig is not None:
                    annots = out_atom_array.get_annotation_categories()
                    if "is_motif_atom_with_fixed_coord" in annots:
                        designed_mask = ~out_atom_array.is_motif_atom_with_fixed_coord.astype(
                            bool
                        )
                        if np.any(designed_mask):
                            out_atom_array = out_atom_array[designed_mask]

                b_factors = out_atom_array.b_factor
                confidence = (
                    float(b_factors.mean()) / 100.0 if len(b_factors) > 0 else 0.0
                )
                trajectory_stack = (
                    rfd3_output.denoised_trajectory_stack
                    if save_trajectories
                    else None
                )
                results.append((out_atom_array, confidence, trajectory_stack))
                if len(results) >= num_designs:
                    break
            if len(results) >= num_designs:
                break

        return results

    # sequence_design (MPNN, INVOKE)

    def _run_sequence_design(
        self, context: DispatchContext, params: dict[str, Any]
    ) -> bytes:
        if not self._mpnn_checkpoint:
            self._rediscover_checkpoints()
        if not self._mpnn_checkpoint:
            raise RuntimeError(
                "MPNN checkpoint not available — run download_weights"
            )
        if not self._assembly:
            raise RuntimeError("No assembly — call init or update_assembly first")

        num_sequences = int(params.get("num_sequences", 10))
        temperature = float(params.get("temperature", 0.1))
        if num_sequences < 1 or num_sequences > 100:
            raise ValueError("num_sequences must be between 1 and 100")
        if temperature < 0.0 or temperature > 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")

        # Selection-derived fixed positions: each ResidueRef →
        # "{chain_id}{residue_index}" string used by MPNN's fixed_residues
        # spec. Chain id requires the assembly to resolve entity_id →
        # chain_id.
        with self._gpu_lock:
            return self._sequence_design_impl(
                num_sequences=num_sequences,
                temperature=temperature,
                context=context,
            )

    def _sequence_design_impl(
        self,
        num_sequences: int,
        temperature: float,
        context: DispatchContext,
    ) -> bytes:
        if self._mpnn_engine is None:
            from mpnn.inference_engines.mpnn import MPNNInferenceEngine

            self._mpnn_engine = MPNNInferenceEngine(
                model_type=self.model_type,
                checkpoint_path=self._mpnn_checkpoint,
                is_legacy_weights=True,
                out_directory=None,
                write_fasta=False,
                write_structures=False,
            )

        from . import molex_io

        assert self._assembly is not None
        atom_array = molex_io.assembly_bytes_to_atom_array_plus(self._assembly)

        # Derive chain roles from the focused entity.
        designed_chains: list[str] | None = None
        fixed_chains: list[str] | None = None

        entity_info = _entity_info_from_assembly(self._assembly, atom_array)
        protein_chains = [
            info["chain_id"]
            for info in entity_info
            if info.get("molecule_type") == "protein" and info.get("chain_id")
        ]
        non_protein_chains = [
            info["chain_id"]
            for info in entity_info
            if info.get("molecule_type") != "protein" and info.get("chain_id")
        ]

        focused_idx = _entity_index_for_id(entity_info, context.focused_entity_id)
        if focused_idx is not None:
            focused = entity_info[focused_idx]
            if focused.get("molecule_type") == "protein":
                designed_chains = [focused["chain_id"]]
                fixed_chains = [
                    c for c in protein_chains if c != focused["chain_id"]
                ] + non_protein_chains

        if designed_chains is None:
            # Fallback: design all protein chains, fix non-proteins.
            designed_chains = protein_chains or None
            fixed_chains = non_protein_chains or None

        # Build EntityId-keyed lookups: chain id, and residue-index → res_id.
        # Each entity's atoms are contiguous in atom_array, so one pass over
        # the concatenated array recovers per-entity residue ordering.
        entity_chain_lookup: dict[int, str] = {}
        entity_res_ids: dict[int, list[int]] = {}
        atom_offset = 0
        for info in entity_info:
            eid = info.get("entity_id")
            if info.get("chain_id") and eid is not None:
                entity_chain_lookup[eid] = info["chain_id"]
            ac = info.get("atom_count", 0)
            end = min(atom_offset + ac, len(atom_array))
            if eid is not None and atom_offset < end:
                seen: list[int] = []
                last = None
                for i in range(atom_offset, end):
                    rid_v = int(atom_array.res_id[i])
                    if rid_v != last:
                        seen.append(rid_v)
                        last = rid_v
                entity_res_ids[eid] = seen
            atom_offset += ac

        def _spec(entity_id: int, residue_index: int) -> str | None:
            """MPNN's "{chain_id}{res_id}" spec for one residue, or None."""
            chain = entity_chain_lookup.get(entity_id)
            resids = entity_res_ids.get(entity_id, [])
            if chain and 0 <= residue_index < len(resids):
                return f"{chain}{resids[residue_index]}"
            return None

        # Two independent sources of fixed residues:
        #   * the user's selection — "hold these where they are";
        #   * the puzzle's design mask — every residue NOT designable is fixed.
        # An empty `designable` means the session gates no design at all, so it
        # contributes nothing rather than fixing everything.
        specs: list[str] = []
        for ref in context.selection:
            spec = _spec(ref.entity_id, ref.residue_index)
            if spec:
                specs.append(spec)

        if context.designable:
            designable_by_entity: dict[int, set[int]] = {}
            for ref in context.designable:
                designable_by_entity.setdefault(ref.entity_id, set()).add(
                    ref.residue_index
                )
            for entity_id, resids in entity_res_ids.items():
                allowed = designable_by_entity.get(entity_id, set())
                for ridx in range(len(resids)):
                    if ridx in allowed:
                        continue
                    spec = _spec(entity_id, ridx)
                    if spec:
                        specs.append(spec)

        # De-duplicate while preserving order (a residue can be both selected
        # and non-designable).
        fixed_residues_list = list(dict.fromkeys(specs)) or None

        input_dict = {
            "structure_path": None,
            "name": "design",
            "number_of_batches": num_sequences,
            "batch_size": 1,
            "temperature": temperature,
            "seed": None,
            "remove_ccds": [],
            "remove_waters": None,
            "occupancy_threshold_sidechain": 0.0,
            "occupancy_threshold_backbone": 0.0,
            "undesired_res_names": [],
            "structure_noise": 0.0,
            "decode_type": "auto_regressive",
            "causality_pattern": "auto_regressive",
            "initialize_sequence_embedding_with_ground_truth": False,
            "features_to_return": None,
            "atomize_side_chains": False,
            "fixed_residues": fixed_residues_list,
            "designed_residues": None,
            "fixed_chains": fixed_chains,
            "designed_chains": designed_chains,
            "bias": None,
            "bias_per_residue": None,
            "omit": ["UNK"],
            "omit_per_residue": None,
            "pair_bias": None,
            "pair_bias_per_residue_pair": None,
            "temperature_per_residue": None,
            "symmetry_groups": None,
            "symmetry_groups_for_training": None,
        }

        outputs = self._mpnn_engine.run(
            input_dicts=[input_dict], atom_arrays=[atom_array]
        )

        sequences: list[str] = []
        scores: list[float] = []
        for output in outputs:
            seq = output.output_dict.get("designed_sequence", "")
            score = output.output_dict.get("sequence_recovery", 0.0)
            sequences.append(seq)
            scores.append(score)

        if not sequences:
            raise RuntimeError("No sequences generated")

        # Encode result as newline-separated "sequence\tscore" lines, same
        # as dummy.py. Consuming GUI panel agrees on this encoding.
        lines = [f"{s}\t{sc:.4f}" for s, sc in zip(sequences, scores)]
        return "\n".join(lines).encode("utf-8")

    # apply_sequence (INVOKE)

    def _run_mpnn_design(
        self, context: DispatchContext, params: dict[str, Any]
    ) -> bytes:
        """Sample sequences with LigandMPNN and commit the best-scoring one."""
        num_sequences = max(1, min(int(params.get("num_sequences", 8)), 32))
        temperature = float(params.get("temperature", 0.1))

        raw = self._sequence_design_impl(num_sequences, temperature, context)
        best_seq, best_score = "", float("-inf")
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            seq, _, score_text = line.partition("\t")
            try:
                score = float(score_text)
            except ValueError:
                continue
            if seq and score > best_score:
                best_seq, best_score = seq, score
        if not best_seq:
            raise RuntimeError("LigandMPNN returned no usable sequence")

        logger.info(
            "mpnn_design: applying best of %d candidates (recovery=%.4f)",
            num_sequences, best_score,
        )
        return self._run_apply_sequence(context, {"sequence": best_seq})

    def _run_apply_sequence(
        self, context: DispatchContext, params: dict[str, Any]
    ) -> bytes:
        """Commit a sequence onto the focused protein chain.

        Each designed residue is rebuilt from a biotite template
        (`biotite.structure.info.residue`) superposed onto the existing
        N/CA/C backbone via Kabsch. Sidechain rotamers are the template
        defaults; expect a downstream rosetta repack to refine clashes.
        """
        if not self._assembly:
            raise RuntimeError(
                "apply_sequence: no assembly — call init or update_assembly first"
            )

        raw_sequence = params.get("sequence", "")
        if not isinstance(raw_sequence, str):
            raise ValueError(
                f"apply_sequence: 'sequence' must be a string, got "
                f"{type(raw_sequence).__name__}"
            )
        sequence = raw_sequence.strip().upper()
        if not sequence:
            raise ValueError("apply_sequence: 'sequence' must be non-empty")

        from biotite.structure import AtomArray, concatenate
        from biotite.structure.info import residue as residue_template

        from . import molex_io

        # Use the basic (non-`_plus`) decoder: apply_sequence only needs
        # chain_id / res_id / res_name / atom_name / coord, and the
        # AtomArrayPlus subclass's slicing path breaks on assemblies
        # loaded without a CCD-backed mirror.
        atom_array = molex_io.assembly_bytes_to_atom_array(self._assembly)
        entity_info = _entity_info_from_assembly(self._assembly, atom_array)

        # Resolve designed entity via focused_entity_id (mirrors
        # _sequence_design_impl); fall back to the first protein entity.
        designed_idx: int | None = None
        idx = _entity_index_for_id(entity_info, context.focused_entity_id)
        if idx is not None and entity_info[idx].get("molecule_type") == "protein":
            designed_idx = idx
        if designed_idx is None:
            for i, info in enumerate(entity_info):
                if info.get("molecule_type") == "protein":
                    designed_idx = i
                    break
        if designed_idx is None:
            raise ValueError(
                "apply_sequence: no protein entity to apply the sequence to"
            )

        atom_start = sum(
            info.get("atom_count", 0)
            for info in entity_info[:designed_idx]
        )
        atom_end = atom_start + entity_info[designed_idx].get("atom_count", 0)
        if atom_end > len(atom_array):
            raise RuntimeError(
                f"apply_sequence: entity_info atom_count exceeds atom_array "
                f"length ({atom_end} > {len(atom_array)})"
            )

        designed_segment = atom_array[atom_start:atom_end]

        # Group designed segment atoms by residue (contiguous res_id runs).
        boundaries: list[tuple[int, int]] = []
        if len(designed_segment) > 0:
            run_start = 0
            run_res = int(designed_segment.res_id[0])
            for i in range(1, len(designed_segment)):
                rid = int(designed_segment.res_id[i])
                if rid != run_res:
                    boundaries.append((run_start, i))
                    run_start = i
                    run_res = rid
            boundaries.append((run_start, len(designed_segment)))

        if len(boundaries) != len(sequence):
            raise ValueError(
                f"apply_sequence: sequence length {len(sequence)} does not "
                f"match designed chain residue count {len(boundaries)}"
            )

        new_segments: list[AtomArray] = []
        for (s, e), one_letter in zip(boundaries, sequence):
            try:
                new_three = _AA_ONE_TO_THREE[one_letter]
            except KeyError as ke:
                raise ValueError(
                    f"apply_sequence: unsupported one-letter code "
                    f"{one_letter!r}; expected one of "
                    f"{''.join(sorted(_AA_ONE_TO_THREE))}"
                ) from ke
            target_residue = designed_segment[s:e]
            new_segments.append(
                _build_template_residue(
                    target_residue, new_three, residue_template
                )
            )

        new_designed = concatenate(new_segments)

        if atom_start > 0 and atom_end < len(atom_array):
            new_array = concatenate([
                atom_array[:atom_start],
                new_designed,
                atom_array[atom_end:],
            ])
        elif atom_start > 0:
            new_array = concatenate([atom_array[:atom_start], new_designed])
        elif atom_end < len(atom_array):
            new_array = concatenate([new_designed, atom_array[atom_end:]])
        else:
            new_array = new_designed

        result_bytes = molex_io.atom_array_to_assembly_bytes(new_array)
        # Keep the plugin's local view in sync; the host's post-invoke
        # broadcast skips the originating plugin, so without this our next
        # op would still see the pre-edit assembly.
        self._assembly = result_bytes
        return result_bytes
