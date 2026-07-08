"""Biotite <-> molex marshaling, owned by foundry.

Biotite is foundry's private dependency, not the shared SDK's, so the
glue that turns a biotite ``AtomArray`` into molex assembly wire bytes
(and back) lives here rather than inside molex. molex exposes the
biotite-agnostic columnar interchange ``molex.PyAtomTable``; these three
functions adapt it to the exact annotation set the Foundry models expect.
"""

import biotite.structure as bs
import numpy as np


def _opt_annotation(atom_array, name):
    """Return annotation ``name`` if set on ``atom_array``, else ``None``."""
    if name in atom_array.get_annotation_categories():
        return atom_array.get_annotation(name)
    return None


def atom_array_to_assembly_bytes(atom_array) -> bytes:
    """Convert a biotite ``AtomArray`` to molex assembly wire bytes.

    Entity boundaries follow the ``entity_id`` annotation when present,
    otherwise molex groups on ``(chain_id, mol_type)``. molex's three-tier
    classification (chain_type -> mol_type -> residue name) covers
    annotation-free input, so absent annotations are passed as ``None``.
    """
    import molex

    n = len(atom_array)

    coords = np.ascontiguousarray(atom_array.coord, dtype=np.float32)
    chain_ids = [str(c) for c in atom_array.chain_id]
    res_ids = np.ascontiguousarray(atom_array.res_id, dtype=np.int32)
    res_names = atom_array.res_name
    atom_names = atom_array.atom_name

    elements = _opt_annotation(atom_array, "element")
    if elements is None:
        # Blank elements: molex infers each from its atom name.
        elements = np.full(n, "", dtype="U1")

    occupancy = _opt_annotation(atom_array, "occupancy")
    if occupancy is not None:
        occupancy = np.ascontiguousarray(occupancy, dtype=np.float32)
    b_factor = _opt_annotation(atom_array, "b_factor")
    if b_factor is not None:
        b_factor = np.ascontiguousarray(b_factor, dtype=np.float32)

    entity_id = _opt_annotation(atom_array, "entity_id")
    if entity_id is not None:
        entity_id = np.ascontiguousarray(entity_id, dtype=np.int32)
    mol_type = _opt_annotation(atom_array, "mol_type")
    chain_type = _opt_annotation(atom_array, "chain_type")
    if chain_type is not None:
        chain_type = np.ascontiguousarray(chain_type, dtype=np.int32)

    table = molex.PyAtomTable.from_columns(
        coords,
        atom_names,
        elements,
        res_ids,
        res_names,
        chain_ids,
        occupancy,
        b_factor,
        mol_type,
        chain_type,
        entity_id,
    )
    return table.to_assembly_bytes()


def assembly_bytes_to_atom_array(assembly_bytes):
    """Convert molex assembly wire bytes to a biotite ``AtomArray``."""
    import molex

    t = molex.PyAtomTable.from_assembly_bytes(assembly_bytes)
    n = len(t)
    if n == 0:
        return bs.AtomArray(0)

    aa = bs.AtomArray(n)
    aa.coord = np.asarray(t.coords, dtype=np.float32)
    aa.chain_id = np.asarray(t.chain_ids)
    aa.res_id = np.asarray(t.res_ids, dtype=np.int32)
    aa.element = np.asarray(t.elements)
    # molex hands back fixed-width byte strings ('S4'/'S3'); biotite wants
    # unicode ('U6'/'U5') so str(aa.atom_name[i]) reads "CA", not "b'CA'".
    aa.atom_name = np.char.decode(t.atom_names, "ascii").astype("U6")
    aa.res_name = np.char.decode(t.res_names, "ascii").astype("U5")

    aa.set_annotation("occupancy", np.asarray(t.occupancies, dtype=np.float32))
    aa.set_annotation("b_factor", np.asarray(t.b_factors, dtype=np.float32))
    aa.set_annotation("entity_id", np.asarray(t.entity_ids, dtype=np.int32))
    aa.set_annotation("mol_type", np.asarray(t.mol_types))
    aa.set_annotation("chain_type", np.asarray(t.chain_types, dtype=np.int32))

    bond_list = bs.BondList(n)
    for i, j, order in t.bonds():
        bond_list.add_bond(int(i), int(j), int(order))
    aa.bonds = bond_list

    return aa


def assembly_bytes_to_atom_array_plus(assembly_bytes):
    """As :func:`assembly_bytes_to_atom_array`, wrapped as an ``AtomArrayPlus``.

    The ``AtomArrayPlus`` marker tells downstream Foundry consumers the
    structure is already fully constructed and CCD template rebuilding can
    be skipped.
    """
    from atomworks.io.utils.atom_array_plus import as_atom_array_plus

    return as_atom_array_plus(assembly_bytes_to_atom_array(assembly_bytes))
