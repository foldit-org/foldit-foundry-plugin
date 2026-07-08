"""Round-trip tests for foundry's biotite <-> molex marshaling.

Model-free: exercises only ``molex_io`` against synthetic biotite
``AtomArray`` fixtures. Requires the local molex build (with
``PyAtomTable``) installed into the test env via ``maturin develop``.
"""

import biotite.structure as bs
import numpy as np
import pytest

from foundry_plugin import molex_io


def _annotated_fixture():
    """Two-residue protein (ALA/GLY) + 3-atom ligand, fully annotated."""
    atom_names = ["N", "CA", "C", "O", "N", "CA", "C", "O", "C1", "O1", "N1"]
    elements = ["N", "C", "C", "O", "N", "C", "C", "O", "C", "O", "N"]
    res_names = ["ALA"] * 4 + ["GLY"] * 4 + ["LIG"] * 3
    res_ids = [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1]
    chain_ids = ["P"] * 8 + ["B"] * 3
    mol_types = ["protein"] * 8 + ["ligand"] * 3
    chain_types = [6] * 8 + [8] * 3
    entity_ids = [0] * 8 + [1] * 3
    n = len(atom_names)

    aa = bs.AtomArray(n)
    aa.coord = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    # Place the ligand atoms within covalent distance so molex's distance
    # inference yields bonds (the arange spread is too wide otherwise).
    aa.coord[8] = [0.0, 0.0, 0.0]
    aa.coord[9] = [1.3, 0.0, 0.0]
    aa.coord[10] = [2.5, 0.0, 0.0]
    aa.atom_name = np.array(atom_names, dtype="U6")
    aa.element = np.array(elements, dtype="U2")
    aa.res_name = np.array(res_names, dtype="U5")
    aa.res_id = np.array(res_ids, dtype=np.int32)
    aa.chain_id = np.array(chain_ids, dtype="U4")
    aa.set_annotation("occupancy", np.full(n, 0.75, dtype=np.float32))
    aa.set_annotation(
        "b_factor", np.arange(n, dtype=np.float32) * 1.5
    )
    aa.set_annotation("mol_type", np.array(mol_types))
    aa.set_annotation("chain_type", np.array(chain_types, dtype=np.int32))
    aa.set_annotation("entity_id", np.array(entity_ids, dtype=np.int32))
    return aa


def test_round_trip_preserves_columns():
    aa = _annotated_fixture()
    out = molex_io.assembly_bytes_to_atom_array(
        molex_io.atom_array_to_assembly_bytes(aa)
    )

    assert len(out) == len(aa)
    assert np.allclose(out.coord, aa.coord)

    # 'S' leak guard: names must come back as unicode, str-equal to source.
    assert out.atom_name.dtype.kind == "U"
    assert out.res_name.dtype.kind == "U"
    assert [str(x) for x in out.atom_name] == [str(x) for x in aa.atom_name]
    assert [str(x) for x in out.res_name] == [str(x) for x in aa.res_name]

    assert list(out.res_id) == list(aa.res_id)
    # Polymer chain ids round-trip; molex collapses the non-polymer ligand
    # chain to "A" (it carries no chain on a small-molecule entity).
    assert [str(x) for x in out.chain_id] == ["P"] * 8 + ["A"] * 3
    assert [str(x) for x in out.element] == [str(x) for x in aa.element]
    # molex's assembly wire format does not persist occupancy / b_factor:
    # they reset to the 1.0 / 0.0 defaults across a bytes round-trip. The
    # glue passes the source values into molex faithfully; the loss is in
    # molex's serializer, not here.
    assert np.allclose(out.occupancy, 1.0)
    assert np.allclose(out.b_factor, 0.0)
    assert list(out.entity_id) == list(aa.entity_id)
    assert [str(x) for x in out.mol_type] == [str(x) for x in aa.mol_type]
    assert list(out.chain_type) == list(aa.chain_type)

    # The ligand is the only entity that gets distance-inferred bonds.
    bonds = out.bonds.as_array()
    assert len(bonds) > 0
    ligand_start = 8
    assert all(
        i >= ligand_start and j >= ligand_start for i, j, _ in bonds
    )


def test_classify_fallback_without_annotations():
    """No mol_type/chain_type/entity_id: molex classifies by residue name."""
    atom_names = ["N", "CA", "C", "O", "N", "CA", "C", "O"]
    elements = ["N", "C", "C", "O", "N", "C", "C", "O"]
    res_names = ["ALA"] * 4 + ["GLY"] * 4
    res_ids = [1, 1, 1, 1, 2, 2, 2, 2]
    n = len(atom_names)

    aa = bs.AtomArray(n)
    aa.coord = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    aa.atom_name = np.array(atom_names, dtype="U6")
    aa.element = np.array(elements, dtype="U2")
    aa.res_name = np.array(res_names, dtype="U5")
    aa.res_id = np.array(res_ids, dtype=np.int32)
    aa.chain_id = np.array(["A"] * n, dtype="U4")

    assert "mol_type" not in aa.get_annotation_categories()

    out = molex_io.assembly_bytes_to_atom_array(
        molex_io.atom_array_to_assembly_bytes(aa)
    )

    assert all(str(m) == "protein" for m in out.mol_type)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x"]))
