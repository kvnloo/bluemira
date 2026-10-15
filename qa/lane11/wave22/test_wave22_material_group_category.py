"""MOAB material-group identity guard — synthetic metadata only, no native transport."""

import h5py

from h5m_raw_group_audit import audit_raw_groups
from test_wave20_raw_material_groups import mk_h5m


def test_material_name_on_volume_category_does_not_claim_pass(tmp_path):
    path = mk_h5m(tmp_path)
    with h5py.File(path, "r+") as db:
        db["tstt/tags/CATEGORY/values"][2] = b"Volume"
    rec = audit_raw_groups(path)
    assert rec["status"] == "UNKNOWN", rec
    assert rec["invalid_material_groups"] == {"102": "CATEGORY=Volume, expected Group"}
    assert rec["unassigned_volume_ids"] == [1]
    assert rec["native_transport_verified"] is False


def test_missing_category_for_material_set_does_not_claim_pass(tmp_path):
    path = mk_h5m(tmp_path)
    with h5py.File(path, "r+") as db:
        db["tstt/tags/CATEGORY/values"][2] = b""
    rec = audit_raw_groups(path)
    assert rec["status"] == "UNKNOWN", rec
    assert rec["invalid_material_groups"] == {"102": "CATEGORY missing, expected Group"}
    assert rec["unassigned_volume_ids"] == [1]


def test_empty_material_name_does_not_claim_pass(tmp_path):
    path = mk_h5m(tmp_path)
    with h5py.File(path, "r+") as db:
        db["tstt/tags/NAME/values"][0] = b"mat:"
    rec = audit_raw_groups(path)
    assert rec["status"] == "UNKNOWN", rec
    assert rec["invalid_material_groups"] == {"102": "empty mat: NAME"}
    assert rec["unassigned_volume_ids"] == [1]


def test_invalid_unused_mat_group_does_not_allow_pass(tmp_path):
    path = mk_h5m(tmp_path)
    with h5py.File(path, "r+") as db:
        # Both volumes still explicitly covered by first material group,
        # while the unused other group is malformed.
        db["tstt/sets/contents"][:] = [100, 101]
        db["tstt/sets/list"][2, 0] = 1
        db["tstt/sets/list"][3, 0] = 1
        db["tstt/tags/CATEGORY/values"][3] = b"Volume"
    rec = audit_raw_groups(path)
    assert rec["status"] == "UNKNOWN", rec
    assert rec["invalid_material_groups"] == {"103": "CATEGORY=Volume, expected Group"}


def test_known_material_group_category_keeps_scoped_pass(tmp_path):
    rec = audit_raw_groups(mk_h5m(tmp_path))
    assert rec["status"] == "PASS", rec
    assert rec["invalid_material_groups"] == {}
    assert rec["native_transport_verified"] is False


def test_crosswalk_cannot_promote_misclassified_group_to_scoped_pass(tmp_path, monkeypatch):
    import h5m_volume_material_crosswalk as crosswalk
    from test_wave20_raw_material_groups import fake_factory

    path = mk_h5m(tmp_path)
    with h5py.File(path, "r+") as db:
        db["tstt/tags/CATEGORY/values"][2] = b"Volume"
    monkeypatch.setattr(crosswalk, "_import_inspector_factory",
                        lambda: fake_factory((1, 2), {1: "steel", 2: "copper"},
                                             ("steel", "copper")))
    rec = crosswalk.inspect_crosswalk(
        path, {"moab_name_tags": ["steel", "copper"], "expected_bom": None}
    )
    assert rec["status"] == "UNKNOWN", rec
    assert rec["raw_material_group_audit"]["invalid_material_groups"] == {
        "102": "CATEGORY=Volume, expected Group"
    }
    assert rec["native_transport_verified"] is False
