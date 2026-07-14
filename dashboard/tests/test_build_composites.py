from dashboard.build_composites import sequences


def test_sequence_discovery_has_baked_valid_clips():
    found = sequences("valid")
    assert 21 in found
