
import pytest

from agents.alignment_agent.schemas import AlignmentStatus
from agents.policy_agent.schemas import PolicyCategory
from evaluation.schema_mapping import (
    map_expected_category,
    map_expected_status,
)


def test_map_valid_status():
    assert map_expected_status("aligned") == AlignmentStatus.ALIGNED
    assert map_expected_status("partial") == AlignmentStatus.PARTIAL
    assert map_expected_status("violated") == AlignmentStatus.VIOLATED
    assert map_expected_status("out_of_scope") == AlignmentStatus.OUT_OF_SCOPE


def test_map_valid_category():
    assert (
        map_expected_category("transparency_notice")
        == PolicyCategory.TRANSPARENCY_NOTICE
    )
    assert (
        map_expected_category("consent")
        == PolicyCategory.CONSENT
    )
    assert (
        map_expected_category("storage_retention")
        == PolicyCategory.STORAGE_RETENTION
    )


def test_invalid_status_raises_error():
    with pytest.raises(ValueError, match="Unknown expected status"):
        map_expected_status("not_a_status")


def test_invalid_category_raises_error():
    with pytest.raises(ValueError, match="Unknown policy category"):
        map_expected_category("not_a_category")
