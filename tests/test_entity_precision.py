"""ENT-04: Entity linker precision >= 90% on 200-sample test set."""

from pathlib import Path

import pytest

from imst_quant.entities.evaluate import evaluate_precision
from imst_quant.entities.linker import EntityLinker


@pytest.fixture
def fixture_path():
    return Path(__file__).parent / "fixtures" / "entity_test_set.csv"


@pytest.fixture
def linker():
    return EntityLinker(confidence_threshold=0.35)


def test_entity_precision_ent04(linker, fixture_path):
    """ENT-04: Entity linker achieves >= 90% precision on labeled test set."""
    assert fixture_path.exists(), f"Missing fixture: {fixture_path}"
    precision = evaluate_precision(linker, fixture_path)
    assert precision >= 0.90, (
        f"ENT-04 failed: precision {precision:.2%} < 90%"
    )
