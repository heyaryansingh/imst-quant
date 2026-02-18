"""Entity linker precision evaluation for ENT-04."""

from pathlib import Path

from .linker import EntityLinker


def evaluate_precision(
    linker: EntityLinker,
    test_set_path: Path | str,
) -> float:
    """
    Compute macro-averaged precision over test set.

    Per row: precision = |predicted ∩ expected| / |predicted|
    if |predicted| > 0 else 1.0 if |expected| == 0 else 0.0
    """
    path = Path(test_set_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    import csv

    precisions = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "")
            subreddit = row.get("subreddit", "")
            expected_str = row.get("expected_tickers", "")
            expected = {
                t.strip().upper()
                for t in expected_str.split(",")
                if t.strip()
            }
            links = linker.link_entities(text, subreddit)
            predicted = {l.asset_id for l in links}
            if len(predicted) > 0:
                prec = len(predicted & expected) / len(predicted)
            else:
                prec = 1.0 if len(expected) == 0 else 0.0
            precisions.append(prec)

    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)
