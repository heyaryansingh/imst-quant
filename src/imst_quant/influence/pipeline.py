"""Monthly influence pipeline and score persistence (INF-05)."""

from pathlib import Path
from typing import Dict

import networkx as nx
import polars as pl
import structlog

from .gnn import InfluenceGNN, prepare_gnn_data
from .graph import build_monthly_graph
from .trainer import InfluenceTrainer

logger = structlog.get_logger()


def run_influence_month(
    silver_dir: Path,
    output_dir: Path,
    year: int,
    month: int,
    min_interactions: int = 50,
) -> Path | None:
    """Build graph, train GCN, write influence scores. Returns output path or None."""
    silver_dir = Path(silver_dir)
    output_dir = Path(output_dir)
    G = build_monthly_graph(silver_dir, year, month, min_interactions)

    if G.number_of_nodes() < 2:
        logger.info("influence_skip_small_graph", year=year, month=month, nodes=G.number_of_nodes())
        return None

    model = InfluenceGNN(input_dim=4, hidden_dim=64, output_dim=1)
    trainer = InfluenceTrainer(model, lr=0.01, epochs=50)
    scores = trainer.train(G)

    rows = [{"author_id": aid, "influence_score": s} for aid, s in scores.items()]
    df = pl.DataFrame(rows)
    df = df.with_columns(pl.lit(f"{year}-{month:02d}").alias("month"))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"influence_{year}-{month:02d}.parquet"
    df.write_parquet(out_path)
    logger.info("influence_written", path=str(out_path), authors=len(scores))
    return out_path


def load_influence_scores(
    influence_dir: Path,
    as_of_month: str,
) -> Dict[str, float]:
    """Load influence scores for given month (carry forward for daily inference)."""
    influence_dir = Path(influence_dir)
    path = influence_dir / f"influence_{as_of_month}.parquet"
    if not path.exists():
        parts = sorted(influence_dir.glob("influence_*.parquet"))
        if not parts:
            return {}
        path = max(parts, key=lambda p: p.stem)
    df = pl.read_parquet(path)
    return dict(zip(df["author_id"].to_list(), df["influence_score"].to_list()))
