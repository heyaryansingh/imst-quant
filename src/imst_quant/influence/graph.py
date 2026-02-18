"""Monthly interaction graph construction (INF-01, INF-04)."""

import re
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set

import networkx as nx
import polars as pl


def _extract_mentions(text: str) -> List[str]:
    """Extract u/ mentions from text."""
    if not text:
        return []
    return re.findall(r"u/([A-Za-z0-9_-]+)", text, re.IGNORECASE)


class InteractionGraph:
    """Build monthly interaction graph from posts (INF-01, INF-04)."""

    def __init__(self, min_interactions: int = 50):
        self.min_interactions = min_interactions

    def build_from_dataframe(
        self,
        df: pl.DataFrame,
        year: int,
        month: int,
    ) -> nx.DiGraph:
        """Build graph for year-month from posts DataFrame.

        Expects columns: author_id, subreddit, date. Edges: co-subreddit-date.
        """
        prefix = f"{year}-{month:02d}"
        if "date" in df.columns:
            month_df = df.filter(pl.col("date").str.starts_with(prefix))
        else:
            month_df = df

        if len(month_df) == 0:
            return nx.DiGraph()

        author_interactions = Counter()
        subreddit_date_authors: Dict[tuple, Set[str]] = defaultdict(set)

        for r in month_df.iter_rows(named=True):
            aid = str(r.get("author_id", ""))
            if not aid:
                continue
            author_interactions[aid] += 1
            sub = str(r.get("subreddit", ""))
            dt = r.get("date", "")
            if isinstance(dt, datetime):
                dt = dt.strftime("%Y-%m-%d")
            if aid and sub:
                subreddit_date_authors[(sub, dt)].add(aid)

        edges = []
        for (_, _), authors in subreddit_date_authors.items():
            if len(authors) > 1:
                for a1, a2 in combinations(authors, 2):
                    edges.append((a1, a2, "co_thread"))
                    edges.append((a2, a1, "co_thread"))

        eligible = {a for a, c in author_interactions.items() if c >= self.min_interactions}
        if len(eligible) < 2:
            return nx.DiGraph()

        G = nx.DiGraph()
        for a in eligible:
            G.add_node(a)
        for src, dst, typ in edges:
            if src in eligible and dst in eligible:
                if G.has_edge(src, dst):
                    G[src][dst]["weight"] = G[src][dst].get("weight", 1) + 1
                else:
                    G.add_edge(src, dst, weight=1, type=typ)
        return G


def build_monthly_graph(
    silver_dir: Path,
    year: int,
    month: int,
    min_interactions: int = 50,
) -> nx.DiGraph:
    """Load silver posts for month and build interaction graph."""
    silver_dir = Path(silver_dir)
    parts = list((silver_dir / "reddit").glob("date=*/posts_enriched.parquet"))
    month_str = f"{year}-{month:02d}"
    dfs = []
    for p in parts:
        d = p.parent.name.replace("date=", "")
        if d.startswith(month_str):
            dfs.append(pl.read_parquet(p))
    if not dfs:
        return nx.DiGraph()
    df = pl.concat(dfs)
    builder = InteractionGraph(min_interactions=min_interactions)
    return builder.build_from_dataframe(df, year, month)
