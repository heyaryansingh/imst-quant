"""Load ticker dictionary and alias mappings from config."""

from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# Project root: src/imst_quant/entities/ -> 4 levels up to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_ALIAS_PATH = _PROJECT_ROOT / "config" / "ticker_aliases.yaml"


def load_aliases(
    path: Path | str | None = None,
) -> Tuple[Dict, Dict[str, List[str]], Dict[str, List[str]]]:
    """Load ticker dict, company_aliases, crypto_aliases from YAML."""
    p = Path(path) if path else DEFAULT_ALIAS_PATH
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    data = yaml.safe_load(p.read_text())
    ticker_dict = data.get("tickers", {})
    company_aliases = data.get("company_aliases", {})
    crypto_aliases = data.get("crypto_aliases", {})
    return ticker_dict, company_aliases, crypto_aliases
