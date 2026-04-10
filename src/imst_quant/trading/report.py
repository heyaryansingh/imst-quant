"""Backtest report generation and export utilities.

This module provides functionality to generate comprehensive backtest reports
and export them in various formats (CSV, JSON, HTML) for analysis and sharing.

Functions:
    generate_report: Create comprehensive backtest report dictionary
    export_csv: Export backtest results to CSV format
    export_json: Export backtest results to JSON format
    export_html: Generate HTML report with charts
    format_metrics: Format metrics for display

Example:
    >>> from imst_quant.trading.report import generate_report, export_json
    >>> from pathlib import Path
    >>> results = {"total_pnl": 0.15, "sharpe": 1.2, "trades": 50}
    >>> report = generate_report(results, strategy_name="momentum")
    >>> export_json(report, Path("backtest_report.json"))
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl


def generate_report(
    backtest_results: Dict[str, Any],
    strategy_name: str = "unnamed",
    description: str = "",
    parameters: Optional[Dict[str, Any]] = None,
    daily_pnl: Optional[pl.Series] = None,
    trades_df: Optional[pl.DataFrame] = None,
) -> Dict[str, Any]:
    """Generate a comprehensive backtest report.

    Creates a structured report combining backtest metrics, strategy metadata,
    and optional daily PnL/trade details.

    Args:
        backtest_results: Dictionary containing backtest metrics (total_pnl,
            sharpe, trades, etc.) from run_backtest().
        strategy_name: Name of the strategy being tested.
        description: Optional description of the strategy logic.
        parameters: Dictionary of strategy parameters used.
        daily_pnl: Optional Series of daily PnL values for detailed analysis.
        trades_df: Optional DataFrame with individual trade records.

    Returns:
        Comprehensive report dictionary with sections:
        - metadata: Report generation info and strategy name
        - summary: Key performance metrics
        - parameters: Strategy configuration
        - analysis: Derived statistics (win rate, avg trade, etc.)
        - daily_pnl: Daily returns if provided
        - trades: Trade-level details if provided

    Example:
        >>> results = {"total_pnl": 0.15, "sharpe": 1.2, "trades": 50}
        >>> report = generate_report(results, strategy_name="MA Crossover")
        >>> print(report["summary"]["total_pnl"])
        0.15
    """
    report: Dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "strategy_name": strategy_name,
            "description": description,
            "version": "1.0.0",
        },
        "summary": {
            "total_pnl": backtest_results.get("total_pnl", 0.0),
            "sharpe_ratio": backtest_results.get("sharpe", 0.0),
            "total_trades": backtest_results.get("trades", 0),
            "max_drawdown": backtest_results.get("max_drawdown", 0.0),
            "sortino_ratio": backtest_results.get("sortino", 0.0),
            "calmar_ratio": backtest_results.get("calmar", 0.0),
            "var_95": backtest_results.get("var", 0.0),
        },
        "parameters": parameters or {},
    }

    # Add analysis section with derived metrics
    analysis: Dict[str, Any] = {}

    if daily_pnl is not None and daily_pnl.len() > 0:
        positive_days = daily_pnl.filter(daily_pnl > 0).len()
        negative_days = daily_pnl.filter(daily_pnl < 0).len()
        total_days = daily_pnl.len()

        analysis["trading_days"] = total_days
        analysis["positive_days"] = positive_days
        analysis["negative_days"] = negative_days
        analysis["win_rate"] = positive_days / total_days if total_days > 0 else 0.0
        analysis["avg_daily_pnl"] = float(daily_pnl.mean()) if daily_pnl.mean() else 0.0
        analysis["best_day"] = float(daily_pnl.max()) if daily_pnl.max() else 0.0
        analysis["worst_day"] = float(daily_pnl.min()) if daily_pnl.min() else 0.0
        analysis["daily_volatility"] = float(daily_pnl.std()) if daily_pnl.std() else 0.0

        report["daily_pnl"] = daily_pnl.to_list()

    if trades_df is not None and trades_df.height > 0:
        if "pnl" in trades_df.columns:
            winning_trades = trades_df.filter(pl.col("pnl") > 0).height
            losing_trades = trades_df.filter(pl.col("pnl") < 0).height
            total_trades = trades_df.height

            analysis["winning_trades"] = winning_trades
            analysis["losing_trades"] = losing_trades
            analysis["trade_win_rate"] = winning_trades / total_trades if total_trades > 0 else 0.0

            avg_win = trades_df.filter(pl.col("pnl") > 0)["pnl"].mean()
            avg_loss = trades_df.filter(pl.col("pnl") < 0)["pnl"].mean()

            analysis["avg_winning_trade"] = float(avg_win) if avg_win else 0.0
            analysis["avg_losing_trade"] = float(avg_loss) if avg_loss else 0.0

            if avg_loss and avg_loss != 0:
                analysis["profit_factor"] = abs(
                    float(avg_win or 0) * winning_trades / (float(avg_loss) * losing_trades)
                ) if losing_trades > 0 else float("inf")
            else:
                analysis["profit_factor"] = float("inf") if winning_trades > 0 else 0.0

        report["trades"] = trades_df.to_dicts()

    report["analysis"] = analysis
    return report


def export_csv(
    report: Dict[str, Any],
    output_path: Path,
    include_daily: bool = True,
    include_trades: bool = True,
) -> List[Path]:
    """Export backtest report to CSV files.

    Creates separate CSV files for summary metrics, daily PnL, and trades
    to enable easy analysis in spreadsheet software.

    Args:
        report: Report dictionary from generate_report().
        output_path: Base path for output files. Actual files will be named
            with suffixes (_summary.csv, _daily.csv, _trades.csv).
        include_daily: Whether to export daily PnL data. Defaults to True.
        include_trades: Whether to export trade details. Defaults to True.

    Returns:
        List of paths to created CSV files.

    Example:
        >>> paths = export_csv(report, Path("results/backtest"))
        >>> print(paths)  # ['results/backtest_summary.csv', ...]
    """
    created_files: List[Path] = []
    base_name = output_path.stem
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export summary metrics
    summary_data = {
        "metric": [],
        "value": [],
    }
    for key, value in report.get("summary", {}).items():
        summary_data["metric"].append(key)
        summary_data["value"].append(value)

    # Add analysis metrics
    for key, value in report.get("analysis", {}).items():
        summary_data["metric"].append(key)
        summary_data["value"].append(value)

    summary_df = pl.DataFrame(summary_data)
    summary_path = output_dir / f"{base_name}_summary.csv"
    summary_df.write_csv(summary_path)
    created_files.append(summary_path)

    # Export daily PnL
    if include_daily and "daily_pnl" in report:
        daily_df = pl.DataFrame({
            "day": list(range(1, len(report["daily_pnl"]) + 1)),
            "pnl": report["daily_pnl"],
        })
        daily_path = output_dir / f"{base_name}_daily.csv"
        daily_df.write_csv(daily_path)
        created_files.append(daily_path)

    # Export trades
    if include_trades and "trades" in report:
        trades_df = pl.DataFrame(report["trades"])
        trades_path = output_dir / f"{base_name}_trades.csv"
        trades_df.write_csv(trades_path)
        created_files.append(trades_path)

    return created_files


def export_json(
    report: Dict[str, Any],
    output_path: Path,
    indent: int = 2,
) -> Path:
    """Export backtest report to JSON format.

    Creates a single JSON file containing the complete report structure,
    suitable for programmatic analysis or archival.

    Args:
        report: Report dictionary from generate_report().
        output_path: Path for the output JSON file.
        indent: JSON indentation level for readability. Defaults to 2.

    Returns:
        Path to the created JSON file.

    Example:
        >>> path = export_json(report, Path("results/backtest.json"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=indent, default=str)

    return output_path


def export_html(
    report: Dict[str, Any],
    output_path: Path,
    title: str = "Backtest Report",
) -> Path:
    """Generate an HTML report with formatted metrics.

    Creates a self-contained HTML file with styled tables showing
    backtest results. Does not require external dependencies.

    Args:
        report: Report dictionary from generate_report().
        output_path: Path for the output HTML file.
        title: Title for the HTML report. Defaults to "Backtest Report".

    Returns:
        Path to the created HTML file.

    Example:
        >>> path = export_html(report, Path("results/report.html"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = report.get("metadata", {})
    summary = report.get("summary", {})
    analysis = report.get("analysis", {})
    parameters = report.get("parameters", {})

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .positive {{ color: #28a745; font-weight: 600; }}
        .negative {{ color: #dc3545; font-weight: 600; }}
        .metric-value {{ font-family: 'Monaco', 'Consolas', monospace; }}
        .metadata {{ color: #888; font-size: 0.9em; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="metadata">
            <strong>Strategy:</strong> {metadata.get('strategy_name', 'N/A')} |
            <strong>Generated:</strong> {metadata.get('generated_at', 'N/A')}
        </div>

        <h2>Performance Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
"""

    # Add summary metrics
    for key, value in summary.items():
        formatted_value = format_metric_value(key, value)
        css_class = get_value_class(key, value)
        html_content += f'            <tr><td>{format_metric_name(key)}</td><td class="metric-value {css_class}">{formatted_value}</td></tr>\n'

    html_content += """        </table>

        <h2>Analysis</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
"""

    # Add analysis metrics
    for key, value in analysis.items():
        formatted_value = format_metric_value(key, value)
        css_class = get_value_class(key, value)
        html_content += f'            <tr><td>{format_metric_name(key)}</td><td class="metric-value {css_class}">{formatted_value}</td></tr>\n'

    html_content += """        </table>
"""

    # Add parameters if present
    if parameters:
        html_content += """
        <h2>Strategy Parameters</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
"""
        for key, value in parameters.items():
            html_content += f'            <tr><td>{key}</td><td class="metric-value">{value}</td></tr>\n'
        html_content += """        </table>
"""

    html_content += """    </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


def format_metric_name(key: str) -> str:
    """Format a metric key for display.

    Converts snake_case keys to Title Case with spaces.

    Args:
        key: Metric key in snake_case format.

    Returns:
        Formatted string suitable for display.

    Example:
        >>> format_metric_name("total_pnl")
        'Total PnL'
    """
    replacements = {
        "pnl": "PnL",
        "var": "VaR",
        "avg": "Average",
    }
    words = key.split("_")
    formatted_words = []
    for word in words:
        if word.lower() in replacements:
            formatted_words.append(replacements[word.lower()])
        else:
            formatted_words.append(word.capitalize())
    return " ".join(formatted_words)


def format_metric_value(key: str, value: Union[int, float, str]) -> str:
    """Format a metric value for display.

    Applies appropriate formatting based on the metric type (percentage,
    currency, ratio, count).

    Args:
        key: Metric key to determine formatting type.
        value: Raw metric value.

    Returns:
        Formatted string representation.

    Example:
        >>> format_metric_value("sharpe_ratio", 1.234567)
        '1.2346'
        >>> format_metric_value("win_rate", 0.65)
        '65.00%'
    """
    if isinstance(value, str):
        return value

    percentage_keys = ["win_rate", "trade_win_rate", "max_drawdown", "var_95", "daily_volatility"]
    ratio_keys = ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "profit_factor"]
    count_keys = ["total_trades", "trading_days", "positive_days", "negative_days",
                  "winning_trades", "losing_trades"]

    if key in percentage_keys:
        return f"{value * 100:.2f}%" if isinstance(value, float) else f"{value}%"
    elif key in ratio_keys:
        return f"{value:.4f}"
    elif key in count_keys:
        return f"{int(value):,}"
    elif "pnl" in key.lower():
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.4f}"
    else:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)


def get_value_class(key: str, value: Union[int, float, str]) -> str:
    """Determine CSS class for metric value coloring.

    Args:
        key: Metric key to determine value interpretation.
        value: Metric value.

    Returns:
        CSS class name: 'positive', 'negative', or empty string.
    """
    if isinstance(value, str):
        return ""

    positive_is_good = ["total_pnl", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
                        "win_rate", "trade_win_rate", "profit_factor", "avg_winning_trade",
                        "best_day", "positive_days", "winning_trades"]
    negative_is_good = ["max_drawdown", "var_95", "avg_losing_trade", "worst_day"]

    if key in positive_is_good:
        return "positive" if value > 0 else "negative" if value < 0 else ""
    elif key in negative_is_good:
        return "negative" if value > 0 else "positive" if value < 0 else ""

    return ""
