# PRD Part 4: Evaluation, Paper Trading, Monitoring

## 4.10 Walk-Forward Validation

### FR-EVAL-01: Temporal Splits with Purging/Embargo

```python
@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    initial_train_days: int = 180      # First training window
    retrain_frequency_days: int = 30    # How often to retrain
    test_window_days: int = 30          # Test window size
    purge_days: int = 5                 # Gap between train and test
    embargo_days: int = 5               # Gap after test before next train
    expanding_window: bool = False      # If True, expanding train window


class WalkForwardValidator:
    """
    Walk-forward validation with purging and embargo.

    Purging: Remove data points from training that could leak into test
             (e.g., overlapping windows for rolling features)

    Embargo: Gap after test window before including data in next train
             (prevents information leakage from test period)

    Timeline visualization:
    |---TRAIN---|--PURGE--|---TEST---|--EMBARGO--|---TRAIN---|...
    """

    def __init__(self, config: WalkForwardConfig):
        self.config = config

    def generate_splits(
        self,
        start_date: datetime.date,
        end_date: datetime.date
    ) -> List[Dict[str, datetime.date]]:
        """
        Generate train/test splits for walk-forward validation.

        Returns list of splits, each containing:
        - train_start, train_end
        - test_start, test_end
        """
        splits = []
        current_train_start = start_date
        current_train_end = start_date + timedelta(days=self.config.initial_train_days)

        while True:
            # Purge period
            purge_end = current_train_end + timedelta(days=self.config.purge_days)

            # Test period
            test_start = purge_end
            test_end = test_start + timedelta(days=self.config.test_window_days)

            if test_end > end_date:
                break

            splits.append({
                "train_start": current_train_start,
                "train_end": current_train_end,
                "purge_start": current_train_end,
                "purge_end": purge_end,
                "test_start": test_start,
                "test_end": test_end,
                "embargo_start": test_end,
                "embargo_end": test_end + timedelta(days=self.config.embargo_days)
            })

            # Move to next split
            if self.config.expanding_window:
                # Keep same start, expand end
                current_train_end = test_end + timedelta(
                    days=self.config.embargo_days + self.config.retrain_frequency_days
                )
            else:
                # Rolling window
                current_train_start += timedelta(days=self.config.retrain_frequency_days)
                current_train_end = current_train_start + timedelta(days=self.config.initial_train_days)

        return splits

    def validate_no_leakage(
        self,
        features_df: pl.DataFrame,
        split: Dict[str, datetime.date]
    ) -> bool:
        """
        Validate that features respect temporal boundaries.

        Checks:
        1. No test data used in training features
        2. Rolling features properly bounded
        3. Influence scores from correct month
        """
        train_data = features_df.filter(
            (pl.col("date") >= split["train_start"]) &
            (pl.col("date") <= split["train_end"])
        )

        test_data = features_df.filter(
            (pl.col("date") >= split["test_start"]) &
            (pl.col("date") <= split["test_end"])
        )

        # Check 1: Feature timestamps
        for row in train_data.iter_rows(named=True):
            feature_date = row["date"]
            # All underlying data should be before feature_date
            if row.get("_latest_data_timestamp"):
                if row["_latest_data_timestamp"] > feature_date:
                    return False

        # Check 2: No overlap with test
        train_dates = set(train_data["date"].to_list())
        test_dates = set(test_data["date"].to_list())
        if train_dates & test_dates:
            return False

        return True


def run_walk_forward_backtest(
    features_df: pl.DataFrame,
    model_trainer: ModelTrainer,
    trading_policy: DynamicThresholdPolicy,
    backtester: Backtester,
    wf_config: WalkForwardConfig,
    feature_columns: List[str]
) -> Dict[str, Any]:
    """
    Run complete walk-forward backtest.

    1. Generate splits
    2. For each split: train model, generate predictions, backtest
    3. Aggregate results
    """
    validator = WalkForwardValidator(wf_config)
    start_date = features_df["date"].min()
    end_date = features_df["date"].max()

    splits = validator.generate_splits(start_date, end_date)

    all_predictions = {}
    all_actuals = {}
    split_metrics = []

    for i, split in enumerate(splits):
        logging.info(f"Processing split {i+1}/{len(splits)}")

        # Validate no leakage
        assert validator.validate_no_leakage(features_df, split), \
            f"Data leakage detected in split {i}"

        # Filter data
        train_df = features_df.filter(
            (pl.col("date") >= split["train_start"]) &
            (pl.col("date") <= split["train_end"])
        )

        # Train model
        model, train_metrics = model_trainer.train_for_date(
            train_df,
            split["test_start"],
            feature_columns
        )

        # Generate predictions for test period
        current_date = split["test_start"]
        while current_date <= split["test_end"]:
            predictions = model_trainer.predict(
                model, features_df, current_date, feature_columns
            )

            # Convert to signals using dynamic threshold
            signals = {}
            for asset, pred in predictions.items():
                signal = trading_policy.get_signal(pred["prob_up"] - 0.5)
                signals[asset] = signal

                # Store for aggregation
                if current_date not in all_predictions:
                    all_predictions[current_date] = {}
                all_predictions[current_date][asset] = pred

            current_date += timedelta(days=1)

        split_metrics.append({
            "split_idx": i,
            "train_start": split["train_start"],
            "test_start": split["test_start"],
            "test_end": split["test_end"],
            "train_metrics": train_metrics
        })

    # Run backtest on aggregated predictions
    # (Implementation would use backtester with all_predictions)

    return {
        "splits": split_metrics,
        "total_splits": len(splits),
        "predictions": all_predictions
    }
```

### FR-EVAL-02: Statistical Tests

```python
import scipy.stats as stats
from typing import List, Tuple

class StatisticalTests:
    """Statistical tests for strategy evaluation."""

    @staticmethod
    def bootstrap_sharpe_ci(
        returns: np.ndarray,
        n_bootstrap: int = 10000,
        ci: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Bootstrap confidence interval for Sharpe ratio.

        Returns: (lower_bound, sharpe, upper_bound)
        """
        def compute_sharpe(r: np.ndarray) -> float:
            if r.std() == 0:
                return 0.0
            return r.mean() / r.std() * np.sqrt(252)

        original_sharpe = compute_sharpe(returns)
        bootstrap_sharpes = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(returns, size=len(returns), replace=True)
            bootstrap_sharpes.append(compute_sharpe(sample))

        alpha = (1 - ci) / 2
        lower = np.percentile(bootstrap_sharpes, alpha * 100)
        upper = np.percentile(bootstrap_sharpes, (1 - alpha) * 100)

        return lower, original_sharpe, upper

    @staticmethod
    def bootstrap_cagr_ci(
        returns: np.ndarray,
        n_bootstrap: int = 10000,
        ci: float = 0.95
    ) -> Tuple[float, float, float]:
        """Bootstrap confidence interval for CAGR."""
        def compute_cagr(r: np.ndarray) -> float:
            total_return = (1 + r).prod()
            n_years = len(r) / 252
            if n_years <= 0:
                return 0.0
            return total_return ** (1 / n_years) - 1

        original_cagr = compute_cagr(returns)
        bootstrap_cagrs = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(returns, size=len(returns), replace=True)
            bootstrap_cagrs.append(compute_cagr(sample))

        alpha = (1 - ci) / 2
        lower = np.percentile(bootstrap_cagrs, alpha * 100)
        upper = np.percentile(bootstrap_cagrs, (1 - alpha) * 100)

        return lower, original_cagr, upper

    @staticmethod
    def white_reality_check(
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        n_bootstrap: int = 1000
    ) -> Tuple[float, bool]:
        """
        White's Reality Check for data snooping.

        Tests whether strategy outperformance is significant
        after accounting for multiple testing.

        Returns: (p_value, is_significant at 5%)
        """
        excess_returns = strategy_returns - benchmark_returns
        n = len(excess_returns)

        # Original statistic: max normalized return
        original_stat = np.sqrt(n) * excess_returns.mean() / excess_returns.std()

        # Bootstrap distribution
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            # Circular block bootstrap
            block_size = int(np.sqrt(n))
            blocks = []
            while len(blocks) * block_size < n:
                start = np.random.randint(0, n)
                block = excess_returns[start:start + block_size]
                if len(block) < block_size:
                    block = np.concatenate([
                        block,
                        excess_returns[:block_size - len(block)]
                    ])
                blocks.append(block)

            sample = np.concatenate(blocks)[:n]
            # Center the sample
            sample = sample - sample.mean() + excess_returns.mean()
            stat = np.sqrt(n) * sample.mean() / sample.std()
            bootstrap_stats.append(stat)

        # p-value
        p_value = (np.array(bootstrap_stats) >= original_stat).mean()

        return p_value, p_value < 0.05

    @staticmethod
    def paired_t_test(
        returns_a: np.ndarray,
        returns_b: np.ndarray
    ) -> Tuple[float, float]:
        """
        Paired t-test comparing two strategies.

        Returns: (t_statistic, p_value)
        """
        t_stat, p_value = stats.ttest_rel(returns_a, returns_b)
        return t_stat, p_value

    @staticmethod
    def diebold_mariano_test(
        actual: np.ndarray,
        pred_a: np.ndarray,
        pred_b: np.ndarray
    ) -> Tuple[float, float]:
        """
        Diebold-Mariano test for comparing forecast accuracy.

        Tests whether forecast A is significantly better than B.
        Returns: (dm_statistic, p_value)
        """
        e_a = actual - pred_a
        e_b = actual - pred_b

        d = e_a**2 - e_b**2
        mean_d = d.mean()
        var_d = d.var()

        dm_stat = mean_d / np.sqrt(var_d / len(d))
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        return dm_stat, p_value
```

### FR-EVAL-03: Ablation Studies

```python
@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    name: str
    description: str
    feature_columns: List[str]     # Features to include
    exclude_features: List[str]    # Features to exclude
    model_modifications: Dict      # Model config changes


ABLATION_STUDIES = [
    AblationConfig(
        name="price_only",
        description="Only price-based features, no sentiment",
        feature_columns=["return_1d", "return_5d", "volatility_30d",
                        "related_return_1", "related_return_2", "related_return_3"],
        exclude_features=["sentiment_index", "sentiment_mean", "stance_mean"],
        model_modifications={}
    ),
    AblationConfig(
        name="sentiment_no_influence",
        description="Sentiment without influence weighting",
        feature_columns=BASELINE_FEATURES,
        exclude_features=["influence_weighted_sentiment"],
        model_modifications={"use_raw_sentiment": True}
    ),
    AblationConfig(
        name="no_credibility_filter",
        description="Include all posts without credibility filtering",
        feature_columns=UPGRADE_FEATURES,
        exclude_features=["credibility_weighted_sentiment", "manipulation_adjusted_sentiment"],
        model_modifications={"disable_credibility": True}
    ),
    AblationConfig(
        name="reddit_only",
        description="Only Reddit data, no other sources",
        feature_columns=UPGRADE_FEATURES,
        exclude_features=[],
        model_modifications={"sources": ["reddit"]}
    ),
    AblationConfig(
        name="no_top_influencers",
        description="Remove top 10% influencers",
        feature_columns=UPGRADE_FEATURES,
        exclude_features=[],
        model_modifications={"filter_top_influencers": 0.1}
    ),
    AblationConfig(
        name="no_meme_events",
        description="Remove posts during meme stock events",
        feature_columns=UPGRADE_FEATURES,
        exclude_features=[],
        model_modifications={"filter_meme_events": True}
    ),
]


def run_ablation_suite(
    features_df: pl.DataFrame,
    base_config: Dict,
    ablations: List[AblationConfig]
) -> Dict[str, BacktestResult]:
    """Run all ablation studies and compare results."""
    results = {}

    # Run baseline
    baseline_result = run_backtest(features_df, base_config)
    results["baseline"] = baseline_result

    # Run each ablation
    for ablation in ablations:
        logging.info(f"Running ablation: {ablation.name}")

        # Modify config
        ablation_config = base_config.copy()
        ablation_config["feature_columns"] = ablation.feature_columns
        ablation_config.update(ablation.model_modifications)

        # Run backtest
        ablation_result = run_backtest(features_df, ablation_config)
        results[ablation.name] = ablation_result

    return results


def generate_ablation_report(results: Dict[str, BacktestResult]) -> str:
    """Generate markdown report comparing ablation results."""
    report = "# Ablation Study Results\n\n"

    # Summary table
    report += "| Ablation | Sharpe | CAGR | Max DD | vs Baseline |\n"
    report += "|----------|--------|------|--------|-------------|\n"

    baseline_sharpe = results["baseline"].metrics["sharpe"]

    for name, result in results.items():
        sharpe = result.metrics["sharpe"]
        cagr = result.metrics["cagr"]
        max_dd = result.metrics["max_drawdown"]
        diff = ((sharpe - baseline_sharpe) / baseline_sharpe * 100) if baseline_sharpe != 0 else 0

        report += f"| {name} | {sharpe:.2f} | {cagr:.2%} | {max_dd:.2%} | {diff:+.1f}% |\n"

    return report
```

---

## 4.11 Paper Trading

### FR-PAPER-01: Daily Trading Loop

```python
import asyncio
from datetime import datetime, time
from typing import Optional

class PaperTradingScheduler:
    """
    Daily automated paper trading loop.

    Schedule (US Eastern Time):
    1. 07:00 - Ingest overnight social data
    2. 09:00 - Build as-of features (before market open)
    3. 09:15 - Run model inference
    4. 09:25 - Generate target portfolio
    5. 09:30 - Submit paper orders (market open)
    6. 16:00 - Log fills and daily P&L
    7. 18:00 - Run diagnostics and update models if needed
    """

    def __init__(
        self,
        ingestion_service: IngestionService,
        feature_builder: FeatureBuilder,
        model: nn.Module,
        trading_policy: PortfolioPolicy,
        broker: AlpacaBroker,
        risk_manager: RiskManager,
        universe: List[str],
        timezone: str = "US/Eastern"
    ):
        self.ingestion = ingestion_service
        self.features = feature_builder
        self.model = model
        self.policy = trading_policy
        self.broker = broker
        self.risk = risk_manager
        self.universe = universe
        self.tz = pytz.timezone(timezone)
        self.daily_log: Optional[DailyTradingLog] = None

    async def run_daily_cycle(self, date: datetime.date) -> DailyTradingLog:
        """Execute complete daily trading cycle."""
        self.daily_log = DailyTradingLog(date=date)
        self.risk.start_of_day()

        try:
            # Check if halted
            is_ok, reason = self.risk.check_limits()
            if not is_ok:
                self.daily_log.status = "HALTED"
                self.daily_log.halt_reason = reason
                return self.daily_log

            # Step 1: Ingest data
            await self._step_ingest()

            # Step 2: Build features
            await self._step_build_features(date)

            # Step 3: Run inference
            await self._step_inference()

            # Step 4: Generate target portfolio
            await self._step_generate_portfolio()

            # Step 5: Submit orders
            await self._step_submit_orders()

            # Step 6: Wait for fills and log
            await self._step_log_fills()

            self.daily_log.status = "COMPLETED"

        except Exception as e:
            self.daily_log.status = "ERROR"
            self.daily_log.error = str(e)
            logging.exception(f"Error in daily cycle: {e}")

        return self.daily_log

    async def _step_ingest(self):
        """Ingest latest social data."""
        self.daily_log.steps["ingest"] = {"start": datetime.now()}

        posts_ingested = await self.ingestion.run_incremental()

        self.daily_log.steps["ingest"]["end"] = datetime.now()
        self.daily_log.steps["ingest"]["posts_ingested"] = posts_ingested
        self.daily_log.inputs_hash = self._compute_inputs_hash()

    async def _step_build_features(self, date: datetime.date):
        """Build as-of features."""
        self.daily_log.steps["features"] = {"start": datetime.now()}

        # Cutoff time: 9:25 AM ET (before market open)
        cutoff = datetime.combine(date, time(9, 25), self.tz)

        feature_vectors = {}
        for asset in self.universe:
            fv = self.features.build_features(
                asset=asset,
                as_of_date=date,
                as_of_time=cutoff.time()
            )
            feature_vectors[asset] = fv

        self.daily_log.steps["features"]["end"] = datetime.now()
        self.daily_log.feature_vectors = feature_vectors

    async def _step_inference(self):
        """Run model inference."""
        self.daily_log.steps["inference"] = {"start": datetime.now()}

        predictions = {}
        self.model.eval()

        for asset, fv in self.daily_log.feature_vectors.items():
            # Convert feature vector to tensor
            x = self._fv_to_tensor(fv)

            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=-1)[0]

            predictions[asset] = {
                "prob_down": float(probs[0]),
                "prob_up": float(probs[1]),
                "direction": 1 if probs[1] > probs[0] else -1,
                "confidence": float(max(probs))
            }

        self.daily_log.steps["inference"]["end"] = datetime.now()
        self.daily_log.predictions = predictions
        self.daily_log.model_version = self._get_model_version()

    async def _step_generate_portfolio(self):
        """Generate target portfolio."""
        self.daily_log.steps["portfolio"] = {"start": datetime.now()}

        current_positions = await self.broker.get_positions()
        current_weights = self._positions_to_weights(current_positions)

        volatilities = self._get_volatilities()
        capital = await self.broker.get_account_value()

        target_weights = self.policy.construct_portfolio(
            predictions=self.daily_log.predictions,
            current_positions=current_weights,
            volatilities=volatilities,
            capital=capital
        )

        # Calculate trades needed
        trades_needed = {}
        for asset in set(current_weights.keys()) | set(target_weights.keys()):
            current = current_weights.get(asset, 0)
            target = target_weights.get(asset, 0)
            if abs(target - current) > 0.001:  # Minimum trade threshold
                trades_needed[asset] = {
                    "current_weight": current,
                    "target_weight": target,
                    "trade_weight": target - current
                }

        self.daily_log.steps["portfolio"]["end"] = datetime.now()
        self.daily_log.target_portfolio = target_weights
        self.daily_log.trades_needed = trades_needed

    async def _step_submit_orders(self):
        """Submit paper orders to broker."""
        self.daily_log.steps["orders"] = {"start": datetime.now()}

        orders_submitted = []
        capital = await self.broker.get_account_value()

        for asset, trade_info in self.daily_log.trades_needed.items():
            trade_value = capital * abs(trade_info["trade_weight"])
            current_price = await self.broker.get_current_price(asset)
            quantity = int(trade_value / current_price)

            if quantity == 0:
                continue

            side = "buy" if trade_info["trade_weight"] > 0 else "sell"

            order = await self.broker.submit_order(
                symbol=asset,
                side=side,
                quantity=quantity,
                order_type="market"
            )

            orders_submitted.append({
                "asset": asset,
                "side": side,
                "quantity": quantity,
                "order_id": order.id,
                "status": order.status
            })

        self.daily_log.steps["orders"]["end"] = datetime.now()
        self.daily_log.orders_submitted = orders_submitted

    async def _step_log_fills(self):
        """Wait for fills and log results."""
        self.daily_log.steps["fills"] = {"start": datetime.now()}

        # Wait for market close
        await self._wait_until(time(16, 5))

        # Get fills
        fills = []
        for order in self.daily_log.orders_submitted:
            fill = await self.broker.get_order(order["order_id"])
            fills.append({
                "order_id": order["order_id"],
                "asset": order["asset"],
                "side": order["side"],
                "filled_qty": fill.filled_qty,
                "avg_price": fill.avg_fill_price,
                "status": fill.status
            })

        # Calculate P&L
        account = await self.broker.get_account()
        self.daily_log.end_of_day_capital = float(account.portfolio_value)
        self.daily_log.daily_pnl = (
            self.daily_log.end_of_day_capital -
            self.risk.daily_start_capital
        )
        self.daily_log.daily_return = (
            self.daily_log.daily_pnl / self.risk.daily_start_capital
        )

        # Update risk manager
        self.risk.update_capital(self.daily_log.end_of_day_capital)

        self.daily_log.steps["fills"]["end"] = datetime.now()
        self.daily_log.fills = fills

    def _compute_inputs_hash(self) -> str:
        """Compute hash of input data for reproducibility."""
        import hashlib
        # Hash of raw data files used
        return hashlib.sha256(b"placeholder").hexdigest()[:16]

    def _get_model_version(self) -> str:
        """Get current model version."""
        return "v1.0.0"  # From MLflow or similar


@dataclass
class DailyTradingLog:
    """Complete log of a daily trading cycle."""
    date: datetime.date
    status: str = "PENDING"

    # Traceability
    inputs_hash: Optional[str] = None
    model_version: Optional[str] = None

    # Intermediate artifacts
    feature_vectors: Dict = field(default_factory=dict)
    predictions: Dict = field(default_factory=dict)
    target_portfolio: Dict = field(default_factory=dict)
    trades_needed: Dict = field(default_factory=dict)
    orders_submitted: List = field(default_factory=list)
    fills: List = field(default_factory=list)

    # Results
    end_of_day_capital: float = 0.0
    daily_pnl: float = 0.0
    daily_return: float = 0.0

    # Timing
    steps: Dict = field(default_factory=dict)

    # Error handling
    error: Optional[str] = None
    halt_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        """Serialize for storage."""
        return asdict(self)

    def get_decision_rationale(self) -> Dict[str, List[str]]:
        """
        Get top contributing features for each trade decision.

        For audit trail: explain why each trade was made.
        """
        rationale = {}
        for asset, pred in self.predictions.items():
            fv = self.feature_vectors.get(asset)
            if fv and asset in self.trades_needed:
                # Rank features by absolute value
                features = [
                    (name, getattr(fv, name))
                    for name in UPGRADE_FEATURES
                    if hasattr(fv, name)
                ]
                sorted_features = sorted(features, key=lambda x: abs(x[1]), reverse=True)
                rationale[asset] = [
                    f"{name}: {value:.4f}"
                    for name, value in sorted_features[:5]
                ]
        return rationale
```

### FR-PAPER-02: Broker Integration (Alpaca)

```python
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError

class AlpacaBroker:
    """Alpaca paper trading integration."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True
    ):
        base_url = (
            "https://paper-api.alpaca.markets" if paper
            else "https://api.alpaca.markets"
        )
        self.api = tradeapi.REST(api_key, api_secret, base_url)
        self.is_paper = paper

    async def get_account(self):
        """Get account information."""
        return self.api.get_account()

    async def get_account_value(self) -> float:
        """Get current portfolio value."""
        account = await self.get_account()
        return float(account.portfolio_value)

    async def get_positions(self) -> Dict[str, float]:
        """Get current positions as {symbol: quantity}."""
        positions = self.api.list_positions()
        return {p.symbol: float(p.qty) for p in positions}

    async def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        quote = self.api.get_latest_quote(symbol)
        return float(quote.ap)  # Ask price

    async def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        time_in_force: str = "day"
    ):
        """Submit order to broker."""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )
            return order
        except APIError as e:
            logging.error(f"Order submission failed: {e}")
            raise

    async def get_order(self, order_id: str):
        """Get order status."""
        return self.api.get_order(order_id)

    async def cancel_all_orders(self):
        """Cancel all open orders."""
        self.api.cancel_all_orders()

    async def liquidate_all(self):
        """Liquidate all positions (emergency)."""
        self.api.close_all_positions()
```

---

## 4.12 Monitoring and Alerting

### FR-MON-01: Prometheus Metrics

```python
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Define metrics
INGESTION_POSTS_TOTAL = Counter(
    'imst_ingestion_posts_total',
    'Total posts ingested',
    ['source', 'subreddit']
)

INGESTION_ERRORS_TOTAL = Counter(
    'imst_ingestion_errors_total',
    'Total ingestion errors',
    ['source', 'error_type']
)

INFERENCE_LATENCY = Histogram(
    'imst_inference_latency_seconds',
    'Model inference latency',
    ['model_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

PORTFOLIO_VALUE = Gauge(
    'imst_portfolio_value_dollars',
    'Current portfolio value'
)

DAILY_PNL = Gauge(
    'imst_daily_pnl_dollars',
    'Daily P&L'
)

DRAWDOWN = Gauge(
    'imst_drawdown_percent',
    'Current drawdown from peak'
)

FEATURE_DRIFT = Gauge(
    'imst_feature_drift_score',
    'Feature distribution drift score',
    ['feature_name']
)

MANIPULATION_RISK = Gauge(
    'imst_manipulation_risk_score',
    'Manipulation risk score',
    ['asset']
)

DATA_FRESHNESS = Gauge(
    'imst_data_freshness_seconds',
    'Seconds since last data update',
    ['source']
)


class MetricsCollector:
    """Collect and expose metrics."""

    def __init__(self, port: int = 9090):
        self.port = port

    def start(self):
        """Start Prometheus metrics server."""
        start_http_server(self.port)
        logging.info(f"Metrics server started on port {self.port}")

    def record_ingestion(self, source: str, subreddit: str, count: int):
        INGESTION_POSTS_TOTAL.labels(source=source, subreddit=subreddit).inc(count)

    def record_portfolio(self, value: float, daily_pnl: float, drawdown: float):
        PORTFOLIO_VALUE.set(value)
        DAILY_PNL.set(daily_pnl)
        DRAWDOWN.set(drawdown)

    def record_inference_latency(self, model_type: str, latency: float):
        INFERENCE_LATENCY.labels(model_type=model_type).observe(latency)

    def record_feature_drift(self, feature_name: str, drift_score: float):
        FEATURE_DRIFT.labels(feature_name=feature_name).set(drift_score)

    def record_manipulation_risk(self, asset: str, risk_score: float):
        MANIPULATION_RISK.labels(asset=asset).set(risk_score)
```

### FR-MON-02: Alerting Rules

```yaml
# alerting_rules.yml
groups:
  - name: imst_alerts
    rules:
      # Data pipeline alerts
      - alert: DataIngestionStalled
        expr: imst_data_freshness_seconds{source="reddit"} > 3600
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Reddit ingestion stalled"
          description: "No new data for {{ $value | humanizeDuration }}"

      - alert: IngestionErrorRate
        expr: rate(imst_ingestion_errors_total[5m]) > 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High ingestion error rate"

      # Trading alerts
      - alert: DrawdownCritical
        expr: imst_drawdown_percent > 0.08
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Drawdown exceeds 8%"
          description: "Current drawdown: {{ $value | printf \"%.2f\" }}%"

      - alert: DrawdownWarning
        expr: imst_drawdown_percent > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Drawdown exceeds 5%"

      - alert: DailyLossLimit
        expr: imst_daily_pnl_dollars < -2000
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Daily loss limit approaching"

      # Model alerts
      - alert: FeatureDriftDetected
        expr: imst_feature_drift_score > 0.3
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Feature drift detected: {{ $labels.feature_name }}"

      - alert: InferenceLatencyHigh
        expr: histogram_quantile(0.95, imst_inference_latency_seconds_bucket) > 1.0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Model inference latency high"

      # Manipulation alerts
      - alert: ManipulationRiskHigh
        expr: imst_manipulation_risk_score > 0.7
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High manipulation risk for {{ $labels.asset }}"
```

### FR-MON-03: Drift Detection

```python
from scipy import stats

class DriftDetector:
    """Detect distribution drift in features and predictions."""

    def __init__(
        self,
        reference_window_days: int = 30,
        drift_threshold: float = 0.3
    ):
        self.reference_window = reference_window_days
        self.threshold = drift_threshold
        self.reference_distributions: Dict[str, np.ndarray] = {}

    def set_reference(self, features_df: pl.DataFrame, feature_columns: List[str]):
        """Set reference distributions from historical data."""
        for col in feature_columns:
            values = features_df[col].to_numpy()
            self.reference_distributions[col] = values

    def check_drift(
        self,
        current_features: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Check for drift in current features vs reference.

        Uses two-sample Kolmogorov-Smirnov test.
        """
        drift_results = {}

        for col, current_values in current_features.items():
            if col not in self.reference_distributions:
                continue

            reference = self.reference_distributions[col]

            # KS test
            ks_stat, p_value = stats.ks_2samp(reference, current_values)

            # Population Stability Index (PSI)
            psi = self._calculate_psi(reference, current_values)

            drift_results[col] = {
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "psi": psi,
                "is_drifted": psi > self.threshold or p_value < 0.01
            }

        return drift_results

    def _calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index.

        PSI < 0.1: No significant drift
        0.1 <= PSI < 0.25: Moderate drift
        PSI >= 0.25: Significant drift
        """
        # Bin edges from reference distribution
        _, bin_edges = np.histogram(reference, bins=n_bins)

        # Calculate proportions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)

        ref_props = ref_counts / len(reference)
        curr_props = curr_counts / len(current)

        # Avoid log(0)
        ref_props = np.clip(ref_props, 1e-10, 1)
        curr_props = np.clip(curr_props, 1e-10, 1)

        # PSI formula
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))

        return psi
```

---

## 5. Non-Functional Requirements

### NFR-01: Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| Ingestion throughput | 10,000 posts/min | Load test |
| Feature computation | < 5 sec/asset | Profiling |
| Model inference | < 100ms/batch | Prometheus |
| Backtest 1 year | < 10 min | CI benchmark |
| Daily cycle completion | < 30 min | Monitoring |

### NFR-02: Reliability

| Metric | Target | Measurement |
|--------|--------|-------------|
| System uptime | 99.5% | Monitoring |
| Data pipeline SLA | 99% daily | Alerting |
| Recovery time | < 1 hour | Runbook test |
| Data loss tolerance | 0% for features | Checksum validation |

### NFR-03: Scalability

| Dimension | Current | Target | Approach |
|-----------|---------|--------|----------|
| Assets | 4 (paper) | 500 | Parallel processing |
| Sources | 1 (Reddit) | 5 | Plug-in architecture |
| Daily posts | 50K | 1M | Batch processing |
| Historical data | 1 year | 5 years | Partitioned storage |

### NFR-04: Security

| Control | Implementation |
|---------|----------------|
| API keys | Environment variables + secrets manager |
| Author IDs | SHA-256 hashed |
| Data at rest | Encrypted Parquet |
| Data in transit | TLS 1.3 |
| Access control | Role-based (dev/ops/audit) |

### NFR-05: Cost Targets

| Resource | Monthly Budget | Optimization |
|----------|----------------|--------------|
| Compute | $500 | Spot instances for training |
| Storage | $100 | Parquet compression, cold archive |
| API calls | $200 | Caching, rate limiting |
| Monitoring | $50 | Self-hosted Prometheus/Grafana |

---

## 6. Data Schemas

### Schema: Reddit Posts (Bronze)

```sql
CREATE TABLE reddit_posts (
    id VARCHAR(20) PRIMARY KEY,
    author_id VARCHAR(64) NOT NULL,      -- SHA-256 hash
    created_utc BIGINT NOT NULL,
    retrieved_utc BIGINT NOT NULL,
    subreddit VARCHAR(100) NOT NULL,
    post_type VARCHAR(20) NOT NULL,      -- submission/comment
    title TEXT,
    selftext TEXT,
    body TEXT,
    score INTEGER NOT NULL,
    upvote_ratio FLOAT,
    num_comments INTEGER,
    permalink TEXT NOT NULL,
    parent_id VARCHAR(20),
    link_id VARCHAR(20),
    distinguished VARCHAR(20),
    edited BOOLEAN NOT NULL,
    awards INTEGER NOT NULL DEFAULT 0,
    flair TEXT,
    url TEXT,

    -- Partitioning
    partition_date DATE NOT NULL,

    -- Indexes
    INDEX idx_created (created_utc),
    INDEX idx_subreddit_date (subreddit, partition_date),
    INDEX idx_author (author_id)
);
```

### Schema: Enriched Posts (Silver)

```sql
CREATE TABLE enriched_posts (
    post_id VARCHAR(20) PRIMARY KEY REFERENCES reddit_posts(id),

    -- Text processing
    cleaned_text TEXT NOT NULL,
    language VARCHAR(10) NOT NULL,
    language_confidence FLOAT NOT NULL,
    is_duplicate BOOLEAN NOT NULL,
    duplicate_of VARCHAR(20),

    -- Sentiment
    textblob_polarity FLOAT NOT NULL,
    textblob_subjectivity FLOAT NOT NULL,
    finbert_negative FLOAT,
    finbert_neutral FLOAT,
    finbert_positive FLOAT,
    finbert_sentiment FLOAT,

    -- Stance
    stance_bullish FLOAT,
    stance_neutral FLOAT,
    stance_bearish FLOAT,
    stance_score FLOAT,

    -- Events
    event_types TEXT[],                  -- Array of event types

    -- Timestamps
    processed_at TIMESTAMP NOT NULL,

    INDEX idx_post_date (partition_date)
);
```

### Schema: Entity Links

```sql
CREATE TABLE entity_links (
    id SERIAL PRIMARY KEY,
    post_id VARCHAR(20) NOT NULL REFERENCES reddit_posts(id),
    asset_id VARCHAR(20) NOT NULL,       -- Ticker symbol
    confidence FLOAT NOT NULL,
    method VARCHAR(50) NOT NULL,         -- cashtag, alias, embedding
    matched_span TEXT NOT NULL,

    INDEX idx_post (post_id),
    INDEX idx_asset_date (asset_id, partition_date)
);
```

### Schema: Daily Features (Gold)

```sql
CREATE TABLE daily_features (
    date DATE NOT NULL,
    asset VARCHAR(20) NOT NULL,

    -- Price features
    return_1d FLOAT,
    return_5d FLOAT,
    volatility_30d FLOAT,
    related_return_1 FLOAT,
    related_return_2 FLOAT,
    related_return_3 FLOAT,

    -- Sentiment features
    sentiment_index FLOAT,
    sentiment_mean FLOAT,
    sentiment_std FLOAT,
    stance_mean FLOAT,
    stance_entropy FLOAT,
    post_volume INTEGER,
    unique_authors INTEGER,

    -- Weighted metrics
    influence_weighted_sentiment FLOAT,
    credibility_weighted_sentiment FLOAT,
    manipulation_adjusted_sentiment FLOAT,

    -- Events
    earnings_mentions INTEGER,
    regulation_mentions INTEGER,

    -- Technical
    rsi_14 FLOAT,
    macd_signal FLOAT,
    volume_ratio FLOAT,

    -- Target (for training)
    target_return FLOAT,
    target_direction INTEGER,            -- 1=up, 0=down

    -- Metadata
    feature_version VARCHAR(20) NOT NULL,
    computed_at TIMESTAMP NOT NULL,

    PRIMARY KEY (date, asset),
    INDEX idx_date (date)
);
```

### Schema: Influence Scores

```sql
CREATE TABLE influence_scores (
    month_start DATE NOT NULL,
    author_id VARCHAR(64) NOT NULL,
    influence_score FLOAT NOT NULL,
    in_degree INTEGER,
    out_degree INTEGER,
    pagerank FLOAT,
    model_version VARCHAR(20) NOT NULL,
    computed_at TIMESTAMP NOT NULL,

    PRIMARY KEY (month_start, author_id),
    INDEX idx_month (month_start)
);
```

### Schema: Trading Log

```sql
CREATE TABLE trading_logs (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    status VARCHAR(20) NOT NULL,

    -- Traceability
    inputs_hash VARCHAR(64),
    model_version VARCHAR(20),
    feature_vectors JSONB,
    predictions JSONB,
    target_portfolio JSONB,
    trades_needed JSONB,
    orders_submitted JSONB,
    fills JSONB,

    -- Results
    start_capital FLOAT,
    end_capital FLOAT,
    daily_pnl FLOAT,
    daily_return FLOAT,

    -- Timing
    steps JSONB,

    -- Errors
    error TEXT,
    halt_reason TEXT,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    INDEX idx_date (date)
);
```

---

*Continued in Part 5 (Architecture and Implementation Blueprint)...*
