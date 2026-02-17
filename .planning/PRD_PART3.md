# PRD Part 3: Forecasting, Trading, and Backtesting

## 4.7 Forecasting Models

### FR-FORE-01: Feature Engineering

```python
@dataclass
class DailyFeatureVector:
    """
    Features for daily return prediction.

    Paper baseline features:
    - Daily return (previous day)
    - 30-day rolling volatility
    - Returns of 3 correlated stocks
    - Sentiment index

    Upgrade features:
    - All sentiment aggregates
    - Influence-weighted metrics
    - Event indicators
    """

    # Identifiers
    date: datetime.date
    asset: str

    # Price features (lagged)
    return_1d: float              # Previous day return
    return_5d: float              # 5-day cumulative return
    volatility_30d: float         # 30-day rolling std of returns

    # Related stocks (paper replication)
    related_return_1: float       # Return of correlated stock 1
    related_return_2: float       # Return of correlated stock 2
    related_return_3: float       # Return of correlated stock 3

    # Sentiment (baseline)
    sentiment_index: float        # TextBlob polarity * influence weighted

    # Sentiment (upgrade)
    sentiment_mean: float
    sentiment_std: float
    stance_mean: float
    stance_entropy: float
    post_volume: int
    unique_authors: int

    # Weighted sentiment (upgrade)
    influence_weighted_sentiment: float
    credibility_weighted_sentiment: float
    manipulation_adjusted_sentiment: float

    # Events (upgrade)
    earnings_mentions: int
    regulation_mentions: int

    # Technical (upgrade)
    rsi_14: float
    macd_signal: float
    volume_ratio: float           # Today vs 20-day avg


# Related stocks mapping (paper replication)
RELATED_STOCKS = {
    "AAPL": ["MSFT", "GOOGL", "META"],     # Tech peers
    "JNJ": ["PFE", "MRK", "UNH"],          # Healthcare peers
    "JPM": ["BAC", "GS", "MS"],            # Financials peers
    "XOM": ["CVX", "COP", "SLB"],          # Energy peers
}


class FeatureBuilder:
    """Build feature vectors with strict as-of semantics."""

    def __init__(
        self,
        sentiment_store: SentimentStore,
        market_store: MarketStore,
        influence_store: InfluenceStore
    ):
        self.sentiment = sentiment_store
        self.market = market_store
        self.influence = influence_store

    def build_features(
        self,
        asset: str,
        as_of_date: datetime.date,
        as_of_time: datetime.time = time(15, 50),  # Before market close
        window_size: int = 30
    ) -> DailyFeatureVector:
        """
        Build feature vector strictly as-of given timestamp.

        CRITICAL: All features use data available BEFORE as_of_time.
        """
        cutoff = datetime.combine(as_of_date, as_of_time)

        # Price features (use PREVIOUS day close, not today)
        prev_date = as_of_date - timedelta(days=1)
        prices = self.market.get_prices(asset, prev_date - timedelta(days=window_size), prev_date)

        returns = prices['close'].pct_change().dropna()
        return_1d = returns.iloc[-1] if len(returns) > 0 else 0.0
        return_5d = (prices['close'].iloc[-1] / prices['close'].iloc[-5] - 1) if len(prices) >= 5 else 0.0
        volatility_30d = returns.std() if len(returns) >= 5 else 0.0

        # Related stock returns
        related = RELATED_STOCKS.get(asset, [])
        related_returns = []
        for rel_asset in related[:3]:
            rel_prices = self.market.get_prices(rel_asset, prev_date - timedelta(days=1), prev_date)
            if len(rel_prices) >= 2:
                related_returns.append(rel_prices['close'].pct_change().iloc[-1])
            else:
                related_returns.append(0.0)
        while len(related_returns) < 3:
            related_returns.append(0.0)

        # Sentiment features (posts BEFORE cutoff only)
        sentiment = self.sentiment.get_aggregates(asset, as_of_date, cutoff)
        influence_scores = self.influence.get_monthly_scores(
            self._get_month_start(as_of_date)
        )

        # Compute sentiment index (paper formula)
        sentiment_index = self._compute_sentiment_index(
            asset, as_of_date, cutoff, influence_scores
        )

        return DailyFeatureVector(
            date=as_of_date,
            asset=asset,
            return_1d=return_1d,
            return_5d=return_5d,
            volatility_30d=volatility_30d,
            related_return_1=related_returns[0],
            related_return_2=related_returns[1],
            related_return_3=related_returns[2],
            sentiment_index=sentiment_index,
            sentiment_mean=sentiment.sentiment_mean,
            sentiment_std=sentiment.sentiment_std,
            stance_mean=sentiment.stance_mean,
            stance_entropy=sentiment.stance_entropy,
            post_volume=sentiment.post_count,
            unique_authors=sentiment.unique_authors,
            influence_weighted_sentiment=sentiment.influence_weighted_sentiment,
            credibility_weighted_sentiment=sentiment.credibility_weighted_sentiment,
            manipulation_adjusted_sentiment=sentiment.manipulation_adjusted_sentiment,
            earnings_mentions=sentiment.earnings_mentions,
            regulation_mentions=sentiment.regulation_mentions,
            rsi_14=self._compute_rsi(prices, 14),
            macd_signal=self._compute_macd_signal(prices),
            volume_ratio=self._compute_volume_ratio(prices)
        )

    def _compute_sentiment_index(
        self,
        asset: str,
        date: datetime.date,
        cutoff: datetime,
        influence_scores: Dict[str, float]
    ) -> float:
        """
        Compute paper-style sentiment index.

        Formula: sum(polarity * influence) / sum(influence)
        """
        posts = self.sentiment.get_raw_posts(asset, date, cutoff)

        if not posts:
            return 0.0

        weighted_sum = 0.0
        weight_sum = 0.0

        for post in posts:
            influence = influence_scores.get(post.author_id, 1.0)
            weighted_sum += post.polarity * influence
            weight_sum += influence

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0

    def _get_month_start(self, date: datetime.date) -> datetime.date:
        return date.replace(day=1)

    def _compute_rsi(self, prices: pd.DataFrame, period: int) -> float:
        delta = prices['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 else 50.0

    def _compute_macd_signal(self, prices: pd.DataFrame) -> float:
        exp1 = prices['close'].ewm(span=12, adjust=False).mean()
        exp2 = prices['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return (macd - signal).iloc[-1] if len(macd) > 0 else 0.0

    def _compute_volume_ratio(self, prices: pd.DataFrame) -> float:
        avg_vol = prices['volume'].rolling(20).mean()
        return (prices['volume'] / avg_vol).iloc[-1] if len(prices) > 0 else 1.0
```

### FR-FORE-02: Sequence Preparation

```python
class SequenceDataset(torch.utils.data.Dataset):
    """
    Prepare sequences for LSTM/CNN/Transformer models.

    Paper uses window sizes: 5, 15, 30 days
    """

    def __init__(
        self,
        features_df: pl.DataFrame,
        window_size: int,
        feature_columns: List[str],
        target_column: str = "target_direction"
    ):
        self.window_size = window_size
        self.feature_columns = feature_columns
        self.target_column = target_column

        # Sort by date and asset
        self.df = features_df.sort(["asset", "date"])

        # Build sequences per asset
        self.sequences = []
        self.targets = []

        for asset in self.df["asset"].unique():
            asset_df = self.df.filter(pl.col("asset") == asset)
            values = asset_df.select(feature_columns).to_numpy()
            targets = asset_df[target_column].to_numpy()

            for i in range(window_size, len(values)):
                seq = values[i - window_size:i]
                target = targets[i]
                self.sequences.append(seq)
                self.targets.append(target)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )


# Feature columns for baseline (paper replication)
BASELINE_FEATURES = [
    "return_1d",
    "volatility_30d",
    "related_return_1",
    "related_return_2",
    "related_return_3",
    "sentiment_index"
]

# Feature columns for upgrade
UPGRADE_FEATURES = BASELINE_FEATURES + [
    "sentiment_mean",
    "sentiment_std",
    "stance_mean",
    "stance_entropy",
    "post_volume",
    "unique_authors",
    "influence_weighted_sentiment",
    "credibility_weighted_sentiment",
    "manipulation_adjusted_sentiment",
    "rsi_14",
    "macd_signal",
    "volume_ratio"
]
```

### FR-FORE-03: Model Architectures (Paper Replication)

```python
class LSTMClassifier(nn.Module):
    """
    LSTM for return direction classification.

    Paper architecture (interpreted):
    - 2 LSTM layers
    - Hidden size: 64
    - Dropout: 0.3
    - Binary classification head
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        out = self.dropout(h_n[-1])
        return self.fc(out)


class CNNClassifier(nn.Module):
    """
    1D CNN for return direction classification.

    Paper architecture (interpreted):
    - Conv layers with kernel sizes 3, 5, 7
    - Max pooling
    - Dense layer for classification
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        num_filters: int = 32,
        kernel_sizes: List[int] = [3, 5, 7],
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, num_filters, k, padding=k//2)
            for k in kernel_sizes
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pooled = self.pool(conv_out).squeeze(-1)
            conv_outputs.append(pooled)

        out = torch.cat(conv_outputs, dim=1)
        out = self.dropout(out)
        return self.fc(out)


class TransformerClassifier(nn.Module):
    """
    Transformer for return direction classification.

    Paper uses transformer (architecture details not specified).
    Using standard encoder-only architecture.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        # Mean pooling over sequence
        x = x.mean(dim=1)
        return self.fc(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

### FR-FORE-04: Training Pipeline

```python
class ModelTrainer:
    """
    Training pipeline with daily retraining (paper replication).

    Paper specifies:
    - Daily retrain on rolling 90-day window
    - Train/val split within window
    """

    def __init__(
        self,
        model_type: str,  # "lstm", "cnn", "transformer"
        window_size: int = 30,
        training_days: int = 90,
        val_ratio: float = 0.2,
        lr: float = 0.001,
        epochs: int = 50,
        early_stopping_patience: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_type = model_type
        self.window_size = window_size
        self.training_days = training_days
        self.val_ratio = val_ratio
        self.lr = lr
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = device

    def create_model(self, input_dim: int) -> nn.Module:
        """Create model based on type."""
        if self.model_type == "lstm":
            return LSTMClassifier(input_dim=input_dim)
        elif self.model_type == "cnn":
            return CNNClassifier(input_dim=input_dim, seq_len=self.window_size)
        elif self.model_type == "transformer":
            return TransformerClassifier(input_dim=input_dim)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train_for_date(
        self,
        features_df: pl.DataFrame,
        target_date: datetime.date,
        feature_columns: List[str]
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """
        Train model for predictions on target_date.

        Uses data from [target_date - training_days, target_date - 1]
        """
        # Filter training window
        start_date = target_date - timedelta(days=self.training_days)
        end_date = target_date - timedelta(days=1)

        train_df = features_df.filter(
            (pl.col("date") >= start_date) &
            (pl.col("date") <= end_date)
        )

        if len(train_df) < self.window_size * 2:
            raise ValueError(f"Insufficient training data: {len(train_df)} rows")

        # Create dataset
        dataset = SequenceDataset(
            train_df,
            window_size=self.window_size,
            feature_columns=feature_columns
        )

        # Train/val split (by time, not random)
        split_idx = int(len(dataset) * (1 - self.val_ratio))
        train_dataset = torch.utils.data.Subset(dataset, range(split_idx))
        val_dataset = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Create and train model
        model = self.create_model(input_dim=len(feature_columns)).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.epochs):
            # Training
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_correct += (predicted == y).sum().item()
                    val_total += y.size(0)

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total if val_total > 0 else 0

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        metrics = {
            "final_val_loss": best_val_loss,
            "final_val_accuracy": val_acc,
            "epochs_trained": epoch + 1
        }

        return model, metrics

    def predict(
        self,
        model: nn.Module,
        features_df: pl.DataFrame,
        target_date: datetime.date,
        feature_columns: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate predictions for target_date.

        Returns dict mapping asset -> {prob_up, prob_down, direction}
        """
        model.eval()
        predictions = {}

        for asset in features_df["asset"].unique():
            # Get sequence ending on target_date - 1
            asset_df = features_df.filter(
                (pl.col("asset") == asset) &
                (pl.col("date") < target_date)
            ).sort("date").tail(self.window_size)

            if len(asset_df) < self.window_size:
                continue

            x = torch.tensor(
                asset_df.select(feature_columns).to_numpy(),
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)[0]

            predictions[asset] = {
                "prob_down": float(probs[0]),
                "prob_up": float(probs[1]),
                "direction": 1 if probs[1] > probs[0] else -1,
                "confidence": float(max(probs))
            }

        return predictions
```

---

## 4.8 Trading Policies

### FR-TRAD-01: Fixed Threshold (Paper Replication - LEAKAGE DEMO)

```python
class FixedThresholdPolicy:
    """
    Paper's fixed threshold method.

    WARNING: This method uses future data to select the optimal threshold.
    Implement ONLY for paper replication and comparison.
    Mark all results as "LEAKAGE DEMO - NOT VALID FOR REAL USE"
    """

    def __init__(self, thresholds_to_search: List[float] = None):
        self.thresholds = thresholds_to_search or [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        ]
        self.optimal_threshold = None
        self._is_leakage = True

    def fit(
        self,
        predictions: Dict[datetime.date, Dict[str, float]],  # date -> asset -> prediction
        actual_returns: Dict[datetime.date, Dict[str, float]]  # date -> asset -> return
    ) -> float:
        """
        Find optimal threshold over ENTIRE period.

        THIS IS LOOKAHEAD BIAS - we are using future returns to choose threshold.
        """
        best_return = float('-inf')
        best_threshold = 0.5

        for threshold in self.thresholds:
            total_return = 0.0
            for date in predictions:
                if date not in actual_returns:
                    continue
                for asset, pred in predictions[date].items():
                    if asset not in actual_returns[date]:
                        continue
                    actual = actual_returns[date][asset]
                    if pred > threshold:
                        total_return += actual
                    elif pred < -threshold:
                        total_return -= actual

            if total_return > best_return:
                best_return = total_return
                best_threshold = threshold

        self.optimal_threshold = best_threshold
        return best_threshold

    def get_signal(self, prediction: float) -> int:
        """
        Get trading signal based on fixed threshold.

        Returns: 1 (long), -1 (short), 0 (no trade)
        """
        if self.optimal_threshold is None:
            raise ValueError("Must call fit() first")

        if prediction > self.optimal_threshold:
            return 1
        elif prediction < -self.optimal_threshold:
            return -1
        return 0
```

### FR-TRAD-02: Dynamic Threshold (Paper Replication)

```python
class DynamicThresholdPolicy:
    """
    Paper's dynamic threshold method (incremental learning).

    Updates threshold each day based on historical predictions and returns.
    NO LOOKAHEAD - only uses data up to current day.

    Paper formula (interpreted):
    T_t = |D_t| - sqrt(2 * ln(2)) * v_t

    Where:
    - D_t = cumulative prediction error
    - v_t = volatility adjustment
    """

    def __init__(
        self,
        initial_threshold: float = 0.5,
        learning_rate: float = 0.1,
        volatility_window: int = 20,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9
    ):
        self.initial_threshold = initial_threshold
        self.threshold = initial_threshold
        self.learning_rate = learning_rate
        self.volatility_window = volatility_window
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        # History for incremental updates
        self.prediction_history: List[float] = []
        self.return_history: List[float] = []
        self.error_history: List[float] = []

    def update(self, prediction: float, actual_return: float) -> None:
        """
        Update threshold based on today's prediction and return.

        Called at end of each day AFTER observing actual return.
        """
        self.prediction_history.append(prediction)
        self.return_history.append(actual_return)

        # Calculate prediction error
        predicted_direction = 1 if prediction > 0 else -1
        actual_direction = 1 if actual_return > 0 else -1
        error = abs(predicted_direction - actual_direction) / 2
        self.error_history.append(error)

        # Calculate cumulative error (D_t)
        cumulative_error = sum(self.error_history)

        # Calculate volatility adjustment (v_t)
        if len(self.return_history) >= self.volatility_window:
            recent_returns = self.return_history[-self.volatility_window:]
            volatility = np.std(recent_returns)
        else:
            volatility = np.std(self.return_history) if self.return_history else 0.01

        # Update threshold using paper-like formula
        sqrt_term = np.sqrt(2 * np.log(2))
        new_threshold = abs(cumulative_error / len(self.error_history)) + sqrt_term * volatility

        # Apply learning rate for smooth updates
        self.threshold = (
            self.learning_rate * new_threshold +
            (1 - self.learning_rate) * self.threshold
        )

        # Clip to valid range
        self.threshold = np.clip(
            self.threshold,
            self.min_threshold,
            self.max_threshold
        )

    def get_signal(self, prediction: float) -> int:
        """
        Get trading signal based on current dynamic threshold.

        Returns: 1 (long), -1 (short), 0 (no trade)
        """
        if prediction > self.threshold:
            return 1
        elif prediction < -self.threshold:
            return -1
        return 0

    def reset(self) -> None:
        """Reset policy state."""
        self.threshold = self.initial_threshold
        self.prediction_history = []
        self.return_history = []
        self.error_history = []
```

### FR-TRAD-03: Portfolio Policy (Production Upgrade)

```python
@dataclass
class PortfolioConstraints:
    """Constraints for portfolio construction."""
    max_leverage: float = 1.0           # Max total exposure / capital
    max_single_position: float = 0.1    # Max single asset weight
    max_sector_exposure: float = 0.3    # Max exposure to single sector
    max_turnover_daily: float = 0.5     # Max daily turnover
    target_volatility: float = 0.15     # Annualized target vol
    min_position_size: float = 0.01     # Min position (below = 0)
    long_only: bool = False             # If True, no shorting


@dataclass
class TransactionCosts:
    """Transaction cost model."""
    commission_bps: float = 5.0         # Basis points per trade
    spread_bps: float = 10.0            # Bid-ask spread
    market_impact_bps: float = 5.0      # Market impact (function of size)
    slippage_vol_mult: float = 0.5      # Slippage = mult * volatility * participation

    def estimate_cost(
        self,
        trade_value: float,
        volatility: float = 0.02,
        participation_rate: float = 0.01
    ) -> float:
        """Estimate total transaction cost in dollars."""
        commission = trade_value * (self.commission_bps / 10000)
        spread = trade_value * (self.spread_bps / 10000)
        impact = trade_value * (self.market_impact_bps / 10000)
        slippage = trade_value * self.slippage_vol_mult * volatility * participation_rate
        return commission + spread + impact + slippage


class PortfolioPolicy:
    """
    Production portfolio construction with risk management.
    """

    def __init__(
        self,
        constraints: PortfolioConstraints,
        costs: TransactionCosts,
        sector_mapping: Dict[str, str] = None
    ):
        self.constraints = constraints
        self.costs = costs
        self.sector_mapping = sector_mapping or {}

    def construct_portfolio(
        self,
        predictions: Dict[str, Dict[str, float]],  # asset -> {prob_up, confidence, ...}
        current_positions: Dict[str, float],       # asset -> current weight
        volatilities: Dict[str, float],            # asset -> recent volatility
        capital: float
    ) -> Dict[str, float]:
        """
        Construct target portfolio from predictions.

        Returns: asset -> target weight
        """
        target_weights = {}

        # Convert predictions to raw signals
        for asset, pred in predictions.items():
            confidence = pred.get("confidence", 0.5)
            direction = pred.get("direction", 0)

            # Scale by confidence
            raw_weight = direction * (confidence - 0.5) * 2  # Map to [-1, 1]
            target_weights[asset] = raw_weight

        # Apply volatility targeting
        target_weights = self._apply_volatility_targeting(
            target_weights, volatilities
        )

        # Apply constraints
        target_weights = self._apply_constraints(target_weights)

        # Apply turnover constraint
        target_weights = self._apply_turnover_constraint(
            target_weights, current_positions
        )

        return target_weights

    def _apply_volatility_targeting(
        self,
        weights: Dict[str, float],
        volatilities: Dict[str, float]
    ) -> Dict[str, float]:
        """Scale weights to target portfolio volatility."""
        # Simple approach: inverse volatility weighting
        scaled = {}
        for asset, weight in weights.items():
            vol = volatilities.get(asset, 0.02)
            # Scale inversely by volatility
            scale = self.constraints.target_volatility / (vol * np.sqrt(252))
            scaled[asset] = weight * min(scale, 2.0)  # Cap scaling
        return scaled

    def _apply_constraints(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply position and leverage constraints."""
        constrained = {}

        # Single position limits
        for asset, weight in weights.items():
            capped = np.clip(
                weight,
                -self.constraints.max_single_position,
                self.constraints.max_single_position
            )
            if self.constraints.long_only:
                capped = max(0, capped)
            constrained[asset] = capped

        # Leverage constraint
        total_exposure = sum(abs(w) for w in constrained.values())
        if total_exposure > self.constraints.max_leverage:
            scale = self.constraints.max_leverage / total_exposure
            constrained = {a: w * scale for a, w in constrained.items()}

        # Sector constraints
        sector_exposure = defaultdict(float)
        for asset, weight in constrained.items():
            sector = self.sector_mapping.get(asset, "Other")
            sector_exposure[sector] += abs(weight)

        for sector, exposure in sector_exposure.items():
            if exposure > self.constraints.max_sector_exposure:
                scale = self.constraints.max_sector_exposure / exposure
                for asset in constrained:
                    if self.sector_mapping.get(asset, "Other") == sector:
                        constrained[asset] *= scale

        # Minimum position size (round small positions to zero)
        for asset in list(constrained.keys()):
            if abs(constrained[asset]) < self.constraints.min_position_size:
                constrained[asset] = 0.0

        return constrained

    def _apply_turnover_constraint(
        self,
        target: Dict[str, float],
        current: Dict[str, float]
    ) -> Dict[str, float]:
        """Limit turnover to reduce transaction costs."""
        all_assets = set(target.keys()) | set(current.keys())
        total_turnover = sum(
            abs(target.get(a, 0) - current.get(a, 0))
            for a in all_assets
        )

        if total_turnover > self.constraints.max_turnover_daily:
            # Scale trades toward target
            scale = self.constraints.max_turnover_daily / total_turnover
            adjusted = {}
            for asset in all_assets:
                current_w = current.get(asset, 0)
                target_w = target.get(asset, 0)
                trade = target_w - current_w
                adjusted[asset] = current_w + trade * scale
            return adjusted

        return target
```

### FR-TRAD-04: Risk Management

```python
@dataclass
class RiskLimits:
    """Risk limits for kill switches."""
    max_daily_loss_pct: float = 0.02     # -2% daily loss triggers halt
    max_drawdown_pct: float = 0.10       # -10% drawdown triggers halt
    max_single_loss_pct: float = 0.05    # -5% on single position
    position_limit_multiple: float = 3.0  # Max 3x target position
    correlation_limit: float = 0.8        # Alert if portfolio correlation > 0.8


class RiskManager:
    """Real-time risk monitoring and kill switches."""

    def __init__(self, limits: RiskLimits, initial_capital: float):
        self.limits = limits
        self.initial_capital = initial_capital
        self.peak_capital = initial_capital
        self.current_capital = initial_capital
        self.daily_start_capital = initial_capital
        self.is_halted = False
        self.halt_reason = None

    def update_capital(self, new_capital: float) -> None:
        """Update capital and check limits."""
        self.current_capital = new_capital
        self.peak_capital = max(self.peak_capital, new_capital)

    def start_of_day(self) -> None:
        """Call at start of trading day."""
        self.daily_start_capital = self.current_capital

    def check_limits(self) -> Tuple[bool, Optional[str]]:
        """
        Check all risk limits.

        Returns: (is_ok, violation_reason)
        """
        if self.is_halted:
            return False, self.halt_reason

        # Daily loss check
        daily_return = (self.current_capital - self.daily_start_capital) / self.daily_start_capital
        if daily_return < -self.limits.max_daily_loss_pct:
            self.halt("Daily loss limit breached: {:.2%}".format(daily_return))
            return False, self.halt_reason

        # Drawdown check
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if drawdown > self.limits.max_drawdown_pct:
            self.halt("Drawdown limit breached: {:.2%}".format(drawdown))
            return False, self.halt_reason

        return True, None

    def halt(self, reason: str) -> None:
        """Halt trading."""
        self.is_halted = True
        self.halt_reason = reason
        logging.critical(f"TRADING HALTED: {reason}")

    def reset_halt(self) -> None:
        """Reset halt status (requires manual intervention)."""
        self.is_halted = False
        self.halt_reason = None
        logging.warning("Trading halt reset manually")
```

---

## 4.9 Backtesting Engine

### FR-BACK-01: Event-Driven Backtester

```python
@dataclass
class BacktestConfig:
    """Configuration for backtest runs."""
    start_date: datetime.date
    end_date: datetime.date
    initial_capital: float = 100_000.0
    execution_price: str = "open"        # "open", "close", "vwap"
    costs: TransactionCosts = field(default_factory=TransactionCosts)
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    slippage_model: str = "fixed"        # "fixed", "volatility", "volume"


@dataclass
class Trade:
    """Record of executed trade."""
    date: datetime.date
    asset: str
    side: str                # "buy" or "sell"
    quantity: float
    price: float
    value: float
    cost: float
    signal_source: str


@dataclass
class DailySnapshot:
    """End-of-day portfolio snapshot."""
    date: datetime.date
    capital: float
    positions: Dict[str, float]    # asset -> quantity
    values: Dict[str, float]       # asset -> market value
    weights: Dict[str, float]      # asset -> weight
    daily_return: float
    cumulative_return: float
    trades_today: List[Trade]


class Backtester:
    """Event-driven backtesting engine."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.capital = config.initial_capital
        self.positions: Dict[str, float] = {}
        self.snapshots: List[DailySnapshot] = []
        self.all_trades: List[Trade] = []

    def run(
        self,
        signals: Dict[datetime.date, Dict[str, int]],    # date -> asset -> signal
        prices: Dict[datetime.date, Dict[str, Dict]],     # date -> asset -> {open, high, low, close, volume}
        volatilities: Dict[datetime.date, Dict[str, float]] = None
    ) -> BacktestResult:
        """
        Run backtest over date range.

        signals: Trading signals per day per asset (-1, 0, 1)
        prices: OHLCV data per day per asset
        """
        current_date = self.config.start_date
        prev_capital = self.capital

        while current_date <= self.config.end_date:
            if current_date not in prices:
                current_date += timedelta(days=1)
                continue

            day_prices = prices[current_date]
            day_signals = signals.get(current_date, {})
            day_vols = (volatilities or {}).get(current_date, {})

            # Execute trades based on signals
            trades_today = self._execute_day(
                current_date, day_signals, day_prices, day_vols
            )

            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(day_prices)

            # Record snapshot
            daily_return = (portfolio_value - prev_capital) / prev_capital if prev_capital > 0 else 0
            cumulative_return = (portfolio_value - self.config.initial_capital) / self.config.initial_capital

            snapshot = DailySnapshot(
                date=current_date,
                capital=portfolio_value,
                positions=self.positions.copy(),
                values=self._position_values(day_prices),
                weights=self._position_weights(day_prices),
                daily_return=daily_return,
                cumulative_return=cumulative_return,
                trades_today=trades_today
            )
            self.snapshots.append(snapshot)

            prev_capital = portfolio_value
            self.capital = portfolio_value
            current_date += timedelta(days=1)

        return self._compile_results()

    def _execute_day(
        self,
        date: datetime.date,
        signals: Dict[str, int],
        prices: Dict[str, Dict],
        volatilities: Dict[str, float]
    ) -> List[Trade]:
        """Execute trades for a single day."""
        trades = []

        # Determine execution price
        for asset, signal in signals.items():
            if asset not in prices:
                continue

            current_position = self.positions.get(asset, 0)
            target_position = signal  # Simplified: signal is target position in shares

            if signal == 0 and current_position != 0:
                # Close position
                trade_qty = -current_position
            elif signal != 0:
                # Open/adjust position
                # Calculate target shares based on equal weight
                price = self._get_execution_price(prices[asset])
                target_value = self.capital * self.config.constraints.max_single_position * signal
                target_shares = target_value / price
                trade_qty = target_shares - current_position
            else:
                continue

            if abs(trade_qty) < 0.01:  # Minimum trade size
                continue

            # Execute trade
            exec_price = self._get_execution_price(prices[asset])
            trade_value = abs(trade_qty * exec_price)
            vol = volatilities.get(asset, 0.02)
            cost = self.config.costs.estimate_cost(trade_value, vol)

            trade = Trade(
                date=date,
                asset=asset,
                side="buy" if trade_qty > 0 else "sell",
                quantity=abs(trade_qty),
                price=exec_price,
                value=trade_value,
                cost=cost,
                signal_source="model"
            )
            trades.append(trade)
            self.all_trades.append(trade)

            # Update position
            self.positions[asset] = self.positions.get(asset, 0) + trade_qty

            # Deduct costs from capital (for cash accounting)
            self.capital -= cost

        return trades

    def _get_execution_price(self, ohlcv: Dict) -> float:
        """Get execution price based on config."""
        if self.config.execution_price == "open":
            return ohlcv["open"]
        elif self.config.execution_price == "close":
            return ohlcv["close"]
        elif self.config.execution_price == "vwap":
            # Approximate VWAP as average of OHLC
            return (ohlcv["open"] + ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 4
        return ohlcv["close"]

    def _calculate_portfolio_value(self, prices: Dict[str, Dict]) -> float:
        """Calculate total portfolio value."""
        cash = self.capital
        positions_value = sum(
            qty * prices[asset]["close"]
            for asset, qty in self.positions.items()
            if asset in prices
        )
        return cash + positions_value

    def _position_values(self, prices: Dict[str, Dict]) -> Dict[str, float]:
        """Get market value of each position."""
        return {
            asset: qty * prices[asset]["close"]
            for asset, qty in self.positions.items()
            if asset in prices
        }

    def _position_weights(self, prices: Dict[str, Dict]) -> Dict[str, float]:
        """Get weight of each position."""
        total = self._calculate_portfolio_value(prices)
        if total == 0:
            return {}
        return {
            asset: (qty * prices[asset]["close"]) / total
            for asset, qty in self.positions.items()
            if asset in prices
        }

    def _compile_results(self) -> "BacktestResult":
        """Compile final backtest results."""
        returns = [s.daily_return for s in self.snapshots]
        return BacktestResult(
            snapshots=self.snapshots,
            trades=self.all_trades,
            metrics=self._calculate_metrics(returns)
        )

    def _calculate_metrics(self, returns: List[float]) -> Dict[str, float]:
        """Calculate performance metrics."""
        returns = np.array(returns)
        n_years = len(returns) / 252

        total_return = (1 + returns).prod() - 1
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        vol = returns.std() * np.sqrt(252)
        sharpe = (cagr - 0.02) / vol if vol > 0 else 0  # Assuming 2% risk-free

        # Sortino (downside deviation)
        downside = returns[returns < 0]
        downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.01
        sortino = (cagr - 0.02) / downside_std if downside_std > 0 else 0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        max_drawdown = drawdown.max()

        calmar = cagr / max_drawdown if max_drawdown > 0 else 0

        # Hit rate
        wins = (returns > 0).sum()
        hit_rate = wins / len(returns) if len(returns) > 0 else 0

        # Win/loss ratio
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # Turnover
        total_traded = sum(t.value for t in self.all_trades)
        avg_capital = np.mean([s.capital for s in self.snapshots])
        turnover = total_traded / avg_capital / n_years if n_years > 0 and avg_capital > 0 else 0

        return {
            "total_return": total_return,
            "cagr": cagr,
            "volatility": vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_drawdown,
            "calmar": calmar,
            "hit_rate": hit_rate,
            "win_loss_ratio": win_loss_ratio,
            "turnover": turnover,
            "total_trades": len(self.all_trades),
            "total_costs": sum(t.cost for t in self.all_trades)
        }


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    snapshots: List[DailySnapshot]
    trades: List[Trade]
    metrics: Dict[str, float]
```

---

*Continued in Part 4...*
