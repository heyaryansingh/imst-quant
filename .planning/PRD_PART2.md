# PRD Part 2: Sentiment, Influence, and Credibility

## 4.4 Sentiment Analysis

### FR-SENT-01: Baseline Sentiment (TextBlob - Paper Replication)

```python
from textblob import TextBlob
from typing import NamedTuple

class SentimentScore(NamedTuple):
    polarity: float      # [-1, 1]
    subjectivity: float  # [0, 1]
    method: str

class BaselineSentimentAnalyzer:
    """Paper replication: TextBlob polarity."""

    def analyze(self, text: str) -> SentimentScore:
        blob = TextBlob(text)
        return SentimentScore(
            polarity=blob.sentiment.polarity,
            subjectivity=blob.sentiment.subjectivity,
            method="textblob"
        )
```

### FR-SENT-02: Time Bucketing (Paper Replication)

```python
from datetime import datetime, timedelta
import polars as pl

# Paper specifies 3-hour intervals aggregated to daily
BUCKET_HOURS = 3
BUCKETS_PER_DAY = 24 // BUCKET_HOURS  # 8 buckets

def assign_time_bucket(timestamp: datetime) -> int:
    """Assign post to 3-hour bucket within day."""
    return timestamp.hour // BUCKET_HOURS

def aggregate_daily_sentiment(
    posts_df: pl.DataFrame,
    date: datetime.date,
    influence_scores: Dict[str, float]
) -> Dict[str, float]:
    """
    Aggregate sentiment to daily level per asset.

    Paper formula (interpreted):
    sentiment_index(asset, day) =
        sum(polarity_i * influence_i) / sum(influence_i)

    Where i iterates over all posts mentioning the asset on that day.
    """
    day_posts = posts_df.filter(
        pl.col("date") == date
    )

    results = {}
    for asset in day_posts["asset_id"].unique():
        asset_posts = day_posts.filter(pl.col("asset_id") == asset)

        weighted_sum = 0.0
        weight_sum = 0.0

        for row in asset_posts.iter_rows(named=True):
            author_id = row["author_id"]
            polarity = row["polarity"]
            influence = influence_scores.get(author_id, 1.0)  # Default 1.0

            weighted_sum += polarity * influence
            weight_sum += influence

        if weight_sum > 0:
            results[asset] = weighted_sum / weight_sum
        else:
            results[asset] = 0.0

    return results
```

**Normalization Options** (configurable):
```python
class NormalizationMethod(Enum):
    SUM_INFLUENCE = "sum_influence"      # Divide by sum of influence weights
    POST_COUNT = "post_count"            # Divide by number of posts
    NONE = "none"                        # Raw weighted sum

# Default: SUM_INFLUENCE (matches paper interpretation)
DEFAULT_NORMALIZATION = NormalizationMethod.SUM_INFLUENCE
```

### FR-SENT-03: Upgraded Sentiment (FinBERT)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class FinBERTAnalyzer:
    """Production sentiment using FinBERT."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.labels = ["negative", "neutral", "positive"]

    @torch.no_grad()
    def analyze(self, text: str) -> Dict[str, float]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

        return {
            "negative": float(probs[0]),
            "neutral": float(probs[1]),
            "positive": float(probs[2]),
            "sentiment_score": float(probs[2] - probs[0]),  # [-1, 1]
            "method": "finbert"
        }

    def batch_analyze(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, float]]:
        """Batch inference for efficiency."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

            for j, p in enumerate(probs):
                results.append({
                    "negative": float(p[0]),
                    "neutral": float(p[1]),
                    "positive": float(p[2]),
                    "sentiment_score": float(p[2] - p[0]),
                    "method": "finbert"
                })
        return results
```

### FR-SENT-04: Stance Classification

```python
class StanceClassifier:
    """
    Classify stance toward specific asset.
    Three-way: bullish / bearish / neutral
    """

    def __init__(self):
        # Use fine-tuned model or weak supervision
        self.model = self._load_or_train_model()

    def classify(
        self,
        text: str,
        asset: str
    ) -> Dict[str, float]:
        """
        Classify stance toward specific asset.
        Returns probabilities for each stance.
        """
        # Prepend asset context
        input_text = f"Asset: {asset}. Text: {text}"

        # Run inference
        probs = self._inference(input_text)

        return {
            "bullish": float(probs[0]),
            "neutral": float(probs[1]),
            "bearish": float(probs[2]),
            "stance_score": float(probs[0] - probs[2]),  # [-1, 1]
        }

    def _load_or_train_model(self):
        """
        Training approach: Weak supervision + small labeled set

        Weak supervision rules:
        - Contains "buy", "long", "moon", "bullish" -> bullish
        - Contains "sell", "short", "crash", "bearish" -> bearish
        - Contains "hold", "wait", "unsure" -> neutral

        Then fine-tune on 500 manually labeled examples.
        """
        # Implementation details in training pipeline
        pass
```

### FR-SENT-05: Event Tagging

```python
class EventType(Enum):
    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    ACQUISITION = "acquisition"
    MERGER = "merger"
    REGULATION = "regulation"
    LAWSUIT = "lawsuit"
    PRODUCT = "product"
    MACRO = "macro"
    HACK = "hack"           # Crypto specific
    PARTNERSHIP = "partnership"
    RUMOR = "rumor"
    NONE = "none"

class EventTagger:
    """Tag posts with event types."""

    PATTERNS = {
        EventType.EARNINGS: [
            r"earnings", r"eps", r"revenue", r"quarterly",
            r"beat", r"miss", r"guidance"
        ],
        EventType.ACQUISITION: [
            r"acqui", r"buyout", r"takeover", r"purchase"
        ],
        EventType.REGULATION: [
            r"sec", r"ftc", r"regulation", r"lawsuit", r"antitrust",
            r"investigate", r"subpoena"
        ],
        EventType.PRODUCT: [
            r"launch", r"release", r"announce", r"unveil",
            r"new product", r"upgrade"
        ],
        EventType.HACK: [
            r"hack", r"exploit", r"breach", r"stolen",
            r"vulnerability", r"attack"
        ],
        EventType.PARTNERSHIP: [
            r"partner", r"collaboration", r"deal", r"agreement"
        ],
        EventType.RUMOR: [
            r"rumor", r"hear", r"reportedly", r"allegedly",
            r"sources say", r"insider"
        ],
    }

    def tag(self, text: str) -> List[EventType]:
        """Tag text with relevant event types."""
        text_lower = text.lower()
        events = []

        for event_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    events.append(event_type)
                    break

        return events if events else [EventType.NONE]
```

### FR-SENT-06: Aggregation Features

```python
@dataclass
class SentimentAggregates:
    """Daily sentiment aggregates per asset."""

    # Volume metrics
    post_count: int
    unique_authors: int
    comment_count: int

    # Raw sentiment
    sentiment_mean: float
    sentiment_std: float
    sentiment_median: float

    # Stance metrics
    stance_mean: float           # E[stance_score]
    stance_entropy: float        # Disagreement measure
    bullish_ratio: float         # Fraction bullish
    bearish_ratio: float         # Fraction bearish

    # Weighted metrics
    influence_weighted_sentiment: float
    credibility_weighted_sentiment: float
    manipulation_adjusted_sentiment: float

    # Event metrics
    earnings_mentions: int
    acquisition_mentions: int
    regulation_mentions: int
    hack_mentions: int

    # Novelty (optional - requires BERTopic)
    topic_novelty_score: Optional[float]


def compute_aggregates(
    posts: List[EnrichedPost],
    influence_scores: Dict[str, float],
    credibility_scores: Dict[str, float],
    manipulation_flags: Dict[str, float]
) -> SentimentAggregates:
    """Compute all sentiment aggregates for a day."""

    sentiments = [p.sentiment_score for p in posts]
    stances = [p.stance_score for p in posts]
    authors = list(set(p.author_id for p in posts))

    # Basic stats
    sentiment_mean = np.mean(sentiments) if sentiments else 0.0
    sentiment_std = np.std(sentiments) if len(sentiments) > 1 else 0.0
    sentiment_median = np.median(sentiments) if sentiments else 0.0

    # Stance entropy
    stance_counts = np.bincount(
        [s > 0.3 for s in stances] +
        [s < -0.3 for s in stances] +
        [abs(s) <= 0.3 for s in stances],
        minlength=3
    )
    stance_probs = stance_counts / len(stances) if stances else np.array([1/3, 1/3, 1/3])
    stance_entropy = -np.sum(stance_probs * np.log(stance_probs + 1e-10))

    # Weighted sentiment
    influence_weighted = _weighted_mean(
        sentiments,
        [influence_scores.get(p.author_id, 1.0) for p in posts]
    )
    credibility_weighted = _weighted_mean(
        sentiments,
        [credibility_scores.get(p.author_id, 0.5) for p in posts]
    )

    # Manipulation adjusted
    manip_weights = [
        1.0 - manipulation_flags.get(p.author_id, 0.0)
        for p in posts
    ]
    manipulation_adjusted = _weighted_mean(sentiments, manip_weights)

    # Event counts
    all_events = [e for p in posts for e in p.event_types]
    event_counts = Counter(all_events)

    return SentimentAggregates(
        post_count=len(posts),
        unique_authors=len(authors),
        comment_count=sum(1 for p in posts if p.post_type == "comment"),
        sentiment_mean=sentiment_mean,
        sentiment_std=sentiment_std,
        sentiment_median=sentiment_median,
        stance_mean=np.mean(stances) if stances else 0.0,
        stance_entropy=stance_entropy,
        bullish_ratio=sum(1 for s in stances if s > 0.3) / len(stances) if stances else 0.0,
        bearish_ratio=sum(1 for s in stances if s < -0.3) / len(stances) if stances else 0.0,
        influence_weighted_sentiment=influence_weighted,
        credibility_weighted_sentiment=credibility_weighted,
        manipulation_adjusted_sentiment=manipulation_adjusted,
        earnings_mentions=event_counts.get(EventType.EARNINGS, 0),
        acquisition_mentions=event_counts.get(EventType.ACQUISITION, 0),
        regulation_mentions=event_counts.get(EventType.REGULATION, 0),
        hack_mentions=event_counts.get(EventType.HACK, 0),
        topic_novelty_score=None,  # Computed separately if enabled
    )

def _weighted_mean(values: List[float], weights: List[float]) -> float:
    if not values:
        return 0.0
    total_weight = sum(weights)
    if total_weight == 0:
        return np.mean(values)
    return sum(v * w for v, w in zip(values, weights)) / total_weight
```

---

## 4.5 Influence Modeling

### FR-INF-01: Graph Construction (Paper Replication)

```python
import networkx as nx
from datetime import datetime
from typing import Set, Tuple

class InteractionGraph:
    """
    Monthly interaction graph for influence modeling.

    Nodes: Authors who posted about target assets
    Edges: Interactions (replies, co-participation)
    """

    def __init__(
        self,
        min_interactions: int = 50,  # Paper threshold
        include_edge_types: Set[str] = None
    ):
        self.min_interactions = min_interactions
        self.include_edge_types = include_edge_types or {
            "reply",           # Comment reply
            "co_thread",       # Same thread participation
            "mention"          # u/ mention
        }

    def build_monthly_graph(
        self,
        posts: List[RedditPost],
        month_start: datetime,
        month_end: datetime
    ) -> nx.DiGraph:
        """Build interaction graph for a month."""

        # Filter to month
        month_posts = [
            p for p in posts
            if month_start <= datetime.fromtimestamp(p.created_utc) < month_end
        ]

        # Count interactions per author
        author_interactions = Counter()
        edges = []

        for post in month_posts:
            author_interactions[post.author_id] += 1

            # Reply edges
            if post.post_type == "comment" and post.parent_id:
                parent_author = self._get_parent_author(post.parent_id, posts)
                if parent_author:
                    edges.append((
                        post.author_id,
                        parent_author,
                        "reply"
                    ))
                    author_interactions[post.author_id] += 1
                    author_interactions[parent_author] += 1

            # Mention edges
            mentions = self._extract_mentions(post.body or post.selftext or "")
            for mentioned in mentions:
                edges.append((
                    post.author_id,
                    mentioned,
                    "mention"
                ))

        # Build co-thread edges
        thread_authors = defaultdict(set)
        for post in month_posts:
            thread_id = post.link_id or post.id
            thread_authors[thread_id].add(post.author_id)

        for thread_id, authors in thread_authors.items():
            if len(authors) > 1:
                for a1, a2 in combinations(authors, 2):
                    edges.append((a1, a2, "co_thread"))
                    edges.append((a2, a1, "co_thread"))

        # Filter by minimum interactions
        eligible_authors = {
            a for a, count in author_interactions.items()
            if count >= self.min_interactions
        }

        # Build graph
        G = nx.DiGraph()
        for author in eligible_authors:
            G.add_node(author)

        for src, dst, edge_type in edges:
            if src in eligible_authors and dst in eligible_authors:
                if edge_type in self.include_edge_types:
                    if G.has_edge(src, dst):
                        G[src][dst]["weight"] += 1
                    else:
                        G.add_edge(src, dst, weight=1, type=edge_type)

        return G

    def _get_parent_author(
        self,
        parent_id: str,
        posts: List[RedditPost]
    ) -> Optional[str]:
        """Look up author of parent post."""
        for p in posts:
            if p.id == parent_id.split("_")[-1]:
                return p.author_id
        return None

    def _extract_mentions(self, text: str) -> List[str]:
        """Extract u/ mentions from text."""
        pattern = r'u/([A-Za-z0-9_-]+)'
        return re.findall(pattern, text)
```

### FR-INF-02: GNN Training (Paper Replication)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class InfluenceGNN(nn.Module):
    """
    2-layer GCN for influence scoring.

    Paper architecture:
    - Input: node features (degree, interaction count)
    - 2 GCNConv layers with ReLU
    - Output: influence score per node
    """

    def __init__(
        self,
        input_dim: int = 4,        # Node feature dimension
        hidden_dim: int = 64,      # Hidden dimension
        output_dim: int = 1,       # Influence score
        dropout: float = 0.5
    ):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # Layer 1
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index, edge_weight)

        # Output influence scores (sigmoid to [0, 1])
        return torch.sigmoid(x).squeeze(-1)


def prepare_gnn_data(G: nx.DiGraph) -> Data:
    """Convert NetworkX graph to PyG Data object."""

    # Node mapping
    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    idx_to_node = {i: node for node, i in node_to_idx.items()}

    # Node features
    # [in_degree, out_degree, total_weight_in, total_weight_out]
    features = []
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        weight_in = sum(G[u][node].get("weight", 1) for u in G.predecessors(node))
        weight_out = sum(G[node][v].get("weight", 1) for v in G.successors(node))
        features.append([in_deg, out_deg, weight_in, weight_out])

    x = torch.tensor(features, dtype=torch.float)

    # Normalize features
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)

    # Edge index
    edge_list = list(G.edges())
    edge_index = torch.tensor([
        [node_to_idx[e[0]] for e in edge_list],
        [node_to_idx[e[1]] for e in edge_list]
    ], dtype=torch.long)

    # Edge weights
    edge_weight = torch.tensor([
        G[e[0]][e[1]].get("weight", 1.0) for e in edge_list
    ], dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        node_mapping=idx_to_node
    )
```

### FR-INF-03: Training Objective

**Paper Interpretation**: The paper does not specify exact loss function. Based on the description that "loss stabilizes quickly", we interpret this as a self-supervised approach.

**Implemented Approach**:
```python
class InfluenceTrainer:
    """
    Self-supervised training for influence scores.

    Pseudo-label approach:
    - Nodes with high in-degree and engagement should have high influence
    - Use PageRank or interaction-weighted degree as pseudo-labels
    """

    def __init__(
        self,
        model: InfluenceGNN,
        lr: float = 0.01,
        epochs: int = 100,
        pseudo_label_method: str = "pagerank"  # or "weighted_degree"
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.epochs = epochs
        self.pseudo_label_method = pseudo_label_method

    def compute_pseudo_labels(self, G: nx.DiGraph) -> torch.Tensor:
        """Compute pseudo influence labels."""
        if self.pseudo_label_method == "pagerank":
            pr = nx.pagerank(G, weight="weight")
            scores = [pr.get(n, 0) for n in G.nodes()]
        else:  # weighted_degree
            scores = []
            for n in G.nodes():
                in_weight = sum(G[u][n].get("weight", 1) for u in G.predecessors(n))
                scores.append(in_weight)

        # Normalize to [0, 1]
        scores = np.array(scores)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return torch.tensor(scores, dtype=torch.float)

    def train(self, data: Data, G: nx.DiGraph) -> Dict[str, float]:
        """Train the influence model."""
        pseudo_labels = self.compute_pseudo_labels(G)

        self.model.train()
        losses = []

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            predictions = self.model(data)

            # MSE loss against pseudo-labels
            loss = F.mse_loss(predictions, pseudo_labels)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

        return {
            "final_loss": losses[-1],
            "loss_history": losses
        }

    def get_influence_scores(
        self,
        data: Data
    ) -> Dict[str, float]:
        """Get final influence scores per author."""
        self.model.eval()
        with torch.no_grad():
            scores = self.model(data).numpy()

        return {
            data.node_mapping[i]: float(scores[i])
            for i in range(len(scores))
        }
```

---

## 4.6 Credibility and Bot Detection

### FR-CRED-01: Author Features

```python
@dataclass
class AuthorProfile:
    author_id: str

    # Account characteristics
    account_age_days: Optional[int]
    total_karma: Optional[int]        # If available
    comment_karma: Optional[int]
    link_karma: Optional[int]

    # Posting behavior
    total_posts: int
    total_comments: int
    avg_posts_per_day: float
    posting_hours_entropy: float       # Regular vs bursty
    weekend_ratio: float               # Weekend activity

    # Content patterns
    avg_post_length: float
    link_ratio: float                  # Posts with external links
    cashtag_density: float             # Cashtags per post
    repetition_rate: float             # Near-duplicate rate

    # Engagement patterns
    avg_score: float
    avg_comment_count: float           # On submissions

    # Diversity
    subreddit_entropy: float           # Posting across subreddits
    asset_diversity: float             # Discussing multiple assets

    # Derived scores
    bot_probability: float
    spam_probability: float
    credibility_score: float


class AuthorProfiler:
    """Build author profiles from posting history."""

    def __init__(self, lookback_days: int = 90):
        self.lookback_days = lookback_days
        self.deduplicator = Deduplicator()

    def build_profile(
        self,
        author_id: str,
        posts: List[RedditPost]
    ) -> AuthorProfile:
        """Build comprehensive author profile."""

        author_posts = [p for p in posts if p.author_id == author_id]

        if not author_posts:
            return self._default_profile(author_id)

        # Posting cadence
        timestamps = [p.created_utc for p in author_posts]
        posting_hours = [datetime.fromtimestamp(t).hour for t in timestamps]
        hour_counts = np.bincount(posting_hours, minlength=24)
        posting_hours_entropy = entropy(hour_counts / hour_counts.sum())

        # Weekend ratio
        weekdays = [datetime.fromtimestamp(t).weekday() for t in timestamps]
        weekend_ratio = sum(1 for w in weekdays if w >= 5) / len(weekdays)

        # Content analysis
        texts = [p.selftext or p.body or "" for p in author_posts]
        avg_length = np.mean([len(t) for t in texts])
        link_posts = sum(1 for p in author_posts if p.url and "reddit.com" not in p.url)
        link_ratio = link_posts / len(author_posts)

        # Cashtag density
        cashtags_per_post = [
            len(CASHTAG_PATTERN.findall(t)) for t in texts
        ]
        cashtag_density = np.mean(cashtags_per_post)

        # Repetition rate
        dup_count = 0
        for i, t in enumerate(texts):
            is_dup, _ = self.deduplicator.is_duplicate(f"{author_id}_{i}", t)
            if is_dup:
                dup_count += 1
        repetition_rate = dup_count / len(texts) if texts else 0

        # Subreddit diversity
        subreddits = [p.subreddit for p in author_posts]
        sub_counts = np.array(list(Counter(subreddits).values()))
        subreddit_entropy = entropy(sub_counts / sub_counts.sum())

        # Compute scores
        bot_prob = self._estimate_bot_probability(
            posting_hours_entropy=posting_hours_entropy,
            repetition_rate=repetition_rate,
            avg_length=avg_length,
            link_ratio=link_ratio
        )

        spam_prob = self._estimate_spam_probability(
            cashtag_density=cashtag_density,
            link_ratio=link_ratio,
            repetition_rate=repetition_rate
        )

        credibility = 1.0 - max(bot_prob, spam_prob)

        return AuthorProfile(
            author_id=author_id,
            account_age_days=None,  # Not always available
            total_karma=None,
            comment_karma=None,
            link_karma=None,
            total_posts=sum(1 for p in author_posts if p.post_type == "submission"),
            total_comments=sum(1 for p in author_posts if p.post_type == "comment"),
            avg_posts_per_day=len(author_posts) / self.lookback_days,
            posting_hours_entropy=posting_hours_entropy,
            weekend_ratio=weekend_ratio,
            avg_post_length=avg_length,
            link_ratio=link_ratio,
            cashtag_density=cashtag_density,
            repetition_rate=repetition_rate,
            avg_score=np.mean([p.score for p in author_posts]),
            avg_comment_count=np.mean([
                p.num_comments for p in author_posts
                if p.post_type == "submission" and p.num_comments is not None
            ]) if any(p.post_type == "submission" for p in author_posts) else 0,
            subreddit_entropy=subreddit_entropy,
            asset_diversity=0,  # Computed separately with entity links
            bot_probability=bot_prob,
            spam_probability=spam_prob,
            credibility_score=credibility
        )

    def _estimate_bot_probability(
        self,
        posting_hours_entropy: float,
        repetition_rate: float,
        avg_length: float,
        link_ratio: float
    ) -> float:
        """
        Rule-based bot probability.

        Bot indicators:
        - Low posting hour entropy (posts at same times)
        - High repetition rate
        - Very short or templated posts
        - High link ratio (spam)
        """
        score = 0.0

        # Low entropy = regular/automated posting
        if posting_hours_entropy < 2.0:
            score += 0.3

        # High repetition
        if repetition_rate > 0.3:
            score += 0.3

        # Very short posts
        if avg_length < 50:
            score += 0.2

        # High link ratio
        if link_ratio > 0.7:
            score += 0.2

        return min(score, 1.0)

    def _estimate_spam_probability(
        self,
        cashtag_density: float,
        link_ratio: float,
        repetition_rate: float
    ) -> float:
        """
        Rule-based spam probability.

        Spam indicators:
        - Excessive cashtags
        - High link ratio
        - Repetitive content
        """
        score = 0.0

        if cashtag_density > 3:
            score += 0.4

        if link_ratio > 0.5:
            score += 0.3

        if repetition_rate > 0.2:
            score += 0.3

        return min(score, 1.0)

    def _default_profile(self, author_id: str) -> AuthorProfile:
        """Default profile for unknown authors."""
        return AuthorProfile(
            author_id=author_id,
            account_age_days=None,
            total_karma=None,
            comment_karma=None,
            link_karma=None,
            total_posts=0,
            total_comments=0,
            avg_posts_per_day=0,
            posting_hours_entropy=3.0,  # Neutral
            weekend_ratio=0.3,
            avg_post_length=100,
            link_ratio=0.1,
            cashtag_density=0.5,
            repetition_rate=0,
            avg_score=1,
            avg_comment_count=0,
            subreddit_entropy=2.0,
            asset_diversity=0,
            bot_probability=0.3,       # Moderate default
            spam_probability=0.1,
            credibility_score=0.5      # Neutral default
        )
```

### FR-CRED-02: Brigade Detection

```python
class BrigadeDetector:
    """
    Detect coordinated inauthentic behavior.

    Signals:
    - Sudden spike in sentiment volume
    - Near-identical messages from multiple accounts
    - Unusual graph densification
    """

    def __init__(
        self,
        spike_threshold: float = 3.0,   # Std devs above mean
        similarity_threshold: float = 0.9,
        time_window_minutes: int = 60
    ):
        self.spike_threshold = spike_threshold
        self.similarity_threshold = similarity_threshold
        self.time_window = time_window_minutes * 60

    def detect_volume_spike(
        self,
        asset: str,
        current_volume: int,
        historical_volumes: List[int]
    ) -> bool:
        """Detect unusual volume spike."""
        if len(historical_volumes) < 7:
            return False

        mean_vol = np.mean(historical_volumes)
        std_vol = np.std(historical_volumes)

        if std_vol == 0:
            return current_volume > mean_vol * 2

        z_score = (current_volume - mean_vol) / std_vol
        return z_score > self.spike_threshold

    def detect_coordinated_posts(
        self,
        posts: List[RedditPost],
        time_window: int = None
    ) -> List[Set[str]]:
        """
        Find clusters of near-identical posts from different authors.

        Returns list of suspicious author sets.
        """
        time_window = time_window or self.time_window

        # Sort by time
        sorted_posts = sorted(posts, key=lambda p: p.created_utc)

        clusters = []
        for i, p1 in enumerate(sorted_posts):
            cluster = {p1.author_id}
            text1 = p1.selftext or p1.body or ""

            for p2 in sorted_posts[i+1:]:
                if p2.created_utc - p1.created_utc > time_window:
                    break

                if p2.author_id == p1.author_id:
                    continue

                text2 = p2.selftext or p2.body or ""

                # Check similarity
                sim = self._jaccard_similarity(text1, text2)
                if sim >= self.similarity_threshold:
                    cluster.add(p2.author_id)

            if len(cluster) >= 3:  # Minimum cluster size
                clusters.append(cluster)

        return clusters

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union

    def get_manipulation_risk(
        self,
        asset: str,
        posts: List[RedditPost],
        historical_volumes: List[int]
    ) -> Dict[str, Any]:
        """Comprehensive manipulation risk assessment."""
        current_volume = len(posts)

        # Volume spike
        volume_spike = self.detect_volume_spike(
            asset, current_volume, historical_volumes
        )

        # Coordinated posts
        coord_clusters = self.detect_coordinated_posts(posts)

        # Flagged authors
        flagged_authors = set()
        for cluster in coord_clusters:
            flagged_authors.update(cluster)

        # Risk score
        risk_score = 0.0
        if volume_spike:
            risk_score += 0.3
        if coord_clusters:
            risk_score += 0.2 * len(coord_clusters)
        risk_score = min(risk_score, 1.0)

        return {
            "asset": asset,
            "volume_spike": volume_spike,
            "coordinated_clusters": len(coord_clusters),
            "flagged_authors": list(flagged_authors),
            "manipulation_risk_score": risk_score,
            "alert": risk_score > 0.5
        }
```

---

*Continued in Part 3...*
