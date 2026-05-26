"""Correlation Network Analysis - Graph-based portfolio correlation analysis.

This module provides network analysis of asset correlations including:
- Correlation network construction
- Community detection (asset clustering)
- Centrality measures (systemic risk indicators)
- Network topology metrics
- Correlation breakdown analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import warnings


@dataclass
class NetworkMetrics:
    """Network topology metrics."""

    num_nodes: int
    num_edges: int
    density: float
    avg_degree: float
    clustering_coefficient: float
    num_communities: int
    modularity: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'density': self.density,
            'avg_degree': self.avg_degree,
            'clustering_coefficient': self.clustering_coefficient,
            'num_communities': self.num_communities,
            'modularity': self.modularity
        }


class CorrelationNetwork:
    """Analyzes correlation structure as a network graph."""

    def __init__(
        self,
        returns_df: pd.DataFrame,
        correlation_threshold: float = 0.3,
        lookback_window: int = 252
    ):
        """Initialize correlation network analyzer.

        Args:
            returns_df: DataFrame with asset returns (columns=assets, index=dates)
            correlation_threshold: Minimum correlation to create edge
            lookback_window: Lookback period for correlation calculation
        """
        self.returns_df = returns_df
        self.correlation_threshold = correlation_threshold
        self.lookback_window = lookback_window

        # Build correlation matrix
        self.corr_matrix = self._build_correlation_matrix()

        # Build adjacency matrix
        self.adj_matrix = self._build_adjacency_matrix()

        # Asset names
        self.assets = list(returns_df.columns)
        self.num_assets = len(self.assets)

    def _build_correlation_matrix(self) -> pd.DataFrame:
        """Build correlation matrix from returns."""
        if len(self.returns_df) < self.lookback_window:
            warnings.warn(
                f"Not enough data points ({len(self.returns_df)}) "
                f"for lookback window ({self.lookback_window})"
            )
            corr = self.returns_df.corr()
        else:
            # Use rolling window on most recent data
            recent_returns = self.returns_df.tail(self.lookback_window)
            corr = recent_returns.corr()

        return corr

    def _build_adjacency_matrix(self) -> np.ndarray:
        """Build adjacency matrix from correlation matrix."""
        adj = np.abs(self.corr_matrix.values) >= self.correlation_threshold
        np.fill_diagonal(adj, False)  # Remove self-loops
        return adj.astype(int)

    def get_network_metrics(self) -> NetworkMetrics:
        """Calculate network topology metrics.

        Returns:
            NetworkMetrics with graph statistics
        """
        num_nodes = self.num_assets
        num_edges = np.sum(self.adj_matrix) // 2  # Undirected graph

        # Density
        max_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_edges if max_edges > 0 else 0.0

        # Average degree
        degrees = np.sum(self.adj_matrix, axis=1)
        avg_degree = np.mean(degrees)

        # Clustering coefficient
        clustering = self._calculate_clustering_coefficient()

        # Community detection
        communities = self._detect_communities()
        num_communities = len(set(communities.values()))

        # Modularity
        modularity = self._calculate_modularity(communities)

        return NetworkMetrics(
            num_nodes=num_nodes,
            num_edges=num_edges,
            density=density,
            avg_degree=avg_degree,
            clustering_coefficient=clustering,
            num_communities=num_communities,
            modularity=modularity
        )

    def _calculate_clustering_coefficient(self) -> float:
        """Calculate global clustering coefficient."""
        coefficients = []

        for i in range(self.num_assets):
            neighbors = np.where(self.adj_matrix[i])[0]
            k = len(neighbors)

            if k < 2:
                continue

            # Count triangles
            triangles = 0
            for j in range(len(neighbors)):
                for l in range(j + 1, len(neighbors)):
                    if self.adj_matrix[neighbors[j], neighbors[l]]:
                        triangles += 1

            # Local clustering coefficient
            max_triangles = k * (k - 1) / 2
            local_coef = triangles / max_triangles if max_triangles > 0 else 0.0
            coefficients.append(local_coef)

        return np.mean(coefficients) if coefficients else 0.0

    def _detect_communities(self) -> Dict[str, int]:
        """Detect communities using simple greedy modularity optimization.

        Returns:
            Dictionary mapping asset to community ID
        """
        # Initialize each node in its own community
        communities = {asset: i for i, asset in enumerate(self.assets)}

        # Greedy merging based on modularity gain
        improved = True
        while improved:
            improved = False
            max_gain = 0.0
            best_merge = None

            # Try all pairs
            for i, asset_i in enumerate(self.assets):
                for j, asset_j in enumerate(self.assets[i+1:], start=i+1):
                    if communities[asset_i] != communities[asset_j]:
                        # Calculate modularity gain
                        gain = self._modularity_gain(
                            i, j, communities[asset_i], communities[asset_j]
                        )
                        if gain > max_gain:
                            max_gain = gain
                            best_merge = (asset_i, asset_j, communities[asset_j])

            if max_gain > 0 and best_merge:
                asset_i, asset_j, old_comm = best_merge
                new_comm = communities[asset_i]

                # Merge communities
                for asset in self.assets:
                    if communities[asset] == old_comm:
                        communities[asset] = new_comm

                improved = True

        return communities

    def _modularity_gain(
        self,
        i: int,
        j: int,
        comm_i: int,
        comm_j: int
    ) -> float:
        """Calculate modularity gain from merging two nodes."""
        m = np.sum(self.adj_matrix) / 2  # Total edges

        if m == 0:
            return 0.0

        # Degrees
        k_i = np.sum(self.adj_matrix[i])
        k_j = np.sum(self.adj_matrix[j])

        # Edge weight
        e_ij = self.adj_matrix[i, j]

        # Modularity gain
        gain = (e_ij - (k_i * k_j) / (2 * m)) / m

        return gain

    def _calculate_modularity(self, communities: Dict[str, int]) -> float:
        """Calculate modularity of community structure."""
        m = np.sum(self.adj_matrix) / 2

        if m == 0:
            return 0.0

        modularity = 0.0
        for i, asset_i in enumerate(self.assets):
            for j, asset_j in enumerate(self.assets):
                if communities[asset_i] == communities[asset_j]:
                    k_i = np.sum(self.adj_matrix[i])
                    k_j = np.sum(self.adj_matrix[j])
                    a_ij = self.adj_matrix[i, j]

                    modularity += a_ij - (k_i * k_j) / (2 * m)

        return modularity / (2 * m)

    def get_centrality_measures(self) -> pd.DataFrame:
        """Calculate centrality measures for all assets.

        Returns:
            DataFrame with centrality measures:
                - degree: Number of connections
                - eigenvector: Eigenvector centrality
                - betweenness: Betweenness centrality (simplified)
        """
        centrality_data = []

        degrees = np.sum(self.adj_matrix, axis=1)

        # Eigenvector centrality
        eigenvalues, eigenvectors = np.linalg.eig(self.adj_matrix)
        max_idx = np.argmax(eigenvalues.real)
        eigenvector_cent = np.abs(eigenvectors[:, max_idx].real)
        eigenvector_cent = eigenvector_cent / np.sum(eigenvector_cent)

        # Betweenness centrality (simplified)
        betweenness = self._calculate_betweenness()

        for i, asset in enumerate(self.assets):
            centrality_data.append({
                'asset': asset,
                'degree': degrees[i],
                'degree_normalized': degrees[i] / (self.num_assets - 1),
                'eigenvector_centrality': eigenvector_cent[i],
                'betweenness_centrality': betweenness[i]
            })

        df = pd.DataFrame(centrality_data)
        df = df.sort_values('eigenvector_centrality', ascending=False)

        return df

    def _calculate_betweenness(self) -> np.ndarray:
        """Calculate betweenness centrality (simplified version)."""
        betweenness = np.zeros(self.num_assets)

        # For each pair of nodes, count shortest paths through each node
        for s in range(self.num_assets):
            for t in range(s + 1, self.num_assets):
                # Find all shortest paths (BFS)
                paths = self._find_shortest_paths(s, t)

                if not paths:
                    continue

                # Count nodes on shortest paths
                for path in paths:
                    for node in path[1:-1]:  # Exclude endpoints
                        betweenness[node] += 1.0 / len(paths)

        # Normalize
        n = self.num_assets
        if n > 2:
            betweenness = betweenness / ((n - 1) * (n - 2) / 2)

        return betweenness

    def _find_shortest_paths(self, source: int, target: int) -> List[List[int]]:
        """Find all shortest paths between two nodes (BFS)."""
        if source == target:
            return [[source]]

        # BFS to find shortest path length
        visited = {source}
        queue = [(source, [source])]
        shortest_length = None
        all_paths = []

        while queue:
            node, path = queue.pop(0)

            if shortest_length and len(path) > shortest_length:
                continue

            for neighbor in range(self.num_assets):
                if self.adj_matrix[node, neighbor]:
                    new_path = path + [neighbor]

                    if neighbor == target:
                        if shortest_length is None:
                            shortest_length = len(new_path)
                        if len(new_path) == shortest_length:
                            all_paths.append(new_path)
                    elif neighbor not in visited or len(new_path) < shortest_length:
                        visited.add(neighbor)
                        queue.append((neighbor, new_path))

        return all_paths

    def identify_systemic_risk_nodes(self, top_n: int = 5) -> pd.DataFrame:
        """Identify assets with highest systemic risk.

        Args:
            top_n: Number of top risk nodes to return

        Returns:
            DataFrame with systemic risk indicators
        """
        centrality_df = self.get_centrality_measures()

        # Systemic risk score (weighted combination)
        centrality_df['systemic_risk_score'] = (
            0.3 * centrality_df['degree_normalized'] +
            0.5 * centrality_df['eigenvector_centrality'] +
            0.2 * centrality_df['betweenness_centrality']
        )

        return centrality_df.nlargest(top_n, 'systemic_risk_score')

    def analyze_correlation_breakdown(
        self,
        stress_returns: pd.Series
    ) -> pd.DataFrame:
        """Analyze how correlations change during stress.

        Args:
            stress_returns: Returns during stress period

        Returns:
            DataFrame with correlation breakdown analysis
        """
        # Normal correlations
        normal_corr = self.corr_matrix

        # Stress correlations
        stress_df = self.returns_df.loc[stress_returns.index]
        if len(stress_df) < 2:
            raise ValueError("Not enough data in stress period")

        stress_corr = stress_df.corr()

        # Calculate changes
        corr_change = stress_corr - normal_corr

        results = []
        for i, asset_i in enumerate(self.assets):
            for j, asset_j in enumerate(self.assets[i+1:], start=i+1):
                results.append({
                    'asset_1': asset_i,
                    'asset_2': asset_j,
                    'normal_corr': normal_corr.iloc[i, j],
                    'stress_corr': stress_corr.iloc[i, j],
                    'corr_change': corr_change.iloc[i, j],
                    'breakdown': corr_change.iloc[i, j] > 0.2  # Threshold for breakdown
                })

        df = pd.DataFrame(results)
        return df.sort_values('corr_change', ascending=False)

    def get_communities(self) -> Dict[str, List[str]]:
        """Get detected communities.

        Returns:
            Dictionary mapping community ID to list of assets
        """
        communities = self._detect_communities()

        # Group by community
        community_map = defaultdict(list)
        for asset, comm_id in communities.items():
            community_map[comm_id].append(asset)

        return dict(community_map)

    def export_network_graph(self) -> Dict:
        """Export network in graph format for visualization.

        Returns:
            Dictionary with nodes and edges for graph visualization
        """
        nodes = []
        centrality = self.get_centrality_measures()

        for _, row in centrality.iterrows():
            nodes.append({
                'id': row['asset'],
                'degree': int(row['degree']),
                'centrality': float(row['eigenvector_centrality'])
            })

        edges = []
        for i, asset_i in enumerate(self.assets):
            for j, asset_j in enumerate(self.assets[i+1:], start=i+1):
                if self.adj_matrix[i, j]:
                    edges.append({
                        'source': asset_i,
                        'target': asset_j,
                        'weight': float(self.corr_matrix.iloc[i, j])
                    })

        return {'nodes': nodes, 'edges': edges}


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)

    # Generate sample returns
    n_assets = 20
    n_days = 500

    returns = pd.DataFrame(
        np.random.randn(n_days, n_assets) * 0.02,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )

    # Add some correlation structure
    returns.iloc[:, :5] += np.random.randn(n_days, 1) * 0.01  # Sector 1
    returns.iloc[:, 5:10] += np.random.randn(n_days, 1) * 0.01  # Sector 2

    # Create network
    network = CorrelationNetwork(
        returns_df=returns,
        correlation_threshold=0.3
    )

    # Get metrics
    metrics = network.get_network_metrics()
    print("Network Metrics:")
    print(metrics.to_dict())

    # Get systemic risk nodes
    risk_nodes = network.identify_systemic_risk_nodes(top_n=5)
    print("\nSystemic Risk Nodes:")
    print(risk_nodes)

    # Get communities
    communities = network.get_communities()
    print(f"\nDetected {len(communities)} communities")
    for comm_id, assets in communities.items():
        print(f"Community {comm_id}: {len(assets)} assets")
