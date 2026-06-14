from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import random

import networkx as nx
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class KMeans:
    def __init__(
        self,
        n_clusters: int,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self._centroids: list[list[float]] = []

    def fit_predict(self, features: pd.DataFrame) -> list[int]:
        data = features.astype(float).values.tolist()
        if not data:
            return []
        n_samples = len(data)
        k = min(self.n_clusters, n_samples)

        best_assignments: list[int] | None = None
        best_inertia: float | None = None

        for init_idx in range(self.n_init):
            rng = random.Random(self.random_state + init_idx * 9973)
            seed_indices = rng.sample(list(range(n_samples)), k=k)
            centroids = [list(data[idx]) for idx in seed_indices]

            assignments = [0 for _ in range(n_samples)]
            for _ in range(self.max_iter):
                changed = False
                for idx, row in enumerate(data):
                    cluster_id = self._nearest_centroid(row, centroids)
                    if assignments[idx] != cluster_id:
                        assignments[idx] = cluster_id
                        changed = True

                grouped: list[list[list[float]]] = [[] for _ in range(k)]
                for idx, cluster_id in enumerate(assignments):
                    grouped[cluster_id].append(data[idx])

                new_centroids: list[list[float]] = []
                for cluster_id in range(k):
                    cluster_points = grouped[cluster_id]
                    if not cluster_points:
                        new_centroids.append(list(data[rng.randrange(n_samples)]))
                        continue
                    dims = len(cluster_points[0])
                    mean = [
                        sum(point[d] for point in cluster_points) / float(len(cluster_points))
                        for d in range(dims)
                    ]
                    new_centroids.append(mean)

                centroids = new_centroids
                if not changed:
                    break

            inertia = 0.0
            for idx, row in enumerate(data):
                c = centroids[assignments[idx]]
                inertia += sum((row[d] - c[d]) ** 2 for d in range(len(row)))

            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_assignments = list(assignments)
                self._centroids = [list(c) for c in centroids]

        return best_assignments or [0 for _ in range(n_samples)]

    @staticmethod
    def _nearest_centroid(row: list[float], centroids: list[list[float]]) -> int:
        best_cluster = 0
        best_distance = float("inf")
        for idx, centroid in enumerate(centroids):
            distance = sum((row[d] - centroid[d]) ** 2 for d in range(len(row)))
            if distance < best_distance:
                best_distance = distance
                best_cluster = idx
        return best_cluster


@dataclass(slots=True)
class _TreeNode:
    prediction: str
    feature_index: int = -1
    threshold: float = 0.0
    left: _TreeNode | None = None
    right: _TreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None or self.right is None or self.feature_index < 0


class _DecisionTreeClassifier:
    def __init__(
        self,
        random_state: int,
        max_depth: int = 12,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ) -> None:
        self._rng = random.Random(random_state)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self._root: _TreeNode | None = None

    def fit(self, x_rows: list[list[float]], y_rows: list[str]) -> None:
        self._root = self._build_tree(x_rows, y_rows, depth=0)

    def predict_one(self, row: list[float]) -> str:
        if self._root is None:
            raise RuntimeError("Tree not fitted.")
        node = self._root
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left if node.left is not None else node
            else:
                node = node.right if node.right is not None else node
        return node.prediction

    def _build_tree(self, x_rows: list[list[float]], y_rows: list[str], depth: int) -> _TreeNode:
        majority = self._majority_label(y_rows)
        node = _TreeNode(prediction=majority)

        if (
            depth >= self.max_depth
            or len(x_rows) < self.min_samples_split
            or len(set(y_rows)) <= 1
        ):
            return node

        n_features = len(x_rows[0]) if x_rows else 0
        if n_features == 0:
            return node

        max_features = max(1, int(math.sqrt(n_features)))
        feature_candidates = list(range(n_features))
        self._rng.shuffle(feature_candidates)
        feature_candidates = feature_candidates[:max_features]

        best_feature = -1
        best_threshold = 0.0
        best_impurity = float("inf")
        best_left_x: list[list[float]] = []
        best_left_y: list[str] = []
        best_right_x: list[list[float]] = []
        best_right_y: list[str] = []

        for feature_idx in feature_candidates:
            values = sorted({row[feature_idx] for row in x_rows})
            if len(values) <= 1:
                continue
            thresholds = [(values[i] + values[i + 1]) / 2.0 for i in range(len(values) - 1)]

            for threshold in thresholds:
                left_x: list[list[float]] = []
                left_y: list[str] = []
                right_x: list[list[float]] = []
                right_y: list[str] = []
                for row, label in zip(x_rows, y_rows):
                    if row[feature_idx] <= threshold:
                        left_x.append(row)
                        left_y.append(label)
                    else:
                        right_x.append(row)
                        right_y.append(label)

                if len(left_x) < self.min_samples_leaf or len(right_x) < self.min_samples_leaf:
                    continue

                impurity = (
                    (len(left_y) / len(y_rows)) * self._gini(left_y)
                    + (len(right_y) / len(y_rows)) * self._gini(right_y)
                )
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_x, best_left_y = left_x, left_y
                    best_right_x, best_right_y = right_x, right_y

        if best_feature < 0:
            return node

        node.feature_index = best_feature
        node.threshold = best_threshold
        node.left = self._build_tree(best_left_x, best_left_y, depth + 1)
        node.right = self._build_tree(best_right_x, best_right_y, depth + 1)
        return node

    @staticmethod
    def _gini(labels: list[str]) -> float:
        if not labels:
            return 0.0
        counts = Counter(labels)
        total = float(len(labels))
        return 1.0 - sum((count / total) ** 2 for count in counts.values())

    @staticmethod
    def _majority_label(labels: list[str]) -> str:
        counts = Counter(labels)
        top_count = max(counts.values())
        winners = sorted(label for label, count in counts.items() if count == top_count)
        return winners[0]


class RandomForestClassifier:
    def __init__(self, n_estimators: int = 200, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._trees: list[_DecisionTreeClassifier] = []

    def fit(self, feature_df: pd.DataFrame, targets: pd.Series) -> None:
        x_rows = feature_df.astype(float).values.tolist()
        y_rows = [str(label) for label in targets.tolist()]
        if not x_rows:
            raise ValueError("RandomForestClassifier.fit requires non-empty data.")

        n_samples = len(x_rows)
        rng = random.Random(self.random_state)
        self._trees = []

        for idx in range(self.n_estimators):
            sample_x: list[list[float]] = []
            sample_y: list[str] = []
            for _ in range(n_samples):
                sample_idx = rng.randrange(n_samples)
                sample_x.append(x_rows[sample_idx])
                sample_y.append(y_rows[sample_idx])

            tree = _DecisionTreeClassifier(random_state=self.random_state + idx * 37)
            tree.fit(sample_x, sample_y)
            self._trees.append(tree)

    def predict(self, feature_df: pd.DataFrame) -> list[str]:
        if not self._trees:
            raise RuntimeError("RandomForestClassifier is not fitted.")
        x_rows = feature_df.astype(float).values.tolist()
        predictions: list[str] = []
        for row in x_rows:
            votes = [tree.predict_one(row) for tree in self._trees]
            vote_counts = Counter(votes)
            top_count = max(vote_counts.values())
            winners = sorted(label for label, count in vote_counts.items() if count == top_count)
            predictions.append(winners[0])
        return predictions


RISK_LOW = 0
RISK_MEDIUM = 1
RISK_HIGH = 2

RISK_NAME_TO_INDEX = {"Low": RISK_LOW, "Medium": RISK_MEDIUM, "High": RISK_HIGH}


@dataclass(slots=True)
class CrimeRiskRunResult:
    predictions: dict[str, int]
    officer_allocation: dict[str, int]
    accuracy: float
    risk_counts: dict[int, int]
    class_balance: dict[str, int]
    warnings: list[str]
    fallback_used: bool


class CrimeRiskPredictor:
    """Challenge 5: crime clustering, classification, and graph integration."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=self.random_state,
        )

    def build_feature_frame(self, city_graph) -> pd.DataFrame:
        graph = city_graph.to_networkx(include_blocked=False)
        rows: list[dict[str, object]] = []
        for node_id, attrs in graph.nodes(data=True):
            population_density = float(attrs.get("population_density", 0.0))
            location_type = str(attrs.get("location_type", "Residential"))
            industrial_proximity = float(self._industrial_proximity(graph, node_id))
            rows.append(
                {
                    "node_id": node_id,
                    "population_density": population_density,
                    "industrial_proximity": industrial_proximity,
                    "location_type": location_type,
                }
            )
        if not rows:
            raise ValueError("City graph has no nodes to build C5 features.")
        return pd.DataFrame(rows)

    def cluster_neighborhoods(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[["population_density", "industrial_proximity"]].copy()
        # Min-max scaling keeps both features on a similar scale.
        for column in features.columns:
            min_value = float(features[column].min())
            max_value = float(features[column].max())
            span = max_value - min_value
            if span == 0:
                features[column] = 0.0
            else:
                features[column] = (features[column] - min_value) / span

        n_clusters = min(3, len(df))
        model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        clustered = df.copy()
        clustered["cluster_id"] = model.fit_predict(features)
        return clustered

    def generate_synthetic_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        labeled = df.copy()

        def compute_label(row: pd.Series) -> str:
            if row["population_density"] > 70 and row["industrial_proximity"] <= 2:
                return "High"
            if row["population_density"] < 30 or row["location_type"] in {"Hospital", "School"}:
                return "Low"
            return "Medium"

        labeled["risk_label"] = labeled.apply(compute_label, axis=1)
        return labeled

    def train_classifier(self, df: pd.DataFrame) -> tuple[float, dict[str, int], list[str], bool]:
        feature_df = self._model_features(df)
        targets = df["risk_label"]
        class_balance = {str(k): int(v) for k, v in targets.value_counts().to_dict().items()}
        warnings: list[str] = []
        fallback_used = False
        sample_count = len(df)
        if sample_count < 2:
            self.model.fit(feature_df, targets)
            warnings.append("tiny_sample_fallback")
            fallback_used = True
            return 1.0, class_balance, warnings, fallback_used

        if int(targets.nunique()) <= 1:
            # Single-class training set: fitting is still valid, but holdout
            # scoring is meaningless. Keep operation stable and explicit.
            self.model.fit(feature_df, targets)
            warnings.append("single_class_fallback")
            fallback_used = True
            return 1.0, class_balance, warnings, fallback_used

        test_size = max(1, int(round(sample_count * 0.2)))
        if test_size >= sample_count:
            test_size = sample_count - 1

        class_counts = targets.value_counts()
        class_count = int(targets.nunique())
        min_class_size = int(class_counts.min())
        can_stratify = (
            class_count > 1
            and min_class_size >= 2
            and test_size >= class_count
            and (sample_count - test_size) >= class_count
        )
        if not can_stratify:
            warnings.append("non_stratified_split")

        try:
            x_train, x_test, y_train, y_test = train_test_split(
                feature_df,
                targets,
                test_size=test_size,
                random_state=self.random_state,
                stratify=targets if can_stratify else None,
            )
            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)
            return float(accuracy_score(y_test, y_pred)), class_balance, warnings, fallback_used
        except Exception:
            # Defensive fallback for pathological tiny/imbalanced slices.
            self.model.fit(feature_df, targets)
            warnings.append("train_split_fallback")
            fallback_used = True
            return 1.0, class_balance, warnings, fallback_used

    def predict_and_update(self, city_graph) -> CrimeRiskRunResult:
        base_df = self.build_feature_frame(city_graph)
        clustered_df = self.cluster_neighborhoods(base_df)
        labeled_df = self.generate_synthetic_labels(clustered_df)
        accuracy, class_balance, warnings, fallback_used = self.train_classifier(labeled_df)

        model_inputs = self._model_features(labeled_df)
        pred_labels = self.model.predict(model_inputs)
        pred_df = labeled_df[["node_id", "population_density"]].copy()
        pred_df["predicted_label"] = pred_labels
        pred_df["predicted_risk"] = pred_df["predicted_label"].map(RISK_NAME_TO_INDEX).astype(int)

        predictions = {
            str(row.node_id): int(row.predicted_risk)
            for row in pred_df.itertuples(index=False)
        }
        # Single batched event: subscribers (notably C3) re-run heavy work at
        # most once per prediction pass instead of once per node.
        city_graph.set_risks_bulk(predictions)

        allocation = self.allocate_officers(pred_df, total_officers=10)
        city_graph.set_officer_allocation_bulk(allocation)
        counts = {
            RISK_LOW: sum(1 for value in predictions.values() if value == RISK_LOW),
            RISK_MEDIUM: sum(1 for value in predictions.values() if value == RISK_MEDIUM),
            RISK_HIGH: sum(1 for value in predictions.values() if value == RISK_HIGH),
        }
        return CrimeRiskRunResult(
            predictions=predictions,
            officer_allocation=allocation,
            accuracy=accuracy,
            risk_counts=counts,
            class_balance=class_balance,
            warnings=warnings,
            fallback_used=fallback_used,
        )

    def run(self, city_graph) -> CrimeRiskRunResult:
        return self.predict_and_update(city_graph)

    def allocate_officers(self, pred_df: pd.DataFrame, total_officers: int = 10) -> dict[str, int]:
        if total_officers < 0:
            raise ValueError("total_officers must be non-negative.")

        ordered = pred_df.sort_values(
            by=["predicted_risk", "population_density"],
            ascending=[False, False],
        )
        allocation = {str(node_id): 0 for node_id in ordered["node_id"]}
        officers_left = total_officers

        high_risk_rows = ordered[ordered["predicted_risk"] == RISK_HIGH]
        for row in high_risk_rows.itertuples(index=False):
            if officers_left == 0:
                break
            allocation[str(row.node_id)] += 1
            officers_left -= 1

        if len(allocation) == 0:
            return allocation

        ranked_node_ids = [str(node_id) for node_id in ordered["node_id"]]
        cursor = 0
        while officers_left > 0:
            node_id = ranked_node_ids[cursor % len(ranked_node_ids)]
            allocation[node_id] += 1
            officers_left -= 1
            cursor += 1

        return allocation

    def _industrial_proximity(self, graph: nx.Graph, node_id: str) -> int:
        industrial_nodes = [
            node
            for node, attrs in graph.nodes(data=True)
            if str(attrs.get("location_type")) == "Industrial"
        ]
        if not industrial_nodes:
            return 10**6
        if node_id in industrial_nodes:
            return 0

        distances: list[int] = []
        for industrial_node in industrial_nodes:
            try:
                distance = nx.shortest_path_length(graph, source=node_id, target=industrial_node)
                distances.append(int(distance))
            except nx.NetworkXNoPath:
                continue
        if not distances:
            return 10**6
        return min(distances)

    def _model_features(self, df: pd.DataFrame) -> pd.DataFrame:
        raw = df[["population_density", "industrial_proximity", "location_type", "cluster_id"]].copy()
        return pd.get_dummies(raw, columns=["location_type"], dtype=float)

