"""Latent tree query utilities built from the hierarchical clustering notebook.

This module exposes :class:`LatentTreeIndex`, a lightweight search helper that
loads the centroid tree exported by ``notebooks/vec-treeing.ipynb`` and lets you
filter candidate clusters (and their member sequences) without running a full
k-NN search.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .classes import Tokenizer
from .encoder import encode
from .logger import setup_logger
from .model import VAEWithSurrogate

logger = setup_logger(__name__)


@dataclass
class TreeQueryResult:
    """Container describing a match returned by :class:`LatentTreeIndex`.

    Attributes
    ----------
    cluster_id:
        Identifier of the matched leaf cluster. ``None`` when the node does not
        correspond to an original leaf (e.g. when internal nodes are returned).
    distance:
        Cosine distance between the query vector and the node centroid. Because
        all centroids are L2-normalised, this equals ``1 - dot``.
    node_id:
        Integer identifier of the node inside ``nodes.parquet``.
    size:
        Number of leaf members represented by the node (1 for leaves).
    members:
        Optional :class:`pandas.DataFrame` holding the rows fetched from
        ``members.parquet`` for this cluster. Present only when the caller asks
        for sequences and the membership table is available.
    """

    cluster_id: Optional[str]
    distance: float
    node_id: int
    size: int
    members: Optional[pd.DataFrame] = None

    def sequences(self, column: str = "sequence") -> List[str]:
        """Return the raw sequences from ``members``.

        Parameters
        ----------
        column:
            Name of the column that stores raw sequences inside
            ``members.parquet``. Defaults to ``"sequence"``.

        Returns
        -------
        list[str]
            All sequences for this cluster. Returns an empty list when
            membership data was not requested or the column is missing.
        """

        if self.members is None:
            return []
        if column not in self.members.columns:
            raise KeyError(
                f"Column '{column}' is not present in the membership table. Available: {list(self.members.columns)}"
            )
        return self.members[column].tolist()


class LatentTreeIndex:
    """Search helper that filters clusters using the hierarchical centroid tree."""

    def __init__(
        self,
        nodes: pd.DataFrame,
        centroids: np.ndarray,
        members_path: Optional[Path] = None,
        members_columns: Optional[Sequence[str]] = None,
    ) -> None:
        if nodes.empty:
            raise ValueError("nodes DataFrame must not be empty")

        nodes = nodes.sort_values("node_id").reset_index(drop=True)
        expected_ids = np.arange(nodes.shape[0], dtype=np.int32)
        if not np.array_equal(nodes["node_id"].to_numpy(), expected_ids):
            raise ValueError("node_id column must contain consecutive integers starting from zero")

        if centroids.shape[0] != nodes.shape[0]:
            raise ValueError(
                "centroids shape mismatch: "
                f"expected {nodes.shape[0]} rows but received {centroids.shape[0]}"
            )

        self._nodes = nodes
        self._centroids = self._ensure_normalised(centroids.astype(np.float32, copy=False))
        self._members_path = Path(members_path) if members_path else None
        self._members_columns = list(members_columns) if members_columns else None

        self._leaf_mask = nodes["is_leaf"].to_numpy()
        self._cluster_ids = nodes["cluster_id"].replace("", np.nan).to_numpy()
        self._size = nodes["size"].to_numpy()
        self._root_id = int(nodes["node_id"].iloc[-1])

        if self._members_columns is not None and "cluster_id" not in self._members_columns:
            self._members_columns = ["cluster_id", *self._members_columns]

        logger.info("Loaded latent tree with %d nodes (%d leaves)", nodes.shape[0], self._leaf_mask.sum())

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        members_path: Optional[str | Path] = None,
        members_columns: Optional[Sequence[str]] = None,
    ) -> "LatentTreeIndex":
        """Load the index artefacts exported by ``vec-treeing.ipynb``.

        The notebook writes ``nodes.parquet`` and ``centroids_all.npy`` inside a
        sub-folder (``hclust/index`` in this repository). Providing the optional
        ``members_path`` enables eager retrieval of raw sequences.
        """

        directory = Path(directory)
        nodes_path = directory / "nodes.parquet"
        centroids_path = directory / "centroids_all.npy"
        if not nodes_path.exists():
            raise FileNotFoundError(f"Missing nodes metadata: {nodes_path}")
        if not centroids_path.exists():
            raise FileNotFoundError(f"Missing centroid matrix: {centroids_path}")

        nodes = pd.read_parquet(nodes_path)
        centroids = np.load(centroids_path)
        return cls(nodes, centroids, Path(members_path) if members_path else None, members_columns)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------
    def query_latent(
        self,
        latent: torch.Tensor | np.ndarray,
        *,
        top_k: Optional[int] = 20,
        max_distance: Optional[float] = None,
        min_cluster_size: int = 1,
        include_internal: bool = False,
        fetch_members: bool = False,
    ) -> List[TreeQueryResult]:
        """Filter clusters for a latent vector.

        Parameters
        ----------
        latent:
            1-D latent vector (``torch`` tensor or ``numpy`` array). The vector
            is automatically L2-normalised prior to distance evaluation.
        top_k:
            Optional number of clusters to return, ordered by cosine distance.
            ``None`` returns all matching clusters.
        max_distance:
            Optional cosine distance threshold. Subtrees whose centroid exceeds
            the threshold are pruned.
        min_cluster_size:
            Ignore leaves whose ``size`` is smaller than this value. This is
            useful when you want clusters with at least ``n`` members.
        include_internal:
            When ``True`` internal nodes that satisfy the distance threshold are
            also returned. By default only leaves are emitted.
        fetch_members:
            When ``True`` the method attempts to read the rows corresponding to
            the selected clusters from ``members.parquet``. The path must be
            provided at construction time.
        """

        vector = self._to_numpy(latent)
        vector = self._ensure_normalised(vector)

        # Compute cosine distance to every node centroid. The tree is small
        # (875 nodes), so a dense computation is trivial.
        dot = self._centroids @ vector
        distances = 1.0 - dot

        mask = self._leaf_mask | include_internal
        mask &= self._size >= max(1, min_cluster_size)
        if max_distance is not None:
            mask &= distances <= max_distance

        candidate_indices = np.flatnonzero(mask)
        if candidate_indices.size == 0:
            logger.info("No clusters matched the provided thresholds")
            return []

        ordered = candidate_indices[np.argsort(distances[candidate_indices])]
        if top_k is not None:
            ordered = ordered[:top_k]

        member_tables: Dict[str, pd.DataFrame] = {}
        if fetch_members:
            cluster_ids = [str(self._cluster_ids[i]) for i in ordered if pd.notna(self._cluster_ids[i])]
            member_tables = self._load_members(cluster_ids)

        results = []
        for idx in ordered:
            cid = self._cluster_ids[idx]
            cid = None if pd.isna(cid) else str(cid)
            members = member_tables.get(cid) if cid is not None else None
            results.append(
                TreeQueryResult(
                    cluster_id=cid,
                    distance=float(distances[idx]),
                    node_id=int(idx),
                    size=int(self._size[idx]),
                    members=members,
                )
            )
        return results

    def query_sequence(
        self,
        sequence: str,
        model: VAEWithSurrogate,
        tokenizer: Tokenizer,
        max_len: int,
        **kwargs,
    ) -> List[TreeQueryResult]:
        """Encode ``sequence`` with ``model`` and run :meth:`query_latent`."""

        latent = encode(model, sequence, tokenizer, max_len)
        return self.query_latent(latent, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().to(dtype=torch.float32, device="cpu").numpy()
        if x.ndim != 1:
            raise ValueError("latent vector must be 1-D")
        return x.astype(np.float32, copy=False)

    @staticmethod
    def _ensure_normalised(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            nrm = np.linalg.norm(x)
            if nrm == 0:
                raise ValueError("Zero-norm vector cannot be normalised")
            return x / nrm
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm = np.clip(nrm, 1e-9, None)
        return x / nrm

    def _load_members(self, cluster_ids: Sequence[str]) -> Dict[str, pd.DataFrame]:
        if not cluster_ids:
            return {}
        if self._members_path is None:
            raise FileNotFoundError("members_path was not provided when building the index")

        try:
            import pyarrow.dataset as ds
        except ImportError as exc:  # pragma: no cover - dependency check
            raise RuntimeError("pyarrow is required to read members.parquet") from exc

        dataset = ds.dataset(str(self._members_path), format="parquet")
        filt = ds.field("cluster_id").isin(list(cluster_ids))
        columns = self._members_columns
        try:
            table = dataset.to_table(filter=filt, columns=columns)
        except FileNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - pyarrow raises specialised errors
            raise RuntimeError(
                f"Failed to load membership data from {self._members_path}: {exc}"
            ) from exc

        df = table.to_pandas()
        grouped = {
            str(cid): group.reset_index(drop=True)
            for cid, group in df.groupby("cluster_id", sort=False)
        }
        return grouped


__all__ = ["LatentTreeIndex", "TreeQueryResult"]
