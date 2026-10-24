"""Thread-safe FAISS vector index manager with partition isolation and remove_ids support."""

import json
import os
import threading
from typing import Any

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]


class PartitionedFaissIndex:
    """
    Thread-safe FAISS index manager ensuring strict partition isolation.
    Uses IndexIDMap2 wrapping IndexFlatIP (inner product on L2-normalized vectors = cosine similarity).
    Supports vector deletion via remove_ids for LRU/TTL eviction and disk snapshots.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._lock = threading.RLock()
        self._partitions: dict[str, Any] = {}
        self._id_to_partition: dict[int, str] = {}

    def _ensure_partition_index(self, partition_key: str):
        if faiss is None:
            raise RuntimeError("faiss-cpu is not installed in current environment.")
        if partition_key not in self._partitions:
            flat_index = faiss.IndexFlatIP(self.dimension)
            id_map = faiss.IndexIDMap2(flat_index)
            self._partitions[partition_key] = id_map
        return self._partitions[partition_key]

    def add(self, partition_key: str, entry_id: int, vector: np.ndarray) -> None:
        """Add a single normalized float32 vector with 64-bit entry_id."""
        with self._lock:
            index = self._ensure_partition_index(partition_key)
            # Ensure 2D float32 array and 1D int64 id array
            vec = np.ascontiguousarray(vector, dtype=np.float32)
            if vec.ndim == 1:
                vec = np.expand_dims(vec, axis=0)
            id_arr = np.array([entry_id], dtype=np.int64)

            index.add_with_ids(vec, id_arr)
            self._id_to_partition[entry_id] = partition_key

    def search(
        self,
        partition_key: str,
        vector: np.ndarray,
        top_k: int = 1,
    ) -> tuple[int, float]:
        """
        Search top-1 in the specified partition index.
        Returns (matched_id, similarity_score). If empty or no match, returns (-1, 0.0).
        """
        with self._lock:
            if partition_key not in self._partitions:
                return -1, 0.0

            index = self._partitions[partition_key]
            if index.ntotal == 0:
                return -1, 0.0

            vec = np.ascontiguousarray(vector, dtype=np.float32)
            if vec.ndim == 1:
                vec = np.expand_dims(vec, axis=0)

            distances, indices = index.search(vec, top_k)
            top_id = int(indices[0][0])
            top_sim = float(distances[0][0])

            if top_id < 0:
                return -1, 0.0
            return top_id, top_sim

    def remove(self, entry_id: int) -> bool:
        """Remove a vector by entry_id across partitions."""
        with self._lock:
            partition_key = self._id_to_partition.get(entry_id)
            if not partition_key or partition_key not in self._partitions:
                return False

            index = self._partitions[partition_key]
            id_arr = np.array([entry_id], dtype=np.int64)
            num_removed = index.remove_ids(id_arr)
            del self._id_to_partition[entry_id]
            return num_removed > 0

    def remove_many(self, entry_ids: list[int]) -> int:
        """Batch remove vectors by entry_id."""
        if not entry_ids:
            return 0
        total_removed = 0
        with self._lock:
            partition_groups: dict[str, list[int]] = {}
            for eid in entry_ids:
                part = self._id_to_partition.get(eid)
                if part and part in self._partitions:
                    partition_groups.setdefault(part, []).append(eid)

            for part, ids in partition_groups.items():
                index = self._partitions[part]
                id_arr = np.array(ids, dtype=np.int64)
                total_removed += index.remove_ids(id_arr)
                for eid in ids:
                    self._id_to_partition.pop(eid, None)

        return total_removed

    def get_total_count(self) -> int:
        """Total number of indexed vectors across all partitions."""
        with self._lock:
            return sum(idx.ntotal for idx in self._partitions.values())

    def get_indexed_ids(self) -> set[int]:
        """Set of all active IDs in FAISS."""
        with self._lock:
            return set(self._id_to_partition.keys())

    def clear(self) -> None:
        """Clear all partitions and vector mappings."""
        with self._lock:
            self._partitions.clear()
            self._id_to_partition.clear()

    def save_snapshot(self, snapshot_dir: str) -> None:
        """Serialize all partition indices and ID map metadata to disk."""
        if faiss is None:
            return
        with self._lock:
            os.makedirs(snapshot_dir, exist_ok=True)
            partitions_list: list[str] = list(self._partitions.keys())
            meta = {
                "dimension": self.dimension,
                "partitions": partitions_list,
                "id_to_partition": {str(k): v for k, v in self._id_to_partition.items()},
            }
            with open(os.path.join(snapshot_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            for idx, part_key in enumerate(partitions_list):
                index_file = os.path.join(snapshot_dir, f"part_{idx}.index")
                faiss.write_index(self._partitions[part_key], index_file)

    def load_snapshot(self, snapshot_dir: str) -> bool:
        """Restore partition indices from disk snapshot."""
        if faiss is None:
            return False
        meta_file = os.path.join(snapshot_dir, "metadata.json")
        if not os.path.exists(meta_file):
            return False

        with self._lock:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.dimension = meta.get("dimension", self.dimension)
            self._partitions.clear()
            self._id_to_partition = {int(k): v for k, v in meta.get("id_to_partition", {}).items()}

            partitions: list[str] = meta.get("partitions", [])
            for idx, part_key in enumerate(partitions):
                index_file = os.path.join(snapshot_dir, f"part_{idx}.index")
                if os.path.exists(index_file):
                    self._partitions[part_key] = faiss.read_index(index_file)

            return True

    def reconcile_with_store(self, valid_store_ids: set[int]) -> int:
        """
        Reconcile FAISS indices against SQLite store IDs.
        Removes any orphaned FAISS vectors not present in the store.
        """
        with self._lock:
            faiss_ids = set(self._id_to_partition.keys())
            orphaned = list(faiss_ids - valid_store_ids)
            if orphaned:
                return self.remove_many(orphaned)
            return 0
