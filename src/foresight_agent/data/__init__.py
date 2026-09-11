"""External trajectory ingestion, validation, and annotation support."""

from .external import ExternalTrajectory, load_trajectories, write_normalized_jsonl

__all__ = ["ExternalTrajectory", "load_trajectories", "write_normalized_jsonl"]
