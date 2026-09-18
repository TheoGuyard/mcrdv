"""Sequence layer: which targets each chaser collects, and in which order."""

from .cluster import ClusterSequence
from .hgs import HgsSequence

__all__ = ["ClusterSequence", "HgsSequence"]
