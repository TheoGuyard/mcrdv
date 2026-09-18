"""Sequence layer partitioning the catalogue by right ascension in one sweep."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import njit

from ..kernel import JIT, JIT_INLINE, KERNEL, SequenceKernel, schedule_evaluate
from ..orbit import TAU
from ..problem import DEPOT
from ..planner import SequenceLayer


class ClusterContext(NamedTuple):
    # Right ascension of each state at the catalogue epoch [rad].
    raan: np.ndarray
    # Number of chasers, one bucket each.
    num_chasers: int
    # Maximum number of targets a chaser collects.
    max_load: int


@njit(**JIT_INLINE)
def _angular_distance(a, b):
    """Distance between angles in radians."""
    d = abs(a - b) % TAU
    return min(d, TAU - d)


@njit(**JIT)
def _medoids(by_raan, k):
    """Seed `k` medoids spread evenly along the RAAN sweep."""
    n = by_raan.size
    p = np.empty(k, dtype=np.int64)
    for s in range(k):
        p[s] = n - 1 if s + 1 == k else round((n - 1) * s / max(k - 1, 1))
    return by_raan[np.unique(p)]


@njit(**JIT)
def _assign(raan, by_raan, medoids, max_load):
    """Assign each target to its nearest medoid, respecting the load limit.

    Returns the medoid of each state (`-1` for the depot) and the targets in
    the order they were assigned.
    """
    n = raan.size
    m = medoids.size
    count = by_raan.size
    counts = np.zeros(m, dtype=np.int64)
    bucket_of = np.full(n, -1, dtype=np.int64)
    assigned = np.empty(count, dtype=np.int64)
    num_assigned = 0

    # Every (target, medoid) pair, laid out by target index then medoid, so
    # that a stable sort on the distance alone leaves the cheapest pairs first
    # and resolves ties by target index, then by medoid.
    distance = np.empty(count * m)
    for d in range(1, n):
        for mi in range(m):
            distance[(d - 1) * m + mi] = _angular_distance(
                raan[d], raan[medoids[mi]]
            )

    for at in np.argsort(distance, kind="mergesort"):
        d = at // m + 1
        mi = at % m
        if bucket_of[d] >= 0 or counts[mi] >= max_load:
            continue
        bucket_of[d] = mi
        counts[mi] += 1
        assigned[num_assigned] = d
        num_assigned += 1

    # A target that found every cluster full falls back on the nearest one
    # with room left, or simply the nearest one when none has any.
    for d in by_raan:
        if bucket_of[d] >= 0:
            continue
        best = 0
        best_full = counts[0] >= max_load
        best_distance = _angular_distance(raan[d], raan[medoids[0]])
        for mi in range(1, m):
            full = counts[mi] >= max_load
            dist = _angular_distance(raan[d], raan[medoids[mi]])
            if (best_full and not full) or (
                full == best_full and dist < best_distance
            ):
                best, best_full, best_distance = mi, full, dist
        bucket_of[d] = best
        counts[best] += 1
        assigned[num_assigned] = d
        num_assigned += 1

    return bucket_of, assigned


@njit(**JIT)
def _cluster(raan, num_chasers, max_load):
    """Target buckets, one per chaser, each swept in RAAN.

    Returns `(offsets, members)`: bucket `c` is `members[offsets[c]:offsets[c + 1]]`.
    Non-empty buckets come first; the rest are empty.
    """
    count = raan.size - 1
    by_raan = 1 + np.argsort(raan[1:], kind="mergesort")
    offsets = np.zeros(num_chasers + 1, dtype=np.int64)

    # Enough chasers for everyone: one target each, in sweep order.
    if num_chasers >= count:
        for c in range(num_chasers):
            offsets[c + 1] = min(c + 1, count)
        return offsets, by_raan

    medoids = _medoids(by_raan, num_chasers)
    bucket_of, assigned = _assign(raan, by_raan, medoids, max_load)

    # Group the targets by medoid in the order they were assigned, then sweep
    # each group in RAAN; the stable sort keeps assignment order on ties.
    m = medoids.size
    sizes = np.zeros(m, dtype=np.int64)
    for d in assigned:
        sizes[bucket_of[d]] += 1
    starts = np.zeros(m + 1, dtype=np.int64)
    starts[1:] = np.cumsum(sizes)
    grouped = np.empty(count, dtype=np.int64)
    fill = starts[:-1].copy()
    for d in assigned:
        grouped[fill[bucket_of[d]]] = d
        fill[bucket_of[d]] += 1

    # There are at most as many medoids as chasers, so dropping the empty
    # groups and padding with empty buckets never has to merge any two.
    members = np.empty(count, dtype=np.int64)
    at = 0
    c = 0
    for mi in range(m):
        group = grouped[starts[mi] : starts[mi + 1]]
        if group.size == 0:
            continue
        group = group[np.argsort(raan[group], kind="mergesort")]
        members[at : at + group.size] = group
        at += group.size
        c += 1
        offsets[c] = at
    offsets[c + 1 :] = at
    return offsets, members


@njit(**KERNEL)
def _cluster_sequence_evaluate(kernel):
    """Cluster targets, then schedule each bucket from the departure orbit."""
    context = kernel.context
    offsets, members = _cluster(
        context.raan, context.num_chasers, context.max_load
    )

    num_plans = offsets.size - 1
    size = members.size + num_plans
    plan_offsets = np.zeros(num_plans + 1, dtype=np.int64)
    states = np.empty(size, dtype=np.int64)
    depart = np.empty(size)
    arrive = np.empty(size)
    costs = np.empty(size)
    cost = np.empty(num_plans)
    time = np.empty(num_plans)

    # Build the plan for each chaser.
    for c in range(num_plans):
        n = offsets[c + 1] - offsets[c] + 1
        seq = np.empty(n, dtype=np.int64)
        seq[0] = DEPOT
        for p in range(1, n):
            seq[p] = members[offsets[c] + p - 1]

        plan_cost, plan_time, plan_depart, plan_arrive, plan_costs = (
            schedule_evaluate(kernel.schedule, seq)
        )
        cost[c] = plan_cost
        time[c] = plan_time

        lo = plan_offsets[c]
        for p in range(n):
            states[lo + p] = seq[p]
            depart[lo + p] = plan_depart[p]
            arrive[lo + p] = plan_arrive[p]
            costs[lo + p] = plan_costs[p]
        plan_offsets[c + 1] = lo + n

    return plan_offsets, states, depart, arrive, costs, cost, time


class ClusterSequence(SequenceLayer):
    """Sequence layer based on a capacitated clustering of the target RAAN."""

    def _kernel(self) -> SequenceKernel:
        problem = self.problem
        context = ClusterContext(
            np.array([state.r for state in problem.states]),
            problem.num_chasers,
            problem.max_load,
        )
        return SequenceKernel(context, _cluster_sequence_evaluate)
