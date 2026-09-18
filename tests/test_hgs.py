"""Hybrid genetic search: configuration, population, moves and recombination."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from mcrdv import Planner
from mcrdv.problem import DEPOT
from mcrdv.schedule import DpSchedule
from mcrdv.planner import ScheduleLayer
from mcrdv.sequence.hgs import (
    Context,
    Crossover,
    Generation,
    HgsConfig,
    HgsSequence,
    Individual,
    Metric,
    Ordering,
    Population,
    Search,
    Subpopulation,
    _bound,
    _inter_relocate,
    _intra_relocate,
    _intra_swap,
    _sequence_key,
    _two_opt,
    _two_opt_star,
    broken_pair_distance,
    edge_recombination,
    giant_tour,
    order_crossover,
    pair_lb,
    selective_route_exchange,
    split,
)
from mcrdv.transfer import QlawTransfer


def context_for(problem, **overrides):
    config = HgsConfig.preset(2, verbose=False, seed=0, **overrides)
    schedule = DpSchedule(transfer=QlawTransfer())
    schedule.initialize(problem)
    context = Context(problem, schedule, config)
    context.build_neighbors()
    return context


@pytest.fixture(scope="module")
def ctx(iridium):
    problem = iridium
    return context_for(problem)


# ============================== configuration ============================= #


def test_presets_get_deeper_with_level():
    iters = [HgsConfig.preset(k).max_iter for k in range(7)]
    assert iters == sorted(iters)
    assert HgsConfig.preset(6).pop_max > HgsConfig.preset(0).pop_max


def test_an_override_beats_the_preset():
    assert HgsConfig.preset(3, max_iter=7).max_iter == 7
    assert (
        HgsConfig.preset(3, max_iter=7).pop_init
        == HgsConfig.preset(3).pop_init
    )


def test_preset_level_is_checked():
    with pytest.raises(ValueError, match="between 0 and 6"):
        HgsConfig.preset(7)


def test_enums_accept_their_plain_strings():
    config = HgsConfig(ordering="raan", crossover="erx", generation="random")
    assert config.ordering is Ordering.RAAN
    assert config.crossover is Crossover.ERX
    assert config.generation is Generation.RANDOM


def test_config_is_immutable_but_derivable():
    config = HgsConfig()
    with pytest.raises(Exception):
        config.seed = 3
    assert dataclasses.replace(config, seed=3).seed == 3


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(pop_min=30, pop_max=10), "pop_min"),
        (dict(pop_max=0, pop_min=0), "pop_max"),
        (dict(nb_neighbors=-1), "nb_neighbors"),
        (dict(target_ratio=1.5), "target_ratio"),
        (dict(penalty_min=10.0, penalty_max=1.0), "penalty_min"),
    ],
)
def test_impossible_configurations_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        HgsConfig(**kwargs)


# =============================== individual =============================== #


def test_individual_derives_cost_violation_and_links(iridium):
    problem = iridium
    ctx = context_for(problem)
    individual = ctx.assemble([[3, 9, 4], [7]])

    assert math.isclose(individual.cost, sum(p.cost for p in individual.plans))
    assert individual.active == 2
    assert len(individual.plans) == problem.num_chasers
    # 3 -> 9 -> 4 is one chain, 7 sits alone; the depot closes both ends.
    assert (individual.pred[3], individual.succ[3]) == (DEPOT, 9)
    assert (individual.pred[9], individual.succ[9]) == (3, 4)
    assert (individual.pred[4], individual.succ[4]) == (9, DEPOT)
    assert (individual.pred[7], individual.succ[7]) == (DEPOT, DEPOT)


def test_violation_counts_the_excess_over_each_limit(iridium):
    problem = iridium
    ctx = context_for(problem)
    overloaded = ctx.assemble([list(range(1, problem.max_load + 4))])
    assert overloaded.violation[0] == 3
    assert not overloaded.feasible


def test_a_feasible_individual_reports_no_violation(ctx):
    individual = ctx.assemble([[3, 9], [7]])
    assert individual.violation == (0.0, 0.0)
    assert individual.feasible


def test_distance_to_itself_is_zero_and_to_a_different_plan_is_not(ctx):
    a = ctx.assemble([[3, 9, 4], [7]])
    b = ctx.assemble([[3, 9, 4], [7]])
    c = ctx.assemble([[7, 4], [3, 9]])
    n = ctx.problem.num_targets
    assert a.distance(b, n) == 0.0
    assert a.distance(c, n) > 0.0


def test_broken_pair_distance_is_bounded():
    pred = np.zeros(5, dtype=np.int64)
    succ = np.zeros(5, dtype=np.int64)
    other = np.array([0, 2, 3, 4, 1], dtype=np.int64)
    d = broken_pair_distance(pred, succ, other, other, 4)
    assert 0.0 <= d <= 2.0
    assert broken_pair_distance(pred, succ, pred, succ, 0) == 0.0


# ================================= metric ================================= #


def test_penalties_start_positive_and_bounded(ctx):
    load, time = ctx.metric.penalties
    assert load > 0.0 and time > 0.0
    assert ctx.config.penalty_min <= load <= ctx.config.penalty_max


def test_a_feasible_plan_is_charged_nothing_extra(ctx):
    plan = ctx.plan([3, 9])
    assert math.isclose(ctx.value(plan), plan.cost, rel_tol=1e-15)


def test_an_overloaded_plan_is_charged_for_the_excess(ctx):
    plan = ctx.plan(list(range(1, ctx.problem.max_load + 3)))
    assert ctx.value(plan) > plan.cost
    assert math.isclose(
        ctx.value(plan) - plan.cost, 2 * ctx.metric.penalties[0], rel_tol=1e-9
    )


def test_adapt_raises_a_weak_penalty_and_lowers_a_strong_one(ctx):
    metric = Metric(ctx.problem, ctx.schedule, ctx.config)
    before = metric.penalties
    metric.adapt([0.0, 1.0])  # first limit missed, second always met
    assert metric.penalties[0] > before[0]
    assert metric.penalties[1] < before[1]


def test_penalties_stay_inside_their_range(ctx):
    metric = Metric(ctx.problem, ctx.schedule, ctx.config)
    for _ in range(200):
        metric.adapt([0.0, 0.0])
    assert all(p <= ctx.config.penalty_max for p in metric.penalties)
    for _ in range(400):
        metric.adapt([1.0, 1.0])
    assert all(p >= ctx.config.penalty_min for p in metric.penalties)


# =============================== population =============================== #


def test_a_subpopulation_trims_itself_back_to_the_minimum(ctx):
    config = dataclasses.replace(ctx.config, pop_min=3, pop_max=5)
    sub = Subpopulation(config, ctx.problem.num_targets)
    for k in range(6):
        sub.add(ctx.metric, ctx.assemble([[1 + k, 2 + k], [5 + k]]))
    assert len(sub) == 3
    assert len(sub.dist) == 3 and all(len(row) == 3 for row in sub.dist)


def test_scores_are_ranks_so_they_stay_in_range(ctx):
    sub = Subpopulation(ctx.config, ctx.problem.num_targets)
    for k in range(4):
        sub.add(ctx.metric, ctx.assemble([[1 + k, 2 + k], [5 + k]]))
    assert len(sub.score) == 4
    assert all(0.0 <= s <= 2.0 for s in sub.score)
    assert sub.best(ctx.metric) is min(sub.individuals, key=ctx.metric.value)


def test_a_clone_is_evicted_before_a_merely_bad_individual(ctx):
    config = dataclasses.replace(ctx.config, pop_min=2, pop_max=3, nb_elite=0)
    sub = Subpopulation(config, ctx.problem.num_targets)
    twin = [[1, 2], [5]]
    sub.add(ctx.metric, ctx.assemble(twin))
    sub.add(ctx.metric, ctx.assemble(twin))  # an exact clone
    sub.add(ctx.metric, ctx.assemble([[9, 4], [6]]))
    sub.add(ctx.metric, ctx.assemble([[7, 3], [8]]))
    assert len(sub) == 2
    # the survivors are no longer identical to one another
    assert (
        sub.individuals[0].distance(
            sub.individuals[1], ctx.problem.num_targets
        )
        > 0.0
    )


def test_the_population_files_by_feasibility(ctx):
    population = Population(ctx.config, ctx.problem.num_targets)
    population.add(ctx.metric, ctx.assemble([[1, 2], [5]]))
    population.add(
        ctx.metric, ctx.assemble([list(range(1, ctx.problem.max_load + 5))])
    )
    assert len(population.feasible) == 1
    assert len(population.infeasible) == 1
    assert len(population) == 2


def test_pick_returns_a_copy_of_a_held_individual(ctx):
    population = Population(ctx.config, ctx.problem.num_targets)
    assert population.pick(ctx.rng) is None
    for k in range(3):
        population.add(ctx.metric, ctx.assemble([[1 + k, 2 + k], [5 + k]]))
    drawn = population.pick(ctx.rng)
    held = population.feasible.individuals + population.infeasible.individuals
    assert drawn is not None
    assert any(drawn.cost == i.cost for i in held)
    assert all(drawn is not i for i in held)


def test_penalties_are_retuned_on_the_adaptation_window(ctx):
    config = dataclasses.replace(ctx.config, adapt_iter=2)
    metric = Metric(ctx.problem, ctx.schedule, config)
    population = Population(config, ctx.problem.num_targets)
    before = metric.penalties
    overloaded = ctx.assemble([list(range(1, ctx.problem.max_load + 5))])
    population.add(metric, overloaded)
    population.add(metric, overloaded)
    assert metric.penalties[0] > before[0]  # the load limit kept being missed


# ============================== local search ============================== #


def test_local_search_never_worsens_the_objective(ctx):
    for seed in range(4):
        ctx.rng = np.random.default_rng(seed)
        child = ctx.generate_random()
        before = sum(ctx.value(p) for p in child.plans)
        ctx.improve(child)
        after = sum(ctx.value(p) for p in child.plans)
        assert after <= before + 1e-9


def test_local_search_improves_a_random_solution(ctx):
    ctx.rng = np.random.default_rng(11)
    child = ctx.generate_random()
    before = sum(ctx.value(p) for p in child.plans)
    ctx.improve(child)
    assert sum(ctx.value(p) for p in child.plans) < before


def test_local_search_keeps_every_target_exactly_once(ctx):
    for seed in range(4):
        ctx.rng = np.random.default_rng(seed)
        child = ctx.generate_random()
        ctx.improve(child)
        visited = [d for plan in child.plans for d in plan.sequence[1:]]
        assert sorted(visited) == list(ctx.problem.targets)


def test_local_search_leaves_the_derived_fields_consistent(ctx):
    ctx.rng = np.random.default_rng(3)
    child = ctx.generate_random()
    ctx.improve(child)
    rebuilt = Individual(ctx.problem, child.plans)
    assert math.isclose(child.cost, rebuilt.cost, rel_tol=1e-12)
    assert child.violation == rebuilt.violation
    assert (
        child.feasible == rebuilt.feasible and child.active == rebuilt.active
    )
    assert np.array_equal(child.pred, rebuilt.pred)
    assert np.array_equal(child.succ, rebuilt.succ)


def test_the_searched_plans_are_the_schedule_layers_own(ctx):
    ctx.rng = np.random.default_rng(7)
    individual = ctx.generate_random()
    ctx.improve(individual)
    for plan in individual.plans:
        assert plan is ctx.schedule.evaluate(plan.sequence)


def test_the_bound_screen_reads_legs_only():
    bounds = np.where(np.eye(4, dtype=bool), 7.0, 100.0)
    assert (
        _bound(bounds, np.array([0, 1, 2, 3]), 4) == 300.0
    )  # never the diagonal
    assert _bound(bounds, np.array([2]), 1) == 0.0


def test_every_move_preserves_the_targets_it_touches(ctx):
    ctx.rng = np.random.default_rng(5)
    search = Search(ctx, ctx.generate_random())
    plans, work = search.plans, search.clocks.work
    penalties = ctx.metric.penalties

    def collected():
        return sorted(
            int(d)
            for s in range(plans.size.size)
            for d in plans.seq[s, 1 : plans.size[s]]
        )

    before = collected()
    for move, args in (
        (_intra_relocate, (0, 1)),
        (_intra_swap, (0, 1)),
        (_two_opt, (0, 1)),
        (_inter_relocate, (0, 1, 1, 1)),
        (_two_opt_star, (0, 1, 1, 1)),
    ):
        move(ctx.search, plans, work, penalties, *args)
        assert collected() == before


def test_a_sequence_is_priced_once_and_as_the_schedule_layer_prices_it(odrc):
    ctx = context_for(odrc)
    ctx.rng = np.random.default_rng(1)
    individual = ctx.generate_random()
    ctx.improve(individual)

    cache = ctx.search.cache
    assert len(cache) > 0
    for plan in individual.plans:
        sequence = np.asarray(plan.sequence, dtype=np.int64)
        # As `np.uint64`, or numba would type each half of the key by value.
        key = tuple(
            np.uint64(h) for h in _sequence_key(sequence, sequence.size)
        )
        assert cache[key] == (plan.cost, plan.time)

    # A plan already priced is not priced again.
    before = len(cache)
    ctx.improve(individual)
    assert len(cache) == before


# ============================== pair bounds =============================== #


def test_the_table_is_additive_over_the_legs(ctx):
    table = ctx.bounds
    assert table.shape == (ctx.problem.num_states,) * 2
    for sequence in (
        (DEPOT, 5),
        (DEPOT, 12, 3, 44, 7),
        (DEPOT, 100, 20, 60, 3),
    ):
        legs = sum(table[a, b] for a, b in zip(sequence, sequence[1:]))
        assert math.isclose(
            legs, ctx.schedule.evaluate_lb(sequence), rel_tol=1e-12
        )


def test_a_cell_is_the_schedule_layers_own_bound(ctx):
    for i, j in ((0, 1), (7, 3), (60, 60), (110, 22)):
        assert math.isclose(
            ctx.bounds[i, j], ctx.schedule.evaluate_lb((i, j)), rel_tol=1e-15
        )


def test_the_table_never_exceeds_a_real_plan(ctx):
    for sequence in ((DEPOT, 5, 17), (DEPOT, 12, 3, 44, 7)):
        legs = sum(ctx.bounds[a, b] for a, b in zip(sequence, sequence[1:]))
        assert legs <= ctx.schedule.evaluate(sequence).cost * (1 + 1e-12)


def test_the_threaded_fill_matches_the_serial_one(iridium):
    problem = iridium
    serial = pair_lb(_fresh_schedule(problem), problem, workers=1)
    threaded = pair_lb(_fresh_schedule(problem), problem)
    assert np.array_equal(serial, threaded)
    assert np.allclose(np.diag(serial), 0.0)


def test_any_schedule_layer_can_supply_the_table(iridium):
    # Derived from `evaluate_lb` alone, so a layer that publishes nothing else
    # still works -- the table is the search's device, not the scheduler's.
    problem = iridium

    class Minimal(ScheduleLayer):
        def _evaluate(self, sequence):
            raise AssertionError("the table must not need a real schedule")

        def _evaluate_lb(self, sequence):
            return float(len(sequence) - 1)

    layer = Minimal(QlawTransfer())
    layer.initialize(problem)
    table = pair_lb(layer, problem, workers=1)
    assert table.shape == (problem.num_states,) * 2
    assert (table == 1.0).all()


def _fresh_schedule(problem):
    schedule = DpSchedule(transfer=QlawTransfer())
    schedule.initialize(problem)
    return schedule


# =============================== crossover ================================ #


def test_a_giant_tour_lists_every_collected_target_once(ctx):
    ctx.rng = np.random.default_rng(1)
    individual = ctx.generate_greedy()
    tour = giant_tour(ctx, individual)
    collected = [d for p in individual.plans for d in p.sequence[1:]]
    assert sorted(tour) == sorted(collected)


@pytest.mark.parametrize("ordering", list(Ordering))
def test_every_ordering_yields_a_full_tour(iridium, ordering):
    problem = iridium
    ctx = context_for(problem, ordering=ordering)
    individual = ctx.generate_greedy()
    assert sorted(giant_tour(ctx, individual)) == list(problem.targets)


@pytest.mark.parametrize("operator", [order_crossover, edge_recombination])
def test_a_child_tour_is_a_permutation_of_its_parents(ctx, operator):
    ctx.rng = np.random.default_rng(2)
    a = giant_tour(ctx, ctx.generate_greedy())
    b = giant_tour(ctx, ctx.generate_random())
    for _ in range(5):
        child = operator(ctx, a, b)
        assert sorted(child) == sorted(a)


def test_srex_covers_every_target_of_both_parents(ctx):
    ctx.rng = np.random.default_rng(4)
    first = ctx.generate_greedy()
    second = ctx.generate_random()
    for _ in range(5):
        buckets = selective_route_exchange(ctx, first, second)
        assert sorted(d for b in buckets for d in b) == list(
            ctx.problem.targets
        )
        assert len(buckets) <= ctx.problem.num_chasers


def test_split_partitions_the_tour_in_order(ctx):
    ctx.rng = np.random.default_rng(6)
    tour = tuple(giant_tour(ctx, ctx.generate_greedy()))
    buckets = split(ctx, tour, ctx.problem.num_chasers)
    assert [d for b in buckets for d in b] == list(tour)  # order preserved
    assert len(buckets) <= ctx.problem.num_chasers


def test_split_into_one_bucket_returns_the_whole_tour(ctx):
    tour = tuple(range(1, 6))
    assert split(ctx, tour, 1) == [list(tour)]
    assert split(ctx, (), 3) == []


@pytest.mark.parametrize("operator", list(Crossover))
def test_every_crossover_operator_produces_a_complete_child(iridium, operator):
    problem = iridium
    ctx = context_for(problem, crossover=operator)
    for _ in range(3):
        ctx.insert(ctx.generate_greedy())
    child = ctx.crossover()
    visited = [d for p in child.plans for d in p.sequence[1:]]
    assert sorted(visited) == list(problem.targets)
    assert len(child.plans) == problem.num_chasers


# =============================== generation =============================== #


@pytest.mark.parametrize("strategy", list(Generation))
def test_every_generator_collects_the_whole_catalogue(iridium, strategy):
    problem = iridium
    ctx = context_for(problem, generation=strategy)
    individual = ctx.generate()
    visited = [d for p in individual.plans for d in p.sequence[1:]]
    assert sorted(visited) == list(problem.targets)
    assert len(individual.plans) == problem.num_chasers


def test_greedy_beats_random_on_cost(iridium):
    problem = iridium
    ctx = context_for(problem)
    greedy = ctx.generate_greedy()
    randoms = [ctx.generate_random() for _ in range(3)]
    assert greedy.cost < min(r.cost for r in randoms)


def test_neighbours_are_the_closest_peers_in_order(ctx):
    for u in (1, 17, 60):
        peers = ctx.neighbors[u]
        assert len(peers) == ctx.config.nb_neighbors
        assert u not in peers
        proximities = [ctx.proximity(u, v) for v in peers]
        assert proximities == sorted(proximities)
        others = [v for v in ctx.problem.targets if v != u and v not in peers]
        assert (
            max(proximities)
            <= min(ctx.proximity(u, v) for v in others) + 1e-12
        )


def test_no_neighbours_requested_means_no_neighbours(iridium):
    problem = iridium
    ctx = context_for(problem, nb_neighbors=0)
    assert all(not peers for peers in ctx.neighbors)


# ================================= layer ================================== #


def test_the_search_returns_a_complete_feasible_mission(iridium):
    problem = iridium
    config = HgsConfig.preset(2, seed=0, verbose=False)
    mission = Planner(HgsSequence(config=config)).solve(problem, verbose=False)
    assert len(mission.plans) == problem.num_chasers
    visited = [d for p in mission.plans for d in p.sequence[1:]]
    assert sorted(visited) == list(problem.targets)
    assert problem.is_feasible(mission)


def test_the_same_seed_gives_the_same_mission(iridium):
    problem = iridium

    def run(seed):
        config = HgsConfig.preset(2, seed=seed, verbose=False)
        return Planner(HgsSequence(config=config)).solve(
            problem, verbose=False
        )

    assert run(0).cost == run(0).cost
    assert [p.sequence for p in run(0).plans] == [
        p.sequence for p in run(0).plans
    ]


def test_the_search_beats_the_greedy_construction(iridium):
    problem = iridium
    ctx = context_for(problem)
    greedy = ctx.generate_greedy().cost
    config = HgsConfig.preset(
        3, seed=0, verbose=False, max_iter=40, max_nimp=10**9
    )
    mission = Planner(HgsSequence(config=config)).solve(problem, verbose=False)
    assert mission.cost < greedy


def test_a_deeper_preset_does_not_do_worse(iridium):
    problem = iridium
    costs = []
    for preset in (1, 2):
        config = HgsConfig.preset(preset, seed=0, verbose=False)
        costs.append(
            Planner(HgsSequence(config=config))
            .solve(problem, verbose=False)
            .cost
        )
    assert costs[1] <= costs[0] * (1 + 1e-9)


def test_the_trace_records_the_run(iridium, capsys):
    problem = iridium
    config = HgsConfig.preset(
        2, seed=0, verbose=False, log_save=True, log_iter=1
    )
    layer = HgsSequence(config=config)
    Planner(layer).solve(problem, verbose=False)
    capsys.readouterr()
    assert len(layer.trace) > 0
    assert all(entry.time >= 0.0 for entry in layer.trace)
    assert layer.trace[-1].num_feasible + layer.trace[-1].num_infeasible > 0
    # The incumbent's own feasibility is recorded, not just the population's
    # counts: a cost logged before any feasible plan exists is not comparable
    # with one logged after.
    assert all(isinstance(entry.feasible, bool) for entry in layer.trace)
    assert layer.trace[-1].feasible == (layer.trace[-1].num_feasible > 0)


def test_keeping_a_trace_is_independent_of_printing(iridium, capsys):
    problem = iridium

    # Quiet, but recorded: what an experiment wants.
    quiet = HgsSequence(
        config=HgsConfig.preset(
            2, seed=0, verbose=False, log_save=True, log_iter=1
        )
    )
    Planner(quiet).solve(problem, verbose=False)
    assert capsys.readouterr().out == ""
    assert len(quiet.trace) > 0

    # Printed, but not kept: what a person watching a run wants.
    loud = HgsSequence(
        config=HgsConfig.preset(
            2, seed=0, verbose=True, log_save=False, log_iter=1
        )
    )
    Planner(loud).solve(problem, verbose=False)
    assert capsys.readouterr().out != ""
    assert len(loud.trace) == 0


def test_a_trace_is_not_kept_by_default(iridium, capsys):
    problem = iridium
    layer = HgsSequence(config=HgsConfig.preset(2, seed=0, verbose=False))
    Planner(layer).solve(problem, verbose=False)
    capsys.readouterr()
    assert len(layer.trace) == 0


def test_a_time_budget_stops_the_search(iridium):
    problem = iridium
    config = HgsConfig.preset(6, seed=0, verbose=False, max_time=1.0)
    import time as _t

    t0 = _t.perf_counter()
    Planner(HgsSequence(config=config)).solve(problem, verbose=False)
    assert (
        _t.perf_counter() - t0 < 30.0
    )  # generous: the budget is checked per iteration


def test_a_fresh_layer_holds_no_context_until_it_is_initialized(iridium):
    # No guard stands in front of `evaluate`: callers reach the layer through
    # `Planner.solve`, which initializes the whole stack first.
    sequence = HgsSequence()
    assert sequence.context is None and sequence.schedule.problem is None
    sequence.initialize(iridium)
    assert sequence.context is not None
    assert sequence.context.problem is iridium
    assert sequence.schedule.problem is iridium


def test_the_layer_publishes_no_kernel(odrc):
    # Nothing sits above a sequence layer; the compiled parts are inside it.
    layer = HgsSequence(config=HgsConfig(verbose=False))
    layer.initialize(odrc)
    assert layer.kernel() is None


def test_with_preset_builds_the_layer_around_that_configuration():
    config = HgsConfig.preset(3, seed=7, verbose=False)
    sequence = HgsSequence(config=config)
    assert sequence.config.seed == 7 and sequence.config.verbose is False
