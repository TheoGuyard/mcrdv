use scorpion::optim::Solver;

#[test]
fn test_preset_levels_0_through_6() {
    for level in 0..=6 {
        let solver = Solver::preset(level);
        assert!(solver.params.limit_time == f64::INFINITY);
        assert!(solver.params.seed == 42);
    }
}

#[test]
#[should_panic(expected = "level")]
fn test_preset_invalid_level_panics() {
    let _ = Solver::preset(99);
}

#[test]
fn test_preset_higher_level_means_larger_population() {
    let s2 = Solver::preset(2);
    let s6 = Solver::preset(6);
    assert!(
        s6.generator.params.pop_init > s2.generator.params.pop_init,
        "Higher level should have larger initial population"
    );
    assert!(
        s6.params.limit_iter > s2.params.limit_iter,
        "Higher level should have more iteration limit"
    );
}
