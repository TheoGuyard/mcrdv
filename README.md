# mcrdv — Multi-Chaser Rendezvous

`mcrdv` is a **multi-chaser rendrezvous** mission planner: given a
orbital states to visit and a swarm of chaser spacecraft, it decides *which*
chaser visit *which* state, *in what order*, and *when* it departs, so that
all orbital states are visited at the lowest possible mission cost.

The package is:
- **Efficient -** Datasets up to thousands of orbital states and swarms with tens of chasers can be solved within minutes.
- **Flexible -** Custom implementation can be used to define the underlying orbital dynamics, as well as the mission operational constraints and objective.
- **Modular -** The solver is built as a stack of layers to construct the mission plan, each of which can be replaced with a custom implementation.

## Quickstart

```rust
use std::rc::Rc;
use mcrdv::{centroid, format_mission, read_debris, DAY};
use mcrdv::{DpSchedule, HgsSequence, Problem, QlawTransfer, Solver};

// Load a debris catalogue.
let path = "data/iridium33.csv";
let debris = read_debris(path);
println!("Loaded {} debris from {path}", debris.len());

// Index 0 is the orbit the chasers depart from, by convention.
let mut states = vec![centroid(&debris)];
states.extend(debris);

// Define the instance and its operational limits.
let problem = Rc::new(Problem::new(
    states,
    15,           // chasers in the swarm
    365.0 * DAY,  // per-chaser mission duration limit
    10,           // per-chaser debris capacity
));
println!("{problem}");

// Build the solver layer stack from the bottom up.
let transfer = QlawTransfer::new();
let schedule = DpSchedule::new(transfer);
let sequence = HgsSequence::new(schedule);
let mut solver = Solver::new(sequence);

// Solver the problem
let mission = solver.solve(&problem);

println!("{}", format_mission(&problem, &mission));
```

A CSV with one debris per row, six Keplerian elements. The header is mandatory
and checked:

```
a[m],e[prop],i[rad],r[rad],o[rad],t[rad]
7164040.5518,0.0019,1.5079,2.8765,0.8909,5.3923
```

Semi-major axis `a` [m], eccentricity `e`, inclination `i` [rad], RAAN `r`
[rad], argument of perigee `o` [rad], true anomaly `t` [rad]. The secular $J_2$
drift rates are derived on construction and carried through propagation.

Five catalogues ship in `data/`: `odrc` (13), `cosmos1408` (4), `iridium33`
(110), `cosmos2251` (583) and `fengyun1C` (1847).

## Extending

Implement the trait for the layer you want to replace and nest it in the stack:

```rust
use mcrdv::TransferLayer;

struct MyTransfer { /* ... */ }

impl TransferLayer for MyTransfer {
    fn initialize(&mut self, problem: &Problem) { /* ... */ }
    fn evaluate(&self, i: usize, j: usize, t: f64) -> (f64, f64) { /* ... */ }
    fn evaluate_lb(&self, i: usize, j: usize, t: f64) -> (f64, f64) { /* ... */ }
    fn hints(&self, i: usize, j: usize, t: f64) -> Vec<f64> { /* ... */ }
}
```

## Tests

Tests can be run as follows:

```bash
cargo test --release
cargo clippy --all-targets
```

## License

This project is distributed under the [MIT license](LICENSE).
