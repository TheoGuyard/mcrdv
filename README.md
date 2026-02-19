# SCORPION

[![Rust](https://img.shields.io/badge/Rust-1.91+-black?logo=rust&logoColor=white)](https://github.com/TheoGuyard/scorpion/blob/main/LICENSE)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://github.com/TheoGuyard/scorpion/blob/main/LICENSE)


**S**equencing via **C**ombinatorial **O**ptimization for **R**endezvous **P**lanning and **I**ntelligent **O**rbital **N**avigation



`scorpion` is a solver for coordinating **space debris remediation** with a **fleet of chasers** spacecraft. 
Given a maximum **mission time** and **load** limit per chaser, it constructs routes to collect debris while optimizing the total fuel consumption (delta-v in `m/s`).
The codebase is organized into the following modules:
- `io`: Input/output utilities for reading debris data and writing solutions.
- `orbit`: Physical oracle to compute transfer costs and times between orbital objects.
- `optim`: Engine based on the Hybrid Genetic Search (HGS) method to optimize chaser routes.


## Quickstart

`scorpion` is written in [Rust](https://www.rust-lang.org/). It can be downloaded and built as follows:

```bash
git clone https://github.com/TheoGuyard/scorpion.git
cd scorpion
cargo build
```

Tests can be run with 

```bash
cargo test --
```

to check that everything is working correctly. The main executable can be used as follows:

```bash
cargo run -- <debris.csv> [options]
```

Details on available options can be listed with `cargo run -- --help`.
Examples of library usage can be run as

```bash
cargo run --example <filename>  
```

where `<filename>` is the name of one of the files in the [example](examples/) folder (without the `.rs` extension).


## I/O format

The input of `scorpion` is a `.csv` file with one row per debris object, represented in [Keplerian elements](https://en.wikipedia.org/wiki/Kepler_orbit). The first line must be the following header `a[m],e[prop],i[rad],Omega[rad],omega[rad],theta[rad]`, and each subsequent line contains 6 comma-separated floating-point values for the following parameters with their respective units:
* `a[m]`: semi-major axis in meters
* `e[prop]`: eccentricity as a proportion in [0,1]
* `i[rad]`: inclination in radians
* `Omega[rad]`: right ascension of ascending node  in radians
* `omega[rad]`: argument of perigee in radians
* `theta[rad]`: true anomaly in radians

See the [data](data/) folder for example input files.
When the `--output-path` option is provided, a plain-text solution file describing the best solution found is written to the specified path.


## Solver parameters

When using the main executable, the solver parameters are set via the `--level` flag. This sets them with a given aggressiveness level between `0` (fast, low quality) and `6` (slow, high quality).
Finer control on the parameters can be achieved when calling the `scorpion` library directly within code.
See the [Params](src/optim/params.rs) struct for details on the available parameters, and the [examples](examples/) folder for example usage of the library.
