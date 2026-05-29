# SCORPION

[![Rust](https://img.shields.io/badge/Rust-1.91+-black?logo=rust&logoColor=white)](https://github.com/TheoGuyard/scorpion/blob/main/LICENSE)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://github.com/TheoGuyard/scorpion/blob/main/LICENSE)


**S**equencing via **C**ombinatorial **O**ptimization for **R**endezvous **P**lanning and **I**ntelligent **O**rbital **N**avigation



`scorpion` is a solver for coordinating **space debris remediation** with a **swarm of chasers** spacecraft. 
Given a cloud of debris and operational constraints, it constructs sequence of debris to be collected for each chaser while optimizing the mission cost.


## Quickstart

`scorpion` is written in [Rust](https://www.rust-lang.org/). It can be downloaded and run with the following commands:

```bash
git clone https://github.com/TheoGuyard/scorpion.git
cd scorpion
cargo run
```

which triggers the main entry point of the code in [src/main.rs](src/main.rs), defining the problem instance and solver parameters.


## I/O format

The input of `scorpion` is a `.csv` file with one row per debris object, represented in [Keplerian elements](https://en.wikipedia.org/wiki/Kepler_orbit). The first line must be the following header `a[m],e[prop],i[rad],r[rad],o[rad],t[rad]`, and each subsequent line contains 6 comma-separated floating-point values for the following parameters with their respective units:
* `a[m]`: semi-major axis in meters
* `e[prop]`: eccentricity as a proportion in [0,1]
* `i[rad]`: inclination in radians
* `r[rad]`: right ascension of ascending node  in radians
* `o[rad]`: argument of perigee in radians
* `t[rad]`: true anomaly in radians

See the [data](data/) folder for example input files.
