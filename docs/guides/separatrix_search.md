# Separatrix search

One can find the separatrix properties such as an **area**, **spiral step**, and **extraction angle** using the function `get_separatrix_properties()`

## Example usage

```python
import toolbox as tb
import xtrack as xt

line = xt.Line(...)

res = tb.get_separatrix_properties(
    line, 
    septum = {
        "entrance": 'en_aper',
        "exit": "ex_aper"
    },
    ex_norm = 1.6e-5,
    twiss = line.twiss4d(),
)
print(res)
```

An example output is 
```text
{'area': np.float64(7.926469222894832e-05), 'spiral_step': np.float64(0.002989079357082834), 'angle': np.float64(-0.007855679616157824)}
```
Compact list of parameters:

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `line` | `xt.Line` | Required | Xtrack beamline used for particle generation and tracking. |
| `septum` | `dict` | Required | Names of the septum entrance and exit aperture elements, given as `{"entrance": "...", "exit": "..."}`. |
| `ex_norm` | `float` | Required | Horizontal normalized emittance used to construct the test particles. |
| `test_range` | `list[float]` | `[-2.0, 0.0]` | Initial search interval in normalized horizontal phase space. The lower boundary must be unstable and the upper boundary stable. |
| `n_particles` | `int` | `2` | Number of test points per iteration. A value of `1` uses bisection; `2` uses a three-section search; larger values use a multi-section search. |
| `num_turns` | `int` | `500` | Number of turns used to determine whether a particle survives or is lost. |
| `ion` | `xt.Particles \| None` | `None` | Optional particle species and longitudinal coordinates. If omitted, the reference particle associated with `line` is used. |
| `precision` | `float` | `1e-4` | Maximum final distance between the unstable and stable initial normalized coordinates. |
| `shrinkage_strength` | `int \| str` | `"max"` | Controls how aggressively the interval is reduced in a multi-particle search. `"max"` retains the closest unstable–stable pair. An integer retains additional neighboring points. Ignored when `n_particles = 1`. |
| `max_iterations` | `int` | `2000` | Maximum number of search iterations before stopping. |
| `twiss` | `xt.TwissTable \| None` | `None` | Optional precomputed Twiss table. If omitted, it is calculated internally using `line.twiss4d()`. |
| `verbose` | `int` | `1` | Output level: `0` disables output, `1` displays progress, and values greater than `1` also produce diagnostic plots. |

**The search works as follows:**
- The separatrix is located by iteratively narrowing an interval in normalized horizontal phase space, x_norm. The initial interval must contain an unstable particle at its lower boundary and a stable particle at its upper boundary.

- At each iteration, particles are generated inside the current interval and tracked for a specified number of turns. Particles lost during tracking are classified as unstable, while surviving particles are classified as stable. The interval is then reduced to the region between the closest unstable and stable initial conditions.

- With `n_particles = 1`, it is a binary search., with `n_particles = 2` its a ternary search. For larger number of particles its a general multi-section bracketing search.

> In benchmarks performed with the SIS18 lattice, the three-section search (`n_particles = 2`) was the fastest on average.

*Since it does tracking under the hood, it could be slow depending on the parameters passed, or depending on what is the context `line` lives on.
The default parameters are empirically taken to provide result faster.*

## Separatrix matching
There is an action object that uses `get_separatrix_properties()` that can be used in the `xtrack` matching:

```python
action = tb.ActionSeparatrix(
    line, 
    septum = {
        "entrance": 'en_aper',
        "exit": "ex_aper"
    },
    ex_norm = 1.6e-5,
    twiss = line.twiss4d(),
    verbose = 0,
)
```

This allows to use the default `xtrack` matching as in the example below:

```python
opt = line.match(
    vary = [
        xt.VaryList(['ks.amp',], step = 1e-4, tag = "sext. amplitude"),
        xt.VaryList(['ks.phase'], step = 1e-3, tag = "sext. phase")
    ],
    targets = [
        action.target('spiral_step', 3e-3, tol = 2e-4),
        action.target('angle', -8e-3, tol = 2e-4)
    ]
)

opt.target_status()
```
In this example, `xtrack` attempts to match the separatrix `"spiral_step"` and `"angle"` by varying the two knobs `"ks.amp"` and `"ks.phase"`.