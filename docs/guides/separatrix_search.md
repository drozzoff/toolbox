One can find the separatrix properties such as an **area**, **spiral step**, and **extraction angle** using the function `get_separatrix_properties()`

## Example usage

```python
from toolbox import get_separatrix_properties
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
{'area': np.float64(7.926469222894832e-05), 'spiral_step': np.float64(0.002989079357082834), 'angle': np.float64(-0.007855679616157824)}`
```

Since it does tracking under the hood, it could be slow depending on the parameters passed, or depending on what is the context `line` lives on.
The default parameters are empirically taken to provide result faster.

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
    twiss = sis18ring.twiss4d(),
    verbose = 0,
)
```

This allows to use the default `xtrack` matching as in the example below:

```python
opt = sis18ring.match(
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
In this example, `xtrack` will attempt to match the `"spiral_step"` and `"angle"` of the separatrix using the 2 knobs `"ks.amp"`and `"ks.phase"`.