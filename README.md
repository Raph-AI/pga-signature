# Principal Geodesic Analysis for time series encoded with signature features


Reference: Raphael Mignot, Marianne Clausel, Konstantin Usevich. Principal Geodesic Analysis for time series encoded with signature features. 2024. (hal-04392568) <https://hal.science/hal-04392568>

## How to use

```python
import numpy as np
import iisignature

# Personal libraries
from pga import pga
from pga import tangent_pga

batch = 3     # nb of time series
stream = 6    # length of time series
channels = 2  # nb of dimensions of time series
X = np.random.rand(batch, stream, channels)

sig_level = 5
SX = iisignature.sig(X, sig_level)

n_components = 6  # nb of Principal Geodesics

principal_directions = pga(SX, channels, sig_level, n_components=n_components)
t_principal_directions = tangent_pga(SX, channels, sig_level, n_components=n_components)
```

## Get projections of data onto PG1, PG2, etc.

```python
from scipy.optimize import fsolve

# Personal libraries
from pga import recenter_group_elems
from pga import d2h
from utils import depth_inds

inds = depth_inds(channels, depth)
projections = {} 
projections_t = {}
for K in range(n_components):
    optimized_v = principal_directions[K]
    optimized_v_t = t_principal_directions[K]
    SXk = recenter_group_elems(SX, channels, depth, inds)
    tis = []
    tis_t = []
    for i in range(len(SX)):
        d2h_t = lambda t: d2h(t, optimized_v, SXk[i], depth, channels)
        initial_guess_t = 0.
        optimized_ti = fsolve(d2h_t, initial_guess_t)[0]
        tis.append(optimized_ti)
        d2h_t = lambda t: d2h(t, optimized_v_t, SXk[i], depth, channels)
        initial_guess_t = 0.
        optimized_ti_t = fsolve(d2h_t, initial_guess_t)[0]
        tis_t.append(optimized_ti_t)
        # sig_proj = group_exp(optimized_ti*optimized_v, depth, inds)
    tis = np.array(tis)
    tis_t = np.array(tis_t)
    projections[K] = tis
    projections_t[K] = tis_t

means = [np.mean(projections[k]) for k in range(n_components)]
means_t = [np.mean(projections_t[k]) for k in range(n_components)]

var = [np.var(projections[k]) for k in range(n_components)]
var_t = [np.var(projections_t[k]) for k in range(n_components)]

print(means)   # should be close to zero
print(means_t) # should be close to zero
print()
print(var)
print(var_t)
```
