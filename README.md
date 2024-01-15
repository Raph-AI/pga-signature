# Principal Geodesic Analysis for time series encoded with signature features

Reference: <https://hal.science/hal-04392568>

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
SX = iisignature.sig(Xs, sig_level)

n_components = 6  # nb of Principal Geodesics

principal_directions = pga(SX, channels, sig_level, n_components=n_components)
t_principal_directions = tangent_pga(SX, channels, sig_level, n_components=n_components)
```
