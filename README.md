# Principal Geodesic Analysis for time series encoded with signature features

## How to use

```python
import numpy as np
import iisignature

# Personal libraries
from pga import pga
from pga import tangent_pga

batch = 3
stream = 6
channels = 2 
X = np.random.rand(batch, stream, channels)

sig_level = 5
SX = iisignature.sig(Xs, sig_level)

n_components = 6

principal_directions = pga(SX, channels, sig_level, n_components=n_components)
t_principal_directions = tangent_pga(SX, channels, sig_level, n_components=n_components)
```