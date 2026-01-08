#!/usr/bin/env python

from __future__ import division
from __future__ import unicode_literals
import scipy.sparse as sp
import numpy as np

m = sp.dok_matrix((10, 100), dtype=np.float64)
full = float(m.shape[0] * m.shape[1])

sqrt3 = np.sqrt(3)
sign = 1

while len(m) / full <= 0.1:
    i = np.random.randint(m.shape[0])
    j = np.random.randint(m.shape[1])
    # m[i,j] = np.random.rand()
    if np.random.rand() < 0.5:
        m[i, j] = sqrt3
    else:
        m[i, j] = -sqrt3

print(m)
