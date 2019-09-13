#!/usr/bin/python3

import numpy as np

y = np.array([1, 2, 3, 4, 5, 6])
NEW_SHAPE = (3,2)
z = y.reshape(NEW_SHAPE)
x = np.max(z)
dim = np.where(z == x)
r = dim[0][0]
c = dim[1][0]