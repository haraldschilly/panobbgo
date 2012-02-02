#!/usr/bin/env python
# just testing basic parallelization that will be used in the actual project

def func(x):
  'x is either a number or a list/vector of numbers'
  import time
  from random import random
  from numpy import pi, sum
  time.sleep(1 + 5 * random())
  return sum(x) * pi


from IPython.parallel import Client
c = Client(profile='unicluster')

lb = c.load_balanced_view()

for i in range(100):
  pass
