#!/usr/bin/env python
# some tests regarding selection from a population
# based on their fitness

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from future.builtins import str
from future.builtins import range
from random import random
from math import exp, sin
#from collections import Counter

alpha = .3

# make statistics
cnt = {}  # Counter()


def pop1(x):
    return random() * 2 ** ((100 - x) / 100.0) / 2


def pop2(x):
    return random() * sin(x / 100.0) ** 2


def pop3(x):
    return random()

for x in range(1000):
    # all fit-values are between 0 and 1
    fit1 = pop1(x)
    fit2 = pop2(x)
    fit3 = pop3(x)

    fits = [fit1, fit2, fit3]
    N = len(fits)

    # additive/laplacian smoothing:
    fits = [((_ * N) + alpha) / (N + alpha * N) for _ in fits]

    fsum = sum(fits)

    thld = random() * fsum

    t = 0.0
    for idx, f in enumerate(fits):
        t += f
        if t > thld:
            print(" " * idx, idx)
            cnt[str(idx)] = cnt.get(str(idx), 0) + 1
            # if idx % 100 == 0: print cnt
            break

print()
print(cnt)
