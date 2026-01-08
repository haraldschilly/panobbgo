#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import scipy.cluster.hierarchy as hcluster
import numpy.random as random
import numpy

import pylab

pylab.ion()

data = random.randn(2, 200)

data[:, :100] += 10

data[:, 40] = [0, 8]
data[:, 80] = [9, 3]

thresh = 4
clusters = hcluster.fclusterdata(numpy.transpose(data), thresh, criterion="maxclust")
pylab.scatter(*data[:, :], c=clusters)
pylab.axis("equal")
title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
print(title)
pylab.title(title)
# pylab.draw()
pylab.ioff()
pylab.show()
