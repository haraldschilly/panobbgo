#!/usr/bin/env python
from scipy.optimize import fmin_bfgs
import numpy as np

N = 5
ANZ = 50

def func(x, *params):
  a, b, c = params
  return a * x.dot(x) + x.dot(np.repeat(b, N)) + c

xx = np.array([ np.random.normal(1, 1.5, size=N) for i in range(ANZ) ])
# freely chosen
params = [ -1.5, 1, -3.1 ]
yy = np.array([ func(x, *params) for x in xx] )
yy += np.random.normal(0,.1,size=ANZ)

def residual(guess):
  #print "x  =", guess
  res = np.sum((func(x, *guess) - y)**2.0 for x, y in zip(xx,yy))
  return res

def gradient(guess):
  a, b, c = guess
  ret = np.empty(3)
  ret[0] = np.sum( 2.*(func(x, *guess) - y) * (x.dot(x))  for x,y in zip(xx,yy) )
  ret[1] = np.sum( 2.*(func(x, *guess) - y) * np.sum(x)   for x,y in zip(xx,yy) )
  ret[2] = np.sum( 2.*(func(x, *guess) - y) * 1.          for x,y in zip(xx,yy) )
  #print "x  =", guess
  #print "f' =", ret
  return ret

guess = np.random.normal(0,1,size=3)
sol = fmin_bfgs(residual, guess, fprime=gradient)

print "params:", params
print sol

y_est = func(xx[0], *sol)
print "f(%s) = %s [really: %s]" % (xx[0], y_est, yy[0])
