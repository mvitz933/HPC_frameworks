import sys
import numpy as np
import pylab as plt

argv=sys.argv
if len(argv)<2:
    print "Usage: python contour.py filename"
else:
    try:
        fname=argv[1]
    except(TypeError,ValueError):
        print "Input error"

data=np.loadtxt(fname,skiprows=1, unpack=False)

fig=plt.figure()
plt.contourf(data)
plt.colorbar()
plt.show()
