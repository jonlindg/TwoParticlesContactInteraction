import numpy as np
import pylab as pl
from twoparticlesystem import TwoParticleSystem

N=100
n=7
es=np.zeros((n,N))
gs=-1/np.linspace(-10,0,N+1)[0:-1]

for i,g in enumerate(gs):
    T=TwoParticleSystem(g)
    for j in range(n):
        es[j,i]=T.energies[j]

for i in range(n):
    pl.plot(-1/gs,es[i,:])
pl.title('Energy states of the relative motion')
pl.xlabel('$-1/g$')
pl.show()
