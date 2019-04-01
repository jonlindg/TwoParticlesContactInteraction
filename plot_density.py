import numpy as np
import pylab as pl
from twoparticlesystem import TwoParticleSystem

N=100
n_rel=0
n_cm=0
g=1.0

x=np.append(np.linspace(-5,0,N)[:-1],-np.linspace(-5,0,N)[::-1])

T=TwoParticleSystem(g)


dens=T.density(x,n_rel,n_cm)
wf_rel=T.relative_wavefunction(n_rel,x)
pl.plot(x,dens,label='Density')
pl.plot(x,wf_rel,label='Relative wavefunction')
pl.legend()
pl.show()

