import numpy as np
import pylab as pl
from twoparticlesystem import TwoParticleSystem

#plot the full energy spectrum

N=100
n=15
es=np.zeros((n,N))
gs=-1/np.linspace(-10,0,N+1)[0:-1]

relative_state_colors=['g','b','r','k','m']
non_interacting_color='y-'
center_of_mass=['-','--',':','-.','--h','--s']
qns=[]
for j in range(n):
    qns.append(TwoParticleSystem.from_absolute_index_repulsive(j))

for i,g in enumerate(gs):
    T=TwoParticleSystem(g)
    for j in range(n):
        es[j,i]=T.energies[qns[j][0][0]]+0.5+qns[j][0][1]

for i in range(n):
    if qns[i][0][0]%2==1:
        pl.plot(-1/gs,es[i,:],non_interacting_color)
    else:
        pl.plot(-1/gs,es[i,:],relative_state_colors[qns[i][0][0]//2]+center_of_mass[qns[i][0][1]],markersize=3)

pl.title('Total energy spectrum')
pl.xlabel('$-1/g$')
pl.show()
