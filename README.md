# TwoParticlesContactInteraction
Python module for computing energies and wavefunctions for two particles in harmonic well with contact interaction.

The motion of the particles can be separated into the relative motion and the center of mass motion. The center of mass motion is not affected by the interaction (thus always given by that of a free harmonic oscillator), while the relative motion is affected by the interaction (although the odd states are not). Thus a state can either be specified by a relative motion quantum number n_rel and a center of mass quantum number n_cm.

## How to use it

First initialize a TwoParticleSystem object with interaction strength g:
```
from twoparticlesystem import TwoParticleSystem
T=TwoParticleSystem(10)
```
initializes a system with g=10.

Access energies by simply typing
```

e0=T.energies[0]
print(e0)

> 1.3506879041124322
```

These are the energies for the relative motion, which is the same as for one particle with a background potential consisting of a harmonic well and a delta function at the origin. If the energy value is being accessed for the first time, they will be computed. Later accesses will use a previously computed value. Energies also supports the iterator protocol:
```
for e in T.energies:
    print(e)
    if e>10:
        break
        
>1.3506879041124322
>1.5
>3.277707307848342
>3.5
>5.226446887193341
```

Note that the for loop must be manually stopped. The wavefunction for the n:th state for the relative motion can be accessed with the relative_wavefunction method, specifying n and the values x to compute the wavefunction for. 
```
import numpy as np
x=np.linspace(-1,1,5)
y=T.relative_wavefunction(0,x)
print(y)
> [0.63052561 0.53591595 0.13932756 0.53591595 0.63052561]
```
Total wavefunction of two coordinates can be accessed using absolute_wavefunction, specifying the relative motion quantum number n_rel and the center of motion quantum number n_cm, as well as the points x and y on which to evaluate the wavefunction. One particle densities are computed via the density method, specifying n_rel, n_cm and a set of points x.
