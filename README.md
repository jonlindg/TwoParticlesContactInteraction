# TwoParticlesContactInteraction
Python module for computing energies and wavefunctions for two particles in harmonic well with contact interaction.

First initialize a TwoParticleSystem object with -1/g=0.1:
```
T=TwoParticleSystem(-0.1)
```

Access energies by simply typing
```
e0=T.energies[0]
```

These are the energies for the relative motion, which is the same as for one particle with a background potential consisting of a harmonic well and a delta function at the origin. If the energy value is being accessed for the first time, they will be computed. Later accesses will use a previously computed value. Energies also supports the iterator protocol
```
for e in T.energies:
    print(e)
    if e>10:
        break
```

Note that the for loop must be manually stopped. Wavefunctions can be accessed with the relative_wavefunction method. Wavefunction of two coordinates in absolute coordinates can be accessed using absolute_wavefunction. One partilce density is computer via the density method.
