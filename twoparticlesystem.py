#------------------------------------------------------------------------------------------
#-----Python module for calculating energies and wavefunctions-----------------------------
#-- for two particles with a contact interaction in a harmonic trap------------------------
#-------author: Jonathan Lindgren----------------------------------------------------------
#------------------------------------------------------------------------------------------

from functools import lru_cache
from math import gamma, factorial,sqrt
import numpy as np
from scipy.optimize import brentq
from scipy.special import psi, hyperu, hermite





#class for handling the energies of the two-particle system
class Energies:

    #computes the ratio Gamma(X+a)/Gamma(X+b)
    @staticmethod
    def _gamma_ratio(X,a,b):
        if (abs((X+b)%1)<1e-20):
            return 0
        res=1.
        while (X+a>20)or(X+b>20):
            res*=(X+a-1)/(X+b-1)
            X-=1
        return gamma(X+a)/gamma(X+b)*res

    #equation in the Busch formula to find the energy
    def _fun(self,E):
        g=self.g
#        if (E>0)and(E%2==0.5):
#            return -g/2/np.sqrt(2)
#        if (E>0)and(E%2==1.5):
#            return NaN
#        try:
        return Energies._gamma_ratio(-E/2,3./4,1./4)+g/np.sqrt(2)/2
#        except:
#            print -E/2,g
#            sys.exit(0)

    def __init__(self,g,xtol=1e-12,rtol=1e-15,maxiter=100):
        self.g=g
        self.idx=0
        self.xtol=xtol
        self.rtol=rtol
        self.maxiter=maxiter

    #approximate value for energy at large interaction
    def _approximate_value_large_g(self,n):
        b=1
        for i in range(1,n//2+1):
            b=b*(i+1./2)/i
        return n+3./2-2*np.sqrt(2)/self.g*b/np.sqrt(np.pi)



    #upper limit to use in the solver for repulsive interactions
    def _safe_upper_limit(self,n):
        return max(n+3./2-1e-3,n+3./2-(n+3./2-self._approximate_value_large_g(n))/5)

    #lower limit to use in the solver for attractive interactions
    def _safe_lower_limit(self,n):
        return min(n-0.5+1e-3,n-0.5-(n-0.5-self._approximate_value_large_g(n-2))/5)    

    #approximate value for weak interactions
    def _approximate_value_small_g(self,n):
        psi0=hermite(n)(0)*1/np.pi**0.25/np.sqrt(2**(n)*factorial(n))
        return n+0.5+self.g*psi0**2/np.sqrt(2)


    #function for obtaining any energy level
    @lru_cache(maxsize=32)
    def _get_energy(self,n):
        if n%2==1:
            return n+0.5
        else:
            if (-1e-10<self.g<1e-10):
                return self._approximate_value_small_g(n)
            else:
                if self.g>0:
                    return brentq(lambda x: self._fun(x),n+0.5,self._safe_upper_limit(n),xtol=self.xtol,rtol=self.rtol,maxiter=self.maxiter)
                else:
                    if n==0:
                        e_min=-1;
                        while self._fun(e_min)<0:
                            e_min*=2
                        return brentq(lambda x: self._fun(x),e_min,n+.5,xtol=self.xtol,rtol=self.rtol,maxiter=self.maxiter)
                    else:
                        print(n+0.5,self._safe_lower_limit(n),self.g)
                        return brentq(lambda x: self._fun(x),self._safe_lower_limit(n),n+.5,xtol=self.xtol,rtol=self.rtol,maxiter=self.maxiter)

    #[] functionality
    def __getitem__(self,key):
        if isinstance(key,slice):
            return [self._get_energy(i) for i in range(key.start,key.stop,key.step)]
            
        if type(key)!=int:
            raise TypeError("indices must be integers or slices, not "+str(type(key)))
        return self._get_energy(key)

    #iterator protocol
    def __iter__(self):
        self.idx=0
        return self

    def __next__(self):
        self.idx+=1
        return self.__getitem__(self.idx-1)



#class for handling a system of two particles in harmonic trap with contact interaction of strength g
class TwoParticleSystem:

    def __init__(self,g,nI=250):

        self.energies=Energies(g)

        #integration tools
        t=np.linspace(-np.pi/2,np.pi/2,nI+2)[1:-1]
        xr=np.tan(t)
        A=np.zeros((nI,nI))
        dx=(t[1]-t[0])*2
        A[1,1]=0.5*(t[1]-t[0])
        A[1,0]=0.5*(t[1]-t[0])
        for i in range(2,nI):
            A[i,:]=A[i-2,:]
            A[i,i-2]+=dx/6
            A[i,i-1]+=2*dx/3
            A[i,i]+=dx/6
        A=A/np.cos(t)**2

        self.Int=A[-1,:]
        self.xs=np.array(xr)

    
        self.g=g


    #the relative wavefunction for an even state with energy e
    def relative_wavefunction_even(self,e,x):

        A=np.sqrt(np.sqrt(2)*2/self.g)/np.sqrt(psi(-e/2+1./4)-psi(-e/2+3./4))
        return -A*self.g/np.sqrt(2*np.pi)/2*gamma(-e/2+1./4)*np.exp(-x**2/2)*hyperu(-e/2+1./4,1./2,x**2)

    #the well known wavefunction for a free harmonic oscillator
    @staticmethod
    def _free_wavefunction(n,x):
        return hermite(n)(x)*1/np.pi**0.25*np.exp(-x**2/2)/np.sqrt(2**n*factorial(n))


    #relative wavefunction for the n:th state
    def relative_wavefunction(self,n,x):
        if type(n)!=int:
            raise TypeError("index must be integer, not "+str(type(n)))
        if n<0:
            raise ValueError("n must not be negative")
        if n%2==0:
            e=self.energies._get_energy(n)
            return self.relative_wavefunction_even(e,x)
        else:
            return TwoParticleSystem._free_wavefunction(n,x)

    #total wavefunction as a function of the position space coordinates of the two particles, for the state with relative excitation n_rel and center of mass excitation n_cm
    def absolute_wavefunction(self,x,y,n_rel,n_cm):
        if type(n_rel)!=int:
            raise TypeError("n_rel must be integer, not "+str(type(n)))
        if n_rel<0:
            raise ValueError("n_cm must not be negative")
        if type(n_cm)!=int:
            raise TypeError("n_cm must be integer, not "+str(type(n)))
        if n_cm<0:
            raise ValueError("n_rel must not be negative")

        return self.relative_wavefunction(self.energies[n_rel],(x-y)/np.sqrt(2))*TwoParticleSystem._free_wavefunction(n_cm,(x+y)/np.sqrt(2))

    #position space density for the state with relative excitation n_rel and center of mass excitation n_cm
    def density(self,x,n_rel,n_cm):
        if type(n_rel)!=int:
            raise TypeError("n_rel must be integer, not "+str(type(n)))
        if n_rel<0:
            raise ValueError("n_cm must not be negative")
        if type(n_cm)!=int:
            raise TypeError("n_cm must be integer, not "+str(type(n)))
        if n_cm<0:
            raise ValueError("n_rel must not be negative")        

        y=np.zeros_like(x)
        
        for i,_x in enumerate(x):    
            y[i]=np.dot(self.Int,self.relative_wavefunction(n_rel,(_x-self.xs)/np.sqrt(2))**2*TwoParticleSystem._free_wavefunction(n_cm,(_x+self.xs)/np.sqrt(2))**2)
        
        return y

    #computes the quantum numbers n_rel and n_cm for the state with absolute quantum number n (the state which has the n:th lowest total energy). Only available for repulsive interactions due to the collapsing bound state on the attractive side. Since the state might be degenerate, it returns a list of tuples (n_rel,n_cm) which might have more than one element
    @staticmethod
    def from_absolute_index_repulsive(n):
        n=int(n)

        #find energy band
        x=np.sqrt(2*n+2+1./4)-1./2
        k=int(np.floor(x))
        if k*(k+1)>=n*2+2:
            k=k-1
        l=n+1-(k*(k+1))//2
        #n is located on the k:th "energy band", starting from k=0
        #there are floor((k+1)/2) non interacting states, and then floor(k/2)+1 interacting states
        if (l<=(k+1)//2):
            #ambiguous which state, return a list
            return [(i*2+1,k-i*2-1) for i in range((k+1)//2)]
        else:
            #determine which interacting state
            return [(2*(k//2)-(l-(k+1)//2-1)*2,(l-(k+1)//2-1)*2+(k%2))]
        

