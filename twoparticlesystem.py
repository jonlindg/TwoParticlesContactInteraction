#------------------------------------------------------------------------------------------
#--Python module for calculating energies and wavefunctions--------------------------------
#-- for two particles with a contact interaction in a harmonic trap------------------------
#-------author: Jonathan Lindgren----------------------------------------------------------
#------------------------------------------------------------------------------------------



from scipy.optimize import brentq
import numpy as np
from math import gamma, factorial,sqrt
from scipy.special import psi, hyperu, hermite
import sys
import pylab as pl

from functools import lru_cache
import unittest


class Energies:

    @staticmethod
    def _gamma_ratio(X,a,b):
        if ((X+b)%1==0.0):
            return 0
        res=1.
        while (X+a>20)or(X+b>20):
            res*=(X+a-1)/(X+b-1)
            X-=1
        return gamma(X+a)/gamma(X+b)*res

    def _fun(self,E):##equation in the Busch formula
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

    def _approximate_value_large_g(self,n):
        b=1
        for i in range(1,n//2+1):
            b=b*(i+1./2)/i
        return n+3./2-2*np.sqrt(2)/self.g*b/np.sqrt(np.pi)


    def _safe_upper_limit(self,n):
        return max(n+3./2-1e-3,n+3./2-(n+3./2-self._approximate_value_large_g(n))/5)
    

    def _approximate_value_small_g(self,n):
        psi0=hermite(n)(0)*1/np.pi**0.25/np.sqrt(2**(n)*factorial(n))
        return n+0.5+self.g*psi0**2/np.sqrt(2)


    @lru_cache(maxsize=32)
    def _get_energy(self,n):
        if n%2==1:
            return n+0.5
        else:
            if (self.g==0):
                return n+0.5
            else:
                return brentq(lambda x: self._fun(x),n+0.5,n+1.5-1e-7,xtol=self.xtol,rtol=self.rtol,maxiter=self.maxiter)

    def __getitem__(self,key):
        if isinstance(key,slice):
            return [self._get_energy(i) for i in range(key.start,key.stop,key.step)]
            
        if type(key)!=int:
            raise TypeError("indices must be integers or slices, not "+str(type(key)))
        return self._get_energy(key)

    def __iter__(self):
        self.idx=0
        return self

    def __next__(self):
        self.idx+=1
        return self.__getitem__(self.idx-1)


class TwoParticleSystem:

    def __init__(self,g):#input is -1/g
        ##initialize all energies and eigenvectors. Nmax is the number of even parity states computed. g is the interaction.

        self.energies=Energies(-1./g)

        #integration tools
        nI=250
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
        INT=A

        self.Int=A[-1,:]
        self.xs=np.array(xr)

    
        self.g=g



    def relative_wavefunction_even(self,e,x):
        A=np.sqrt(np.sqrt(2)*2/self.g)/np.sqrt(psi(-e/2+1./4)-psi(-e/2+3./4))
        return -A*self.g/np.sqrt(2*np.pi)/2*gamma(-e/2+1./4)*np.exp(-x**2/2)*hyperu(-e/2+1./4,1./2,x**2)

    def free_wavefunction(self,n,x):
        return hermite(n)(x)*1/np.pi**0.25*np.exp(-x**2/2)/np.sqrt(2**n*factorial(n))


    def relative_wavefunction(self,n,x):
        if type(n)!=int:
            raise TypeError("index must be integer, not "+str(type(n)))
        if n<0:
            raise ValueError("n must not be negative")
        if n%2==0:
            e=self.energies._get_energy(n)
            return self.relative_wavefunction_even(e,x)
        else:
            return self.free_wavefunction(n,x)

    
    def absolute_wavefunction(self,x,y,n_rel,n_cm):
        if type(n_rel)!=int:
            raise TypeError("n_rel must be integer, not "+str(type(n)))
        if n_rel<0:
            raise ValueError("n_cm must not be negative")
        if type(n_cm)!=int:
            raise TypeError("n_cm must be integer, not "+str(type(n)))
        if n_cm<0:
            raise ValueError("n_rel must not be negative")

        return self.relative_wavefunction(self.energies[n_rel],(x-y)/np.sqrt(2))*self.free_wavefunction(n_cm,(x+y)/np.sqrt(2))

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
            y[i]=np.dot(self.Int,self.relative_wavefunction(n_rel,(_x-self.xs)/np.sqrt(2))**2*self.free_wavefunction(n_cm,(_x+self.xs)/np.sqrt(2))**2)
        
        return y



class TwoParticleTest(unittest.TestCase):
    def setUp(self):
        self.T1=TwoParticleSystem(-1000)
        self.T2=TwoParticleSystem(-100)
        self.T3=TwoParticleSystem(-10)
        self.T4=TwoParticleSystem(-1)
        self.T5=TwoParticleSystem(-0.1)
        self.T6=TwoParticleSystem(-0.01)
        self.T7=TwoParticleSystem(-0.001)
        self.T8=TwoParticleSystem(-0.0001)


    def test_approximate_values_small_g(self):
        self.assertAlmostEqual((0.5-self.T1.energies[0])/(0.5-self.T1.energies._approximate_value_small_g(0)),1.,3)
        self.assertAlmostEqual((2.5-self.T1.energies[2])/(2.5-self.T1.energies._approximate_value_small_g(2)),1.,3)
        self.assertAlmostEqual((4.5-self.T1.energies[4])/(4.5-self.T1.energies._approximate_value_small_g(4)),1.,3)
        self.assertAlmostEqual((6.5-self.T1.energies[6])/(6.5-self.T1.energies._approximate_value_small_g(6)),1.,3)
        self.assertAlmostEqual((0.5-self.T2.energies[0])/(0.5-self.T2.energies._approximate_value_small_g(0)),1.,2)
        self.assertAlmostEqual((2.5-self.T2.energies[2])/(2.5-self.T2.energies._approximate_value_small_g(2)),1.,2)
        self.assertAlmostEqual((4.5-self.T2.energies[4])/(4.5-self.T2.energies._approximate_value_small_g(4)),1.,2)
        self.assertAlmostEqual((6.5-self.T2.energies[6])/(6.5-self.T2.energies._approximate_value_small_g(6)),1.,2)

    def test_approximate_values_large_g(self):
        self.assertAlmostEqual((1.5-self.T7.energies[0])/(1.5-self.T7.energies._approximate_value_large_g(0)),1.,3)
        self.assertAlmostEqual((3.5-self.T7.energies[2])/(3.5-self.T7.energies._approximate_value_large_g(2)),1.,3)
        self.assertAlmostEqual((5.5-self.T7.energies[4])/(5.5-self.T7.energies._approximate_value_large_g(4)),1.,3)
        self.assertAlmostEqual((7.5-self.T7.energies[6])/(7.5-self.T7.energies._approximate_value_large_g(6)),1.,3)
        self.assertAlmostEqual((1.5-self.T8.energies[0])/(1.5-self.T8.energies._approximate_value_large_g(0)),1.,4)
        self.assertAlmostEqual((3.5-self.T8.energies[2])/(3.5-self.T8.energies._approximate_value_large_g(2)),1.,4)
        self.assertAlmostEqual((5.5-self.T8.energies[4])/(5.5-self.T8.energies._approximate_value_large_g(4)),1.,4)
        self.assertAlmostEqual((7.5-self.T8.energies[6])/(7.5-self.T8.energies._approximate_value_large_g(6)),1.,4)

    def test_upper_limit(self):
        for T in [self.T1,self.T2,self.T3,self.T4,self.T5,self.T6,self.T7,self.T8]:
            self.assertTrue(T.energies._approximate_value_large_g(0)<T.energies._safe_upper_limit(0))
            self.assertTrue(T.energies._approximate_value_large_g(2)<T.energies._safe_upper_limit(2))
            self.assertTrue(T.energies._approximate_value_large_g(8)<T.energies._safe_upper_limit(8))
            self.assertTrue(T.energies[0]<T.energies._safe_upper_limit(0))
            self.assertTrue(T.energies[2]<T.energies._safe_upper_limit(2))
            self.assertTrue(T.energies[8]<T.energies._safe_upper_limit(8))

    def test_normalization(self):
        x=np.linspace(-10,10,1000)
        norm = sum(self.T2.relative_wavefunction_even(self.T2.energies[0],x)**2*(x[1]-x[0]))
        self.assertAlmostEqual(norm,1,3)
        norm = sum(self.T2.relative_wavefunction_even(self.T2.energies[2],x)**2*(x[1]-x[0]))
        self.assertAlmostEqual(norm,1,3)
        norm = sum(self.T2.relative_wavefunction_even(self.T2.energies[4],x)**2*(x[1]-x[0]))
        self.assertAlmostEqual(norm,1,3)

        norm = sum(self.T4.relative_wavefunction_even(self.T4.energies[0],x)**2*(x[1]-x[0]))
        self.assertAlmostEqual(norm,1,3)
        norm = sum(self.T4.relative_wavefunction_even(self.T4.energies[2],x)**2*(x[1]-x[0]))
        self.assertAlmostEqual(norm,1,3)
        norm = sum(self.T4.relative_wavefunction_even(self.T4.energies[4],x)**2*(x[1]-x[0]))
        self.assertAlmostEqual(norm,1,3)

        norm = sum(self.T7.relative_wavefunction_even(self.T7.energies[0],x)**2*(x[1]-x[0]))
        self.assertAlmostEqual(norm,1,3)
        norm = sum(self.T7.relative_wavefunction_even(self.T7.energies[2],x)**2*(x[1]-x[0]))
        self.assertAlmostEqual(norm,1,3)
        norm = sum(self.T7.relative_wavefunction_even(self.T7.energies[4],x)**2*(x[1]-x[0]))
        self.assertAlmostEqual(norm,1,3)

        norm = sum(self.T4.density(self.T7.energies[0],x)*(x[1]-x[0]))
        self.assertAlmostEqual(norm,1,3)
        norm = sum(self.T4.density(self.T7.energies[2],x)*(x[1]-x[0]))
        self.assertAlmostEqual(norm,1,3)
        norm = sum(self.T4.density(self.T7.energies[4],x)*(x[1]-x[0]))
        self.assertAlmostEqual(norm,1,3)

    

if __name__=="__main__":

#    unittest.main()

    x=np.linspace(-5,5,1000)
#    nu=0.123
#    y=hyperu(nu,.5,x**2)*np.exp(-x**2/2)
#    pl.plot(x,y)
#    pl.show()




#    e=np.linspace(-1,50,1000)
#    v=np.zeros(1000)
#    for i in range(1000):
#        v[i]=fun(e[i],-1/(-0.1*10))
#    pl.plot(e,v)
#    for i in range(50):
#        pl.plot([i*2+1,i*2+1],[-10,10],'--')
#    pl.plot([-1,50],[0,0],'--')
#    pl.ylim([-10,10])
#    pl.xlim([-1,15])

    T=TwoParticleSystem(-2./5)

    e=np.linspace(-1,50,1000)
    v=np.zeros(1000)
    for i in range(1000):
        v[i]=T.energies._fun(e[i])
    pl.plot(e,v)
    for i in range(50):
        pl.plot([i*2+1.5,i*2+1.5],[-10,10],'--')
    pl.plot([-1,50],[0,0],'--')
    pl.ylim([-10,10])
    pl.xlim([-1,15])
    

    x=np.linspace(-10,10,1000)
    y=T.density(x,0,0)
    pl.figure()
    pl.plot(x,y)
    pl.show()



    print(T.energies[0])
    print(T.energies[1])
    print(T.energies[2])
    print(T.energies[3])
    print(T.energies[4])
    print(T.energies[5])

    


    print((1.5-T.energies._approximate_value_large_g(0))/(1.5-T.energies[0]),(1.5-T.energies[0]))
    print((3.5-T.energies._approximate_value_large_g(2))/(3.5-T.energies[2]),(3.5-T.energies[2]))
    print((5.5-T.energies._approximate_value_large_g(4))/(5.5-T.energies[4]),(5.5-T.energies[4]))
    print((7.5-T.energies._approximate_value_large_g(6))/(7.5-T.energies[6]),(7.5-T.energies[6]))
    print((9.5-T.energies._approximate_value_large_g(8))/(9.5-T.energies[8]),(9.5-T.energies[8]))
    print((11.5-T.energies._approximate_value_large_g(10))/(11.5-T.energies[10]),(11.5-T.energies[10]))


    T=TwoParticleSystem(-10000)
    print((0.5-T.energies[0])/(0.5-T.energies._approximate_value_small_g(0)),T.energies._approximate_value_small_g(0))
    print((2.5-T.energies[2])/(2.5-T.energies._approximate_value_small_g(2)),T.energies._approximate_value_small_g(2))
    print((4.5-T.energies[4])/(4.5-T.energies._approximate_value_small_g(4)),T.energies._approximate_value_small_g(4))


    hggh
#    for e in T.energies:
#        print(e,0)
#        if (e>5):
#            break
#    print(T.energies[0:6:2])

    asd
    x=np.linspace(-10,10,100000)
    y=T.relative_wavefunction_even(T.energies[0],x)
    print(sum(y**2)*(x[1]-x[0]))
    pl.figure()
    pl.plot(x,y)
    pl.show()


    T=TwoParticleSystem(-100*np.sqrt(2))
    print(T.energies+0.5)
    asds
    T=TwoPar
