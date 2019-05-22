import unittest
import numpy as np
from twoparticlesystem import TwoParticleSystem


class TwoParticleTest(unittest.TestCase):
    def setUp(self):
        self.T1=TwoParticleSystem(-1./(-1000))
        self.T2=TwoParticleSystem(-1./(-100))
        self.T3=TwoParticleSystem(-1./(-10))
        self.T4=TwoParticleSystem(-1./(-1))
        self.T5=TwoParticleSystem(-1./(-0.1))
        self.T6=TwoParticleSystem(-1./(-0.01))
        self.T7=TwoParticleSystem(-1./(-0.001))
        self.T8=TwoParticleSystem(-1./(-0.0001))


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

        norm = sum(self.T4.density(x,0,0)*(x[1]-x[0]))
        self.assertAlmostEqual(norm,1,3)
        norm = sum(self.T4.density(x,2,0)*(x[1]-x[0]))
        self.assertAlmostEqual(norm,1,3)
        norm = sum(self.T4.density(x,0,2)*(x[1]-x[0]))
        self.assertAlmostEqual(norm,1,3)

if __name__=="__main__":

    unittest.main()
