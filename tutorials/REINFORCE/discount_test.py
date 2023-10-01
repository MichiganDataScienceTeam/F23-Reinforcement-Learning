import pytest
import torch
import numpy as np
from main import discount_rewards

class TestDiscountRewards():

    def testGammaOne(self):
        """Verify results when gamma = 1."""
        gamma = 1
        arr1 = np.array([-20, -30, -20, 10, 0, 40])
        true_gs = torch.from_numpy(np.array([-20, 0, 30, 50, 40, 40]))
        values = discount_rewards(arr1, gamma)

        assert(true_gs.equal(values))
    
    def testGammaFrac(self):
        """Verify results when 0 < gamma < 1."""
        gamma = 0.97
        arr1 = np.array([-20, -30, -20, 10, 0, 40])
        true_gs = torch.from_numpy(np.array([-20-30*gamma-20*gamma**2+10*gamma**3+40*gamma**5, 
                                             -30-20*gamma+10*gamma**2+40*gamma**4, 
                                             -20+10*gamma+40*gamma**3,
                                             10+40*gamma**2,
                                             40*gamma,
                                             40], dtype=np.float32))
        values = discount_rewards(arr1, gamma)

        assert(true_gs.equal(values))

    def testGammaZero(self):
        """Verify results when gamma = 0."""
        gamma = 0
        arr1 = np.array([-20, -30, -20, 10, 0, 40])
        true_gs = torch.from_numpy(np.array([-20, -30, -20, 10, 0, 40]))
        values = discount_rewards(arr1, gamma)

        assert(true_gs.equal(values))
