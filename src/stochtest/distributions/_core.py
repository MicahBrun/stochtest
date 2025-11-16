# SPDX-FileCopyrightText: 2025-present Micah Brown
#
# SPDX-License-Identifier: MIT
from typing import Iterable
import scipy.stats as ss
import numpy as np

class _DistributionsStatisticalAssertion:
    def __init__(self, samples: Iterable):
        self._samples = samples

    def has_distribution(self, reference_samples: Iterable, margin = 0.01, confidence = 0.95, n_bootstraps = 1000, random_state=None):
        self._validate_single_assertion()
        
        n=len(self._samples)

        generator = np.random.default_rng(random_state)
        sample_bootstraps = generator.choice(self._samples, size=(n_bootstraps, n), replace=True)

        ks_output = ss.ks_2samp(sample_bootstraps, reference_samples, axis=1)
        ks_distances = ks_output.statistic

        successes = ks_distances <= margin
        success_rate = sum(successes)/n_bootstraps

        base_ks_distance = ss.ks_2samp(self._samples, reference_samples).statistic

        self._enforce_single_assertion()

        if (success_rate < confidence):
            raise AssertionError(f"Confidence ({success_rate}) less than ({confidence}). Distance={base_ks_distance}")


    def _validate_single_assertion(self):
        if self._samples is None:
            raise RuntimeError(f"Multiple assertion methods called on the same {self.__class__.__name__} instance.")
    
    def _enforce_single_assertion(self):
        self._validate_single_assertion()
        self._samples = None