# SPDX-FileCopyrightText: 2025-present Micah Brown
#
# SPDX-License-Identifier: MIT
from typing import Iterable
import scipy.stats as ss
import numpy as np

class _DistributionsStatisticalAssertion:
    def __init__(self, samples: Iterable):
        self._samples = samples

    def has_distribution(self, reference_samples: Iterable, margin = 0.02, confidence = 0.95, n_bootstraps = 500, random_state=None):
        self._validate_single_assertion()
        MAX_INDIVIDUAL_SAMPLES_PER_BATCH = 1_000_000

        generator = np.random.default_rng(random_state)

        n=len(self._samples)
        max_batch_size = max(1, MAX_INDIVIDUAL_SAMPLES_PER_BATCH//n)

        ks_distances = np.array(
            [d
                for batch_size in _DistributionsStatisticalAssertion.batch_sizes(
                    n_bootstraps, 
                    max_batch_size)
                for d in _DistributionsStatisticalAssertion._apply_ks_2samp_to_batch(
                    self._samples, 
                    reference_samples, 
                    batch_size, 
                    generator)])

        successes = ks_distances <= margin
        success_rate = sum(successes)/n_bootstraps

        base_ks_distance = ss.ks_2samp(self._samples, reference_samples).statistic

        self._enforce_single_assertion()

        if (success_rate < confidence):
            raise AssertionError(f"Confidence ({success_rate}) less than ({confidence}). Distance={base_ks_distance}")
    
    @staticmethod
    def _apply_ks_2samp_to_batch(samples, reference_samples, n_bootstraps_in_batch, generator):
        n=len(samples)
        sample_bootstraps = generator.choice(samples, size=(n_bootstraps_in_batch, n), replace=True)
        ks_output = ss.ks_2samp(sample_bootstraps, reference_samples, axis=1)
        return ks_output.statistic


    def _validate_single_assertion(self):
        if self._samples is None:
            raise RuntimeError(f"Multiple assertion methods called on the same {self.__class__.__name__} instance.")
    
    def _enforce_single_assertion(self):
        self._validate_single_assertion()
        self._samples = None
    
    @staticmethod
    def batch_sizes(n, d):
        full_batches = n // d
        remainder = n % d
        for _ in range(full_batches):
            yield d
        if remainder:
            yield remainder