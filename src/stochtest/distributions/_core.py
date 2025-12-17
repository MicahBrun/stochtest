# SPDX-FileCopyrightText: 2025-present Micah Brown
#
# SPDX-License-Identifier: MIT
from typing import Iterable, Union, Callable
import scipy.stats as ss
import numpy as np

class _DistributionsStatisticalAssertion:
    def __init__(self, samples: Iterable):
        self._samples = samples
    
    def has_normal_distribution(self, loc, scale, margin = 0.02, confidence = 0.95, n_bootstraps = 500, random_state:np.random.RandomState=None):
        """
        Asserts that the samples follow a Normal distribution N(loc, scale).

        This checks if the KS distance between the empirical CDF of the samples 
        and the theoretical CDF of N(loc, scale) is consistently small.

        Args:
            loc (float): The mean of the target normal distribution.
            scale (float): The standard deviation of the target normal distribution.
            margin (float, optional): The maximum acceptable KS distance. 
                                     Defaults to 0.02.
            confidence (float, optional): The required proportion of bootstraps 
                                          that must fall within the margin. 
                                          Defaults to 0.95 (95%).
            n_bootstraps (int, optional): Number of bootstrap resamples to generate. 
                                          Defaults to 500.
            random_state (optional): Seed or generator for reproducibility.
        
        Raises:
            AssertionError: If the confidence threshold is not met for the given margin.
        """
        cdf = lambda x: ss.norm.cdf(x, loc=loc, scale=scale)
        self._has_distribution_from_cdf(cdf, margin, confidence, n_bootstraps, random_state)

    def has_distribution(self, reference: Union[Iterable[float], Callable[[float], float]], margin = 0.02, confidence = 0.95, n_bootstraps = 500, random_state=None):
        """
        Asserts that the samples follow a target distribution provided by a
        reference dataset or a cumulative distribution function (CDF).

        Args:
            reference (Union[Iterable, Callable]): 
                - If Iterable: treated as a reference sample (Two-sample KS test).
                - If Callable: treated as a theoretical CDF (One-sample KS test).
            margin (float, optional): The maximum acceptable KS distance. 
                                     Defaults to 0.02.
            confidence (float, optional): The required proportion of bootstraps 
                                          that must fall within the margin. 
                                          Defaults to 0.95.
            n_bootstraps (int, optional): Number of bootstrap resamples. Defaults to 500.
            random_state (optional): Seed or generator for reproducibility.

        Raises:
            TypeError: If reference is neither an iterable nor a callable.
            AssertionError: If the distributions are not sufficiently similar.
        """
        if callable(reference):
            self._has_distribution_from_cdf(reference, margin, confidence, n_bootstraps, random_state)
        elif isinstance(reference, Iterable):
            self._has_distribution_from_reference_samples(reference, margin, confidence, n_bootstraps, random_state)
        else:
            raise TypeError("reference must be iterable or a callable CDF")

    def _has_distribution_from_cdf(self, reference_cdf: Callable[[float], float], margin: float, confidence: float, n_bootstraps: int, random_state=None):
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
                for d in _DistributionsStatisticalAssertion._apply_ks_1samp_to_batch(
                    self._samples, 
                    reference_cdf, 
                    batch_size, 
                    generator)])

        successes = ks_distances <= margin
        success_rate = sum(successes)/n_bootstraps

        base_ks_distance = ss.ks_1samp(self._samples, reference_cdf).statistic

        self._enforce_single_assertion()

        if (success_rate < confidence):
            raise AssertionError(f"Confidence ({success_rate}) less than ({confidence}). Distance={base_ks_distance}")

    def _has_distribution_from_reference_samples(self, reference_samples: Iterable, margin: float, confidence: float, n_bootstraps: int, random_state=None):
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
    def _apply_ks_1samp_to_batch(samples, reference_cdf, n_bootstraps_in_batch, generator):
        n=len(samples)
        sample_bootstraps = generator.choice(samples, size=(n_bootstraps_in_batch, n), replace=True)
        ks_output = ss.ks_1samp(sample_bootstraps, reference_cdf, axis=1)
        return ks_output.statistic
    
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