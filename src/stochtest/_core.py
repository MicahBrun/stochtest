# SPDX-FileCopyrightText: 2025-present Micah Brown
#
# SPDX-License-Identifier: MIT
from scipy import stats as ss
from typing import Iterable
import numpy as np

class _StatisticalAssertion():
    def __init__(self, *samples: Iterable):
        self._samples = samples

    def has_acceptance_rate_less_than(self, target_rate: float, alpha=0.05):
        """
        Asserts that the acceptance rate is significantly less than the target_rate using a binomial test.

        Args:
            target_rate (float): The threshold proportion (0.0 to 1.0).
            alpha (float, optional): Significance level. Defaults to 0.05.
        """
        self._validate_single_assertion()

        samples = self._get_samples_singleton()
        arr = np.asarray(samples, dtype=bool)

        self._enforce_single_assertion()

        self._assert_acceptance_rate_check(arr, target_rate, alpha, "less")
    def has_acceptance_rate_greater_than(self, target_rate: float, alpha=0.05):
        """
        Asserts that the acceptance rate is significantly greater than the target_rate using a binomial test.

        Args:
            target_rate (float): The threshold proportion (0.0 to 1.0).
            alpha (float, optional): Significance level. Defaults to 0.05.
        """
        self._validate_single_assertion()

        samples = self._get_samples_singleton()
        arr = np.asarray(samples, dtype=bool)

        self._enforce_single_assertion()

        self._assert_acceptance_rate_check(arr, target_rate, alpha, "greater")
    def has_acceptance_rate_between(self, minimum: float, maximum: float, alpha=0.05):
        """
        Asserts that the acceptance rate is significantly between minimum and maximum
        using Two One-Sided binomial Tests (TOST).

        Args:
            minimum (float): The lower bound proportion.
            maximum (float): The upper bound proportion.
            alpha (float, optional): Significance level. Defaults to 0.05.
        """
        self._validate_single_assertion()

        samples = self._get_samples_singleton()
        arr = np.asarray(samples, dtype=bool)

        self._enforce_single_assertion()

        k = np.sum(arr)
        n = arr.size

        result_min = ss.binomtest(k, n, minimum, alternative="greater")
        result_max = ss.binomtest(k, n, maximum, alternative="less")

        pvalue = max(result_min.pvalue, result_max.pvalue)

        if (pvalue > alpha):
            raise AssertionError(
                f"Observed rate ({result_min.statistic:.4g}) is not significantly "
                f"between {minimum:.4g} and {maximum:.4g} "
                f"(p={pvalue:.4g} >= alpha={alpha:.4g}). (n={n})"
            )
    def has_acceptance_rate_of(self, target, margin, alpha=0.05):
        """
        Asserts that the acceptance rate is significantly within margin
        using Two One-Sided binomial Tests (TOST).

        Args:
            target (float): The target acceptence rate.
            margin (float): The margin.
            alpha (float, optional): Significance level. Defaults to 0.05.
        """
        self.has_acceptance_rate_between(target-margin, target+margin, alpha)
    
    def has_expected_value_less_than(self, target: float, alpha=0.05):
        """
        Asserts that the population mean is significantly less than the target using a t-test.

        Args:
            target (float): The target mean.
            alpha (float, optional): Significance level. Defaults to 0.05.

        Note:
            For small samples, this test relies on the assumption that the data is 
            normally distributed.
        """
        self._validate_single_assertion()

        samples = self._get_samples_singleton()
        arr = np.asarray(samples, dtype=np.float64)

        self._enforce_single_assertion()

        self._assert_expected_value_check(arr, target, alpha, "less")
    def has_expected_value_greater_than(self, target: float, alpha=0.05):
        """
        Asserts that the population mean is significantly greater than the target using a t-test.

        Args:
            target (float): The target mean.
            alpha (float, optional): Significance level. Defaults to 0.05.

        Note:
            For small samples, this test relies on the assumption that the data is 
            normally distributed.
        """
        self._validate_single_assertion()

        samples = self._get_samples_singleton()
        arr = np.asarray(samples, dtype=np.float64)

        self._enforce_single_assertion()

        self._assert_expected_value_check(arr, target, alpha, "greater")
    def has_expected_value_between(self, minimum: float, maximum: float, alpha=0.05):
        """
        Asserts that the population mean is significantly between minimum and maximum
        using two one-sided t-tests (TOST).

        Args:
            minimum (float): The lower bound value.
            maximum (float): The upper bound value.
            alpha (float, optional): Significance level. Defaults to 0.05.

        Note:
            For small samples, this test relies on the assumption that the data is 
            normally distributed.
        """
        self._validate_single_assertion()

        samples = self._get_samples_singleton()
        arr = np.asarray(samples, dtype=np.float64)

        self._enforce_single_assertion()

        result_min = ss.ttest_1samp(arr, minimum, alternative="greater")
        result_max = ss.ttest_1samp(arr, maximum, alternative="less")
        
        pvalue = max(result_min.pvalue, result_max.pvalue)
        if (pvalue > alpha):
            raise AssertionError(
                f"Sample mean ({samples.mean():.4g}) is not significantly "
                f"between {minimum:.4g} and {maximum:.4g} "
                f"(p={pvalue:.4g} >= alpha={alpha:.4g}). (n={samples.size})"
            )
    def has_expected_value_of(self, target: float, margin: float, alpha=0.05):
        """
        Asserts that the population mean is significantly within a margin of target
        using two one-sided t-tests (TOST).

        Args:
            target (float): The target value.
            margin (float): The margin.
            alpha (float, optional): Significance level. Defaults to 0.05.
        
        Note:
            For small samples, this test relies on the assumption that the data is 
            normally distributed.
        """
        self.has_expected_value_between(target-margin, target+margin, alpha)

    
    _acceptance_conditions = ["less", "greater"]
    def _assert_acceptance_rate_check(self, samples: np.ndarray[bool], target_rate: float, alpha: float, acceptance_condition='greater'):
        k = np.sum(samples)
        n = samples.size

        result = ss.binomtest(k, n, target_rate, alternative=_StatisticalAssertion._acceptance_condition_to_alternative(acceptance_condition))
        if (result.pvalue > alpha):
            raise AssertionError(
                f"Observed rate ({result.statistic:.4g}) is not significantly "
                f"{_StatisticalAssertion._acceptance_condition_to_description(acceptance_condition)} "
                f"target ({target_rate:.4g}) (p={result.pvalue:.4g} >= alpha={alpha:.4g}). (n={n})"
            )
    
    def _assert_expected_value_check(self, samples: np.ndarray[np.float64], target: float, alpha:float, acceptance_condition='greater'):
        result = ss.ttest_1samp(samples, target, alternative=acceptance_condition)

        if (result.pvalue > alpha):
            raise AssertionError(
                f"Sample mean ({samples.mean():.4g}) is not significantly "
                f"{_StatisticalAssertion._acceptance_condition_to_description(acceptance_condition)} "
                f"target ({target:.4g}) (p={result.pvalue:.4g} >= alpha={alpha:.4g}). (n={samples.size})"
            )
    
    def _validate_single_assertion(self):
        if self._samples is None:
            raise RuntimeError(f"Multiple assertion methods called on the same {self.__class__.__name__} instance.")
    
    def _enforce_single_assertion(self):
        self._validate_single_assertion()
        self._samples = None
        
    @staticmethod
    def _acceptance_condition_to_description(acceptance_condition: str) -> str:
        match acceptance_condition:
            case "less":
                return "less than"
            case "greater":
                return "greater than"
            case _:
                return ""
    
    @staticmethod
    def _acceptance_condition_to_alternative(acceptance_condition: str) -> str:
        #acceptance_conditions are a subset of alternatives in scipy
        return acceptance_condition
    
    def _get_samples_singleton(self) -> Iterable:
        if (len(self._samples) != 1):
            raise ValueError(f"{len(self._samples)} sample collections provided, exactly 1 was expected.")
        return self._samples[0]