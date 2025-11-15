# SPDX-FileCopyrightText: 2025-present Micah Brown
#
# SPDX-License-Identifier: MIT
from scipy import stats as ss
from typing import Iterable
import numpy as np

class StatisticalAssertion():
    def __init__(self, *samples: Iterable):
        self._samples = samples

    def has_acceptance_rate_less_than(self, target_rate, alpha=0.05):
        self._validate_single_assertion()

        samples = self._get_samples_singleton()
        arr = np.asarray(samples, dtype=bool)

        self._enforce_single_assertion()

        return self._assert_acceptance_rate_check(arr, target_rate, alpha, "less")
    def has_acceptance_rate_greater_than(self, target_rate, alpha=0.05):
        self._validate_single_assertion()

        samples = self._get_samples_singleton()
        arr = np.asarray(samples, dtype=bool)

        self._enforce_single_assertion()

        return self._assert_acceptance_rate_check(arr, target_rate, alpha, "greater")
    
    _acceptance_conditions = ["less", "greater"]
    def _assert_acceptance_rate_check(self, samples: np.ndarray[np.bool], target_rate: float, alpha: float, acceptance_condition='greater'):

        k = np.sum(samples)
        n = samples.size

        result = ss.binomtest(k, n, target_rate, alternative=StatisticalAssertion._acceptance_condition_to_alternative(acceptance_condition))
        if (result.pvalue > alpha):
            raise AssertionError(
                f"Observed rate ({result.statistic:.4f}) is not significantly "
                f"{StatisticalAssertion._acceptance_condition_to_description(acceptance_condition)} than "
                f"target ({target_rate:.4f}) (p={result.pvalue:.4e} >= alpha={alpha})."
            )
    
    def _validate_single_assertion(self):
        if self._samples is None:
            raise RuntimeError("Multiple assertion methods called on the same StatisticalAssertion instance.")
    
    def _enforce_single_assertion(self):
        self._validate_single_assertion()
        self._samples = None
        
    def _validate_alternative(self, alternative: str):
        if alternative not in self._acceptance_conditions:
            raise ValueError(f"alternative ({alternative}) is not a valid argument.")
        
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