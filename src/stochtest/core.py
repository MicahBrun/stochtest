# SPDX-FileCopyrightText: 2025-present Micah Brown
#
# SPDX-License-Identifier: MIT
from scipy import stats as ss
from typing import TypeVar, Generic, Iterable
import numpy as np

T = TypeVar('T')
class StatisticalAssertion(Generic[T]):
    _accepted_alternatives = ["less", "greater", "two-sided"]
    def __init__(self, samples: Iterable[T]):
        self._samples = samples
    
    def has_acceptance_rate(self, target_rate, alpha=0.05, alternative='less'):
        self._validate_single_assertion()
        self._validate_alternative(alternative)

        arr = np.asarray(self._samples, dtype=bool)
        self._samples = None

        k = sum(arr)
        n = len(arr)

        result = ss.binomtest(k, n, target_rate, alternative=alternative)
        if (result.pvalue < alpha):
            raise AssertionError(
                f"Observed rate ({result.statistic:.4f}) is significantly {StatisticalAssertion._alternative_to_description(alternative)} than"
                f"target ({target_rate:.4f}) (p={result.pvalue:.4e} < alpha={alpha})."
            )
    
    def _validate_single_assertion(self):
        if self._samples == None:
            raise RuntimeError("Multiple assertion methods called on the same StatisticalAssertion instance.")
        
    def _validate_alternative(self, alternative: str):
        if alternative not in self._accepted_alternatives:
            raise ValueError(f"alternative ({alternative}) is not a valid argument.")
        
    @staticmethod
    def _alternative_to_description(alternative):
        match alternative:
            case "less":
                return "less than"
            case "greater":
                return "greater than"
            case "two-sided":
                return "different from"
            case _:
                return ""