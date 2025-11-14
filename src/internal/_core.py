# SPDX-FileCopyrightText: 2025-present Micah Brown
#
# SPDX-License-Identifier: MIT
from scipy import stats as ss
from typing import TypeVar, Generic, Iterable
import numpy as np

class StatisticalAssertion():
    _acceptance_conditions = ["less", "greater"]
    def __init__(self, samples: Iterable):
        self._samples = samples
    
    def has_acceptance_rate(self, target_rate, alpha=0.05, acceptance_condition='less'):
        self._validate_single_assertion()
        self._validate_alternative(acceptance_condition)

        arr = np.asarray(self._samples, dtype=bool)
        self._samples = None

        k = np.sum(arr)
        n = arr.size

        result = ss.binomtest(k, n, target_rate, alternative=StatisticalAssertion._acceptance_condition_to_alternative(acceptance_condition))
        if (result.pvalue >= alpha):
            raise AssertionError(
                f"Observed rate ({result.statistic:.4f}) is not significantly {StatisticalAssertion._acceptance_condition_to_description(acceptance_condition)} than"
                f"target ({target_rate:.4f}) (p={result.pvalue:.4e} >= alpha={alpha})."
            )
    
    def _validate_single_assertion(self):
        if self._samples is None:
            raise RuntimeError("Multiple assertion methods called on the same StatisticalAssertion instance.")
        
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
        #conditions are identical to alternative
        return acceptance_condition