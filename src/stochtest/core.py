# SPDX-FileCopyrightText: 2025-present Micah Brown
#
# SPDX-License-Identifier: MIT
from scipy import stats as ss
from typing import TypeVar, Generic, Iterable

T = TypeVar('T')
class StatisticalAssertion(Generic[T]):
    def __init__(self, samples: Iterable[T]):
        self._samples = samples
    
    def has_acceptance_rate(self, target_rate, alpha=0.05):
        k = sum(self._samples)
        n = len(self._samples)

        p_value = ss.binomtest(k, n, target_rate, alternative='less')
        if (p_value < alpha):
            observed_rate = k / n
            raise AssertionError(
                f"Observed rate ({observed_rate:.4f}) is significantly less than"
                f"target ({target_rate:.4f}) (p={p_value:.4e} < alpha={alpha})."
            )