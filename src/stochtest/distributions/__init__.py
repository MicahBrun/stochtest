# SPDX-FileCopyrightText: 2025-present Micah Brown
#
# SPDX-License-Identifier: MIT
from typing import Iterable

from ._core import _DistributionsStatisticalAssertion

def assert_that(samples: Iterable) -> _DistributionsStatisticalAssertion:
    return _DistributionsStatisticalAssertion(samples)
