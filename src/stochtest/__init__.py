# SPDX-FileCopyrightText: 2025-present Micah Brown
#
# SPDX-License-Identifier: MIT
from typing import TypeVar, Iterable

import core

T=core.T
def assert_that(samples: Iterable[T]) -> core.StatisticalAssertion[T]:
    return core.StatisticalAssertion(samples)