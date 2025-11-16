# SPDX-FileCopyrightText: 2025-present Micah Brown
#
# SPDX-License-Identifier: MIT
from typing import Iterable

import stochtest._core as _core

def assert_that(*samples: Iterable) -> _core._StatisticalAssertion:
    return _core._StatisticalAssertion(*samples)