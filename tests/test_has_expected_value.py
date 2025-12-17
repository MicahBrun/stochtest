# SPDX-FileCopyrightText: 2025-present Micah Brown
#
# SPDX-License-Identifier: MIT
import pytest
import scipy.stats as ss

import stochtest

def test_has_expected_value_less_than_happy_path():
    rvs = ss.norm.rvs(size=10000, random_state=42)
    
    stochtest.assert_that(rvs).has_expected_value_less_than(0.05, alpha=0.05)

def test_has_expected_value_less_than_happy_path():
    rvs = ss.norm.rvs(size=10000, random_state=42)
    
    stochtest.assert_that(rvs).has_expected_value_greater_than(-0.05, alpha=0.05)

def test_has_expected_value_between_happy_path():
    rvs = ss.norm.rvs(size=10000, random_state=42)
    
    stochtest.assert_that(rvs).has_expected_value_between(-0.05, 0.05, alpha=0.05)

def test_has_expected_value_between_happy_path():
    rvs = ss.norm.rvs(size=10000, random_state=42)
    
    stochtest.assert_that(rvs).has_expected_value_of(0, 0.05, alpha=0.05)