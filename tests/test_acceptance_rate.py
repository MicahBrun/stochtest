# SPDX-FileCopyrightText: 2025-present Micah Brown
#
# SPDX-License-Identifier: MIT
import pytest

import stochtest

def test_has_acceptance_rate_greater_than_happy_path():
    # Arrange
    data = ([True] + [False]) * 200
    
    stochtest.assert_that(data).has_acceptance_rate_greater_than(0.45, alpha=0.05)

def test_has_acceptance_rate_greater_than_fails_when_empirical_rate_less_than():
    # Arrange
    data = [True] * 10 + [False] * 90 

    # Act
    with pytest.raises(AssertionError) as excinfo:
        stochtest.assert_that(data).has_acceptance_rate_greater_than(0.5, alpha=0.05)
    
    # Assert
    assert "is not significantly greater than target" in str(excinfo.value)

def test_has_acceptance_rate_greater_than_fails_when_not_enough_for_significance():
    # Arrange
    data = [True] * 50 + [False] * 50 

    # Act
    with pytest.raises(AssertionError) as excinfo:
        stochtest.assert_that(data).has_acceptance_rate_greater_than(0.45, alpha=0.05)
    
    # Assert
    assert "is not significantly greater than target" in str(excinfo.value)

def test_has_acceptance_rate_less_than_happy_path():
    # Arrange
    data = ([True] + [False]) * 200
    
    stochtest.assert_that(data).has_acceptance_rate_less_than(0.55, alpha=0.05)

def test_has_acceptance_rate_less_than_fails_when_empirical_rate_greater_than():
    # Arrange
    data = [True] * 90 + [False] * 10

    # Act
    with pytest.raises(AssertionError) as excinfo:
        stochtest.assert_that(data).has_acceptance_rate_less_than(0.5, alpha=0.05)
    
    # Assert
    assert "is not significantly less than target" in str(excinfo.value)

def test_has_acceptance_rate_less_than_fails_when_not_enough_for_significance():
    # Arrange
    data = [True] * 50 + [False] * 50 

    # Act
    with pytest.raises(AssertionError) as excinfo:
        stochtest.assert_that(data).has_acceptance_rate_less_than(0.55, alpha=0.05)
    
    # Assert
    assert "is not significantly less than target" in str(excinfo.value)