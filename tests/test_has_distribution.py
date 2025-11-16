import scipy.stats as ss
import numpy as np
import pytest

import stochtest.distributions

def test_has_distribution_passes_for_normal():
    random_state = np.random.RandomState(42)
    reference_samples = ss.norm.rvs(size=10000, random_state=random_state)
    test_samples = ss.norm.rvs(size=10000, random_state=random_state)

    stochtest.distributions.assert_that(test_samples).has_distribution(reference_samples, random_state=random_state, margin=0.02)

def test_has_distribution_fails_with_normal_with_different_loc():
    # Arrange
    random_state = np.random.RandomState(42)
    reference_samples = ss.norm.rvs(size=10000, random_state=random_state)
    test_samples = ss.norm.rvs(loc=0.2, size=10000, random_state=random_state)


    # Act
    with pytest.raises(AssertionError) as excinfo:
        stochtest.distributions.assert_that(test_samples).has_distribution(reference_samples, random_state=random_state, margin=0.02)
    
    # Assert
    assert " less than" in str(excinfo.value)