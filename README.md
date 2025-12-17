# stochtest

[![PyPI - Version](https://img.shields.io/pypi/v/stochtest.svg)](https://pypi.org/project/stochtest)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/stochtest.svg)](https://pypi.org/project/stochtest)

-----

Stochtest is a Python library for performing statistical assertions on stochastic outputs (i.e. outputs described by probability distributions) using statistical methods. It enables automated testing for probabilistic code and Monte Carlo simulations.

## Installation

```console
pip install stochtest
```

## Usage

### 1. Asserting Rates & Means

Use `stochtest.assert_that(samples)` to validate acceptance rates (proportions) or expected values (means).

```python
import stochtest
import numpy as np

# --- Acceptance Rates (Proportions) ---
samples = np.random.normal(1000)

stochtest.assert_that(samples > 0).has_acceptance_rate_greater_than(0.4)
stochtest.assert_that(outcomes).has_acceptance_rate_between(0.45, 0.55)

# --- Expected Values (Means) ---
# Validated using one-sample Student's t-tests
returns = np.random.normal(1, 0.5, 1000)

stochtest.assert_that(returns).has_expected_value_greater_than(0.5)
stochtest.assert_that(returns).has_expected_value_between(0.9, 1.1)

```

### 2. Asserting Distributions

Use `stochtest.distributions.assert_that(samples)` to verify that an entire dataset matches a specific shape or theoretical distribution using a **Bootstrapped Kolmogorov-Smirnov (KS) Test**.

This asserts that the distributions are **sufficiently similar** by ensuring the KS distance is consistently within a specific `margin` with high confidence.

```python
import numpy as np
import scipy.stats
import stochtest.distributions

samples = np.random.normal(0, 1, 1000)

# Assert samples follow a normal distribution (mean=0, std=1)
stochtest.distributions.assert_that(samples).has_normal_distribution(
    loc=0, scale=1, margin=0.05
)

# Assert samples match a custom CDF (e.g., Uniform)
stochtest.distributions.assert_that(samples).has_distribution(
    lambda x: scipy.stats.uniform.cdf(x), 
    margin=0.05
)

```

## License

`stochtest` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
