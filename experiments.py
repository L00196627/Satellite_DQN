"""
Statistical Distributions Module
================================

This module provides wrapper functions for generating random numbers based on
statistical distributions, primarily Exponential.

It is designed to ensure **simulation reproducibility** by utilizing the centralized
NumPy random number generator defined in `globals.rnd_np`. It also handles compatibility
between different NumPy generator interfaces (Legacy `RandomState` vs New `Generator`).

Key Functions:
--------------
* `exponential`: Standard exponential distribution (e.g., for inter-arrival times).
* `truncated_exponential_trunc`: Rejection sampling for bounded values.
* `truncated_exponential_unbounded`: Wrapper for unbounded generation with a compatible signature.
"""
import globals

# Load the config only if you need it locally (you can still read from globals.config)
config = globals.config

def exponential(_=None):
    """
    Exponential distribution: The call may accept an argument (e.g., ‘Task’),
    which is ignored for compatibility.
    Use globals.rnd_np for reproducibility.
    """
    mean = config.get("arrival_time_exponential", 0.5)
    # globals.rnd_np can be a Generator (default_rng) or a fallback RandomState
    try:
        return float(globals.rnd_np.exponential(scale=mean))
    except AttributeError:
        # fallback RandomState
        return float(globals.rnd_np.exponential(mean))


# --- truncated_exponential: two versions available ---
def truncated_exponential_trunc(mean, lower, upper):
    """
    ABRIDGED version (like the one you had before), but it uses globals.rnd_np.
    """
    # If `globals.rnd_np` is a Generator:
    try:
        while True:
            value = float(globals.rnd_np.exponential(scale=mean))
            if lower <= value <= upper:
                return value
    except AttributeError:
        # fallback RandomState
        while True:
            value = float(globals.rnd_np.exponential(mean))
            if lower <= value <= upper:
                return value


def truncated_exponential_unbounded(mean, lower=None, upper=None):
    """
    UNCLIPPED version for CPU times (returns an exponential value with no limits).
    Use ‘mean’ as the scaling parameter.
    If you want different behavior, change it here.
    """
    try:
        return float(globals.rnd_np.exponential(scale=mean))
    except AttributeError:
        return float(globals.rnd_np.exponential(mean))
