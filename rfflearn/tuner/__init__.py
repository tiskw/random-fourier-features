"""
The __init__.py file for the 'rfflearn.explainer' module.
"""

from .hptuner import RFF_dim_std_tuner, RFF_dim_std_err_tuner

# Declare published functions and variables.
__all__ = ["RFF_dim_std_tuner", "RFF_dim_std_err_tuner"]

# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
