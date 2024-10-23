"""
The __init__.py file for the 'rfflearn.explainer' module.
"""

from .permutation import permutation_feature_importance, permutation_plot
from .shapley     import shap_feature_importance, shap_plot

# Declare published functions and variables.
__all__ = ["permutation_feature_importance", "permutation_plot",
           "shap_feature_importance", "shap_plot"]

# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
