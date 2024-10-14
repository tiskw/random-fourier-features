"""
Wrapper functions for the SHAP method.
"""

# Declare published functions and variables.
__all__ = ["shap_feature_importance", "shap_plot"]

# Import 3rd-party packages.
import numpy as np
import shap


def shap_feature_importance(model, Xs):
    """
    Calculate SHAP values using shap library, and set the feature importance
    (absolute of SHAP values) as model.feature_importances_.
    """
    # Get shap values using shap library.
    reference   = np.zeros((1, Xs.shape[1]))
    explainer   = shap.KernelExplainer(model.predict, reference)
    shap_values = explainer.shap_values(Xs)

    # Calculate the average of the absolute of shap values for each feature
    # and set the average values as model.feature_importances_ for providing
    # compatible interface with scikit-learn.
    avg_shap_values = np.mean(np.abs(shap_values), axis=0)
    setattr(model, "feature_importances_", avg_shap_values)

    return shap_values


def shap_plot(*pargs, **kwargs):
    """
    This function is just an alias of the shap.summary_plot function
    because the shap.summary_plot is implemented very well!!
    """
    return shap.summary_plot(*pargs, **kwargs, plot_type="violin", color="coolwarm")


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
