"""
Wrapper functions for the permutation feature importance.
"""

# Declare published functions and variables.
__all__ = ["permutation_feature_importance", "permutation_plot"]

# Import 3rd-party packages.
import numpy as np
import matplotlib.pyplot as plt
import sklearn.inspection


def permutation_feature_importance(model, Xs, ys, **kwargs):
    """
    Calculate permutation importance, and set the feature importance
    (mean of permutation importance for each trial) as model.feature_importances_.
    """
    # Calculate permutation importance.
    permutation_importance = sklearn.inspection.permutation_importance(model, Xs, ys, **kwargs)

    # Calculate the average of permutation importance for each feature and set the average values
    # as model.feature_importances_ for providing compatible interface with scikit-learn.
    setattr(model, "feature_importances_", permutation_importance.importances_mean)

    return permutation_importance.importances


def permutation_plot(permutation_importances, feature_names, show: bool = True):
    """
    Visualize permutation importance as a box diagram.
    The input arguments are:
      - permutation_importance: np.array with shape (num_features, num_repeats),
      - feature_names: list with length num_features,
      - show: True or False.
    """
    # Sort faetures by the average of permutation order.
    sorted_idx  = np.mean(permutation_importances, axis = 1).argsort()
    importances = permutation_importances[sorted_idx].T
    label_names = np.array(feature_names)[sorted_idx]

    # Plot box diagram.
    plt.boxplot(importances, labels=label_names, vert=False)
    plt.xlabel("Permutation feature importances (impact on model output)")
    plt.grid()

    if show:
        plt.show()


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
