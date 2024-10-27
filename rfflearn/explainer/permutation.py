"""
Wrapper functions for the permutation feature importance.
"""

# Declare published functions and variables.
__all__ = ["permutation_feature_importance", "permutation_plot"]

# Import 3rd-party packages.
import numpy as np
import matplotlib.pyplot as plt
import sklearn.inspection


def permutation_feature_importance(model, Xs: np.ndarray, ys: np.ndarray, **kwargs: dict):
    """
    Calculate permutation importance, and set the feature importance (mean of
    permutation importance for each trial) as model.feature_importances_.

    Args:
        model  (object)    : An estimator that has already been fitted.
        Xs     (np.ndarray): Data of shape (n_samples, n_features),
                             on which permutation importance will be computed.
        ys     (np.ndarray): Targets of shape (n_samples,) or (n_samples, n_classes).
                             Specify None for unsupervised model.
        kwargs (dict)      : Additional keyword arguments for
                             sklearn.inspection.permutation_importance.

    Returns:
        (np.ndarray): Raw permutation importance scores of shape (n_features, n_repeats).

    Notes:
        This function add an attribute `feature_importances_` to the `model` instance.
    """
    # Calculate permutation importance.
    # Returnd value of `sklearn.inspection.permutation_importance` is
    # a dictionary-like object with the following keys.
    #   - importances_mean: Mean of feature importance over n_repeats of shape (n_features,).
    #   - importances_std : Standard deviation over n_repeats of shape (n_features,).
    #   - importances     : Raw permutation importance scores of shape (n_features, n_repeats).
    permutation_importance = sklearn.inspection.permutation_importance(model, Xs, ys, **kwargs)

    # Calculate the average of permutation importance for each feature and set the average values
    # as model.feature_importances_ for providing compatible interface with scikit-learn.
    setattr(model, "feature_importances_", permutation_importance.importances_mean)

    return permutation_importance.importances


def permutation_plot(permutation_importances: np.ndarray, feature_names: list, show: bool = True):
    """
    Visualize permutation importance as a box diagram.

    Args:
        permutation_importance (np.ndarray): Raw scores of shape (num_features, num_repeats).
        feature_names          (list)      : List of feature names.
        show                   (bool)      : Shows plot if True.
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
