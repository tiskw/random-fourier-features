#!/usr/bin/env python3
#
# Wrapper functions for the permutation feature importance.
#
##################################################### SOURCE START #####################################################


import numpy as np
import matplotlib.pyplot as mpl
import sklearn.inspection


### Calculate permutation importance, and set the feature importance
### (mean of permutation importance for each trial) as model.feature_importances_.
def permutation_feature_importance(model, Xs, ys, **kwargs):

    ### Calculate permutation importance.
    permutation_importance = sklearn.inspection.permutation_importance(model, Xs, ys, **kwargs)

    ### Calculate the average of permutation importance for each feature and set the average values
    ### as model.feature_importances_ for providing compatible interface with scikit-learn.
    setattr(model, "feature_importances_", permutation_importance.importances_mean)

    return permutation_importance.importances


### Visualize permutation importance as a box diagram.
### The input arguments are:
###   - permutation_importance: np.array with shape (num_features, num_repeats),
###   - feature_names: list with length num_features,
###   - show: True or False.
def permutation_plot(permutation_importances, feature_names, show = True):

    ### Sort faetures by the average of permutation order.
    sorted_idx  = np.mean(permutation_importances, axis = 1).argsort()
    importances = permutation_importances[sorted_idx].T
    label_names = feature_names[sorted_idx]

    ### Plot box diagram.
    mpl.boxplot(importances, labels = label_names, vert = False)
    mpl.xlabel("Permutation feature importances (impact on model output)")
    mpl.grid()

    if show: mpl.show()


##################################################### SOURCE FINISH ####################################################
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
