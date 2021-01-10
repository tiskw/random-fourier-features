#!/usr/bin/env python3
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : January 09, 2021
##################################################### SOURCE START #####################################################


import numpy as np
import shap


### Calculate SHAP values using shap library, and set the feature importance
### (absolute of SHAP values) as model.feature_importances_.
def shap_feature_importance(model, Xs):

    ### Get shap values using shap library.
    reference   = np.zeros((1, Xs.shape[1]))
    explainer   = shap.KernelExplainer(model.predict, reference)
    shap_values = explainer.shap_values(Xs)

    ### Calculate the average of the absolute of shap values for each feature and set the average values
    ### as model.feature_importances_ for providing compatible interface with scikit-learn.
    avg_shap_values = np.mean(np.abs(shap_values), axis = 0)
    setattr(model, "feature_importances_", avg_shap_values)

    return shap_values


### This function is just an alias of the shap.summary_plot function
### because the shap.summary_plot is implemented very well!!
def shap_plot(*pargs, **kwargs):

    return shap.summary_plot(*pargs, **kwargs, plot_type = "violin", color = "coolwarm")


##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
