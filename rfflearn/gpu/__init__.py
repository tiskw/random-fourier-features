#!/usr/bin/env python3
#
# __init__.py file for the module 'rfflearn.gpu'.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : January 29, 2021
##################################################### SOURCE START #####################################################

import functools
import pkgutil
import sys

### Import RFF-related modules.
from .rfflearn_gpu_common import seed
from .rfflearn_gpu_svc    import RFFSVC, ORFSVC, QRFSVC
from .rfflearn_gpu_gp     import RFFGPR, ORFGPR, QRFGPR, RFFGPC, ORFGPC, QRFGPC
from .rfflearn_gpu_pca    import RFFPCA, ORFPCA

### Import optuna-related modules if `optuna` is available.
if pkgutil.get_loader("optuna") is not None:

    from ..tuner import tuner

    RFFSVC_tuner = functools.partial(tuner.RFF_dim_std_tuner,     model_class = RFFSVC)
    ORFSVC_tuner = functools.partial(tuner.RFF_dim_std_tuner,     model_class = ORFSVC)
    QRFSVC_tuner = functools.partial(tuner.RFF_dim_std_tuner,     model_class = QRFSVC)
    RFFGPC_tuner = functools.partial(tuner.RFF_dim_std_err_tuner, model_class = RFFGPC)
    ORFGPC_tuner = functools.partial(tuner.RFF_dim_std_err_tuner, model_class = ORFGPC)
    QRFGPC_tuner = functools.partial(tuner.RFF_dim_std_err_tuner, model_class = QRFGPC)
    RFFGPR_tuner = functools.partial(tuner.RFF_dim_std_err_tuner, model_class = RFFGPR)
    ORFGPR_tuner = functools.partial(tuner.RFF_dim_std_err_tuner, model_class = ORFGPR)
    QRFGPR_tuner = functools.partial(tuner.RFF_dim_std_err_tuner, model_class = QRFGPR)

else: print("rfflearn.cpu: package 'optuna' not found. SKip loading optuna-related functions.", file = sys.stderr)

### Import shap-related modules if `shap` is available.
if pkgutil.get_loader("shap") is not None:

    from ..explainer.shap import shap_feature_importance
    from ..explainer.shap import shap_plot

else: print("rfflearn.cpu: package 'shap' not found. SKip loading shap-related functions.", file = sys.stderr)

##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
