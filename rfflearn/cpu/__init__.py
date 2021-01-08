#!/usr/bin/env python3
#
# __init__.py file for the module 'rfflearn.cpu'.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : January 08, 2021
##################################################### SOURCE START #####################################################

import functools
import pkgutil

# Import RFF-related modules
from .rfflearn_cpu_common     import seed
from .rfflearn_cpu_regression import RFFRegression, ORFRegression
from .rfflearn_cpu_svc        import RFFSVC, ORFSVC, RFFBatchSVC, ORFBatchSVC
from .rfflearn_cpu_gp         import RFFGPR, ORFGPR, RFFGPC, ORFGPC
from .rfflearn_cpu_pca        import RFFPCA, ORFPCA
from .rfflearn_cpu_cca        import RFFCCA, ORFCCA

# Import optuna-related modules if `optuna` is available.
if pkgutil.get_loader("optuna") is not None:

    from ..tuner import tuner

    RFFRegression_tuner = functools.partial(tuner.RFF_dim_std_tuner,     model = RFFRegression)
    ORFRegression_tuner = functools.partial(tuner.RFF_dim_std_tuner,     model = RFFRegression)
    RFFSVC_tuner        = functools.partial(tuner.RFF_dim_std_tuner,     model = RFFSVC)
    ORFSVC_tuner        = functools.partial(tuner.RFF_dim_std_tuner,     model = ORFSVC)
    RFFGPC_tuner        = functools.partial(tuner.RFF_dim_std_err_tuner, model = RFFGPC)
    ORFGPC_tuner        = functools.partial(tuner.RFF_dim_std_err_tuner, model = ORFGPC)
    RFFGPR_tuner        = functools.partial(tuner.RFF_dim_std_err_tuner, model = RFFGPR)
    ORFGPR_tuner        = functools.partial(tuner.RFF_dim_std_err_tuner, model = ORFGPR)

##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
