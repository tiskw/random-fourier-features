#!/usr/bin/env python3
#
# __init__.py file for the module 'rfflearn.gpu'.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : January 08, 2021
##################################################### SOURCE START #####################################################

import functools
import pkgutil

# Import RFF-related modules
from .rfflearn_gpu_common import seed
from .rfflearn_gpu_svc    import RFFSVC, ORFSVC
from .rfflearn_gpu_gp     import RFFGPR, ORFGPR, RFFGPC, ORFGPC, GPKernelParameterEstimator

# Import optuna-related modules if `optuna` is available.
if pkgutil.get_loader("optuna") is not None:

    from ..tuner import tuner

    RFFSVC_tuner = functools.partial(tuner.RFF_dim_std_tuner,     model = RFFSVC)
    ORFSVC_tuner = functools.partial(tuner.RFF_dim_std_tuner,     model = ORFSVC)
    RFFGPC_tuner = functools.partial(tuner.RFF_dim_std_err_tuner, model = RFFGPC)
    ORFGPC_tuner = functools.partial(tuner.RFF_dim_std_err_tuner, model = ORFGPC)
    RFFGPR_tuner = functools.partial(tuner.RFF_dim_std_err_tuner, model = RFFGPR)
    ORFGPR_tuner = functools.partial(tuner.RFF_dim_std_err_tuner, model = ORFGPR)

##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
