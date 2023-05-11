"""
The __init__.py file for the 'rfflearn.cpu' module.
"""

# Import standard packages.
import functools
import pkgutil
import sys

# Import RFF-related modules.
from .rfflearn_cpu_common     import seed
from .rfflearn_cpu_regression import RFFRegression, ORFRegression
from .rfflearn_cpu_svc        import RFFSVC, ORFSVC, QRFSVC, RFFBatchSVC, ORFBatchSVC, QRFBatchSVC
from .rfflearn_cpu_gp         import RFFGPR, ORFGPR, QRFGPR, RFFGPC, ORFGPC, QRFGPC
from .rfflearn_cpu_pca        import RFFPCA, ORFPCA
from .rfflearn_cpu_cca        import RFFCCA, ORFCCA


# Import optuna-related modules if `optuna` is available.
if pkgutil.get_loader("optuna") is not None:

    from ..tuner import tuner

    RFFRegression_tuner = functools.partial(tuner.RFF_dim_std_tuner,     model_class = RFFRegression)
    ORFRegression_tuner = functools.partial(tuner.RFF_dim_std_tuner,     model_class = RFFRegression)
    RFFSVC_tuner        = functools.partial(tuner.RFF_dim_std_tuner,     model_class = RFFSVC)
    ORFSVC_tuner        = functools.partial(tuner.RFF_dim_std_tuner,     model_class = ORFSVC)
    RFFGPC_tuner        = functools.partial(tuner.RFF_dim_std_err_tuner, model_class = RFFGPC)
    ORFGPC_tuner        = functools.partial(tuner.RFF_dim_std_err_tuner, model_class = ORFGPC)
    RFFGPR_tuner        = functools.partial(tuner.RFF_dim_std_err_tuner, model_class = RFFGPR)
    ORFGPR_tuner        = functools.partial(tuner.RFF_dim_std_err_tuner, model_class = ORFGPR)

else: print("rfflearn.cpu: package 'optuna' not found. SKip loading tuner submodule.", file = sys.stderr)

# Import shap-related modules if `shap` is available.
if pkgutil.get_loader("shap") is not None:

    from ..explainer.shap        import shap_feature_importance, shap_plot
    from ..explainer.permutation import permutation_feature_importance, permutation_plot

else: print("rfflearn.cpu: package 'shap' not found. SKip loading explainer submodule.", file = sys.stderr)


# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
