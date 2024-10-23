"""
The __init__.py file for the 'rfflearn.cpu' module.
"""

# Import standard packages.
import functools
import importlib
import pathlib
import sys

# Import RFF-related modules.
from .rfflearn_cpu_common     import seed
from .rfflearn_cpu_regression import RFFRegressor, ORFRegressor, QRFRegressor
from .rfflearn_cpu_svc        import RFFSVC, ORFSVC, QRFSVC
from .rfflearn_cpu_svr        import RFFSVR, ORFSVR, QRFSVR
from .rfflearn_cpu_gp         import RFFGPR, ORFGPR, QRFGPR, RFFGPC, ORFGPC, QRFGPC
from .rfflearn_cpu_pca        import RFFPCA, ORFPCA, QRFPCA
from .rfflearn_cpu_cca        import RFFCCA, ORFCCA, QRFCCA

# Append the root directory of rfflearn to Python path.
sys.path.append(str(pathlib.Path(__file__).parent.parent))

# Import version info.
rfflearn_common = importlib.import_module("rfflearn_common")
__version__ = rfflearn_common.__version__

# Declare published functions and variables.
__all__ = [
    "__version__", "seed",
    "RFFRegressor", "ORFRegressor", "QRFRegressor",
    "RFFSVC", "ORFSVC", "QRFSVC",
    "RFFSVR", "ORFSVR", "QRFSVR",
    "RFFGPR", "ORFGPR", "QRFGPR", "RFFGPC", "ORFGPC", "QRFGPC",
    "RFFPCA", "ORFPCA", "QRFPCA",
    "RFFCCA", "ORFCCA", "QRFCCA",
]

# Import optuna-related modules if `optuna` is available.
if importlib.util.find_spec("optuna") is not None:

    # Import tuner module.
    tuner   = importlib.import_module("tuner")
    hptuner = tuner.hptuner

    RFFRegressor_tuner = functools.partial(hptuner.RFF_dim_std_tuner, model_class=RFFRegressor)
    ORFRegressor_tuner = functools.partial(hptuner.RFF_dim_std_tuner, model_class=ORFRegressor)
    QRFRegressor_tuner = functools.partial(hptuner.RFF_dim_std_tuner, model_class=QRFRegressor)
    RFFSVC_tuner       = functools.partial(hptuner.RFF_dim_std_tuner, model_class=RFFSVC)
    ORFSVC_tuner       = functools.partial(hptuner.RFF_dim_std_tuner, model_class=ORFSVC)
    QRFSVC_tuner       = functools.partial(hptuner.RFF_dim_std_tuner, model_class=QRFSVC)
    RFFGPC_tuner       = functools.partial(hptuner.RFF_dim_std_err_tuner, model_class=RFFGPC)
    ORFGPC_tuner       = functools.partial(hptuner.RFF_dim_std_err_tuner, model_class=ORFGPC)
    QRFGPC_tuner       = functools.partial(hptuner.RFF_dim_std_err_tuner, model_class=QRFGPC)
    RFFGPR_tuner       = functools.partial(hptuner.RFF_dim_std_err_tuner, model_class=RFFGPR)
    ORFGPR_tuner       = functools.partial(hptuner.RFF_dim_std_err_tuner, model_class=ORFGPR)
    QRFGPR_tuner       = functools.partial(hptuner.RFF_dim_std_err_tuner, model_class=QRFGPR)

    __all__ += [
        "RFFRegressor_tuner", "ORFRegressor_tuner", "QRFRegressor_tuner",
        "RFFSVC_tuner", "ORFSVC_tuner", "QRFSVC_tuner",
        "RFFGPC_tuner", "ORFGPC_tuner", "QRFGPC_tuner",
        "RFFGPR_tuner", "ORFGPR_tuner", "QRFGPR_tuner",
    ]

else:
    sys.stderr.write("rfflearn.cpu: package 'optuna' not found. Skip loading tuner submodule.\n")

# Import shap-related modules if `shap` is available.
if importlib.util.find_spec("shap") is not None:

    # Import explainer module.
    explainer = importlib.import_module("explainer")

    # Load all available variables/functions in the explainer module.
    names = [name for name in explainer.__dict__ if not name.startswith("_")]

    # Update the global dictionary.
    globals().update({name: getattr(explainer, name) for name in names})

    # Update the published variable/function names.
    __all__ += names

else:
    sys.stderr.write("rfflearn.cpu: package 'shap' not found. SKip loading explainer submodule.\n")

# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
