"""
__init__.py file for the module 'rfflearn.gpu'.
"""

# Import standard libraries.
import functools
import importlib
import pathlib
import warnings
import sys

# Import RFF-related modules.
from .rfflearn_gpu_common import seed
from .rfflearn_gpu_svc    import RFFSVC, ORFSVC, QRFSVC
from .rfflearn_gpu_gp     import RFFGPR, ORFGPR, QRFGPR, RFFGPC, ORFGPC, QRFGPC
from .rfflearn_gpu_pca    import RFFPCA, ORFPCA, QRFPCA

# Append the root directory of rfflearn to Python path.
sys.path.append(str(pathlib.Path(__file__).parent.parent))

# Import version info.
rfflearn_common = importlib.import_module("rfflearn_common")
__version__ = rfflearn_common.__version__

# Declare published functions and variables.
__all__ = [
    "__version__", "seed",
    "RFFSVC", "ORFSVC", "QRFSVC",
    "RFFGPR", "ORFGPR", "QRFGPR", "RFFGPC", "ORFGPC", "QRFGPC",
    "RFFPCA", "ORFPCA", "QRFPCA",
]

# Import optuna-related modules if `optuna` is available.
if importlib.util.find_spec("optuna") is not None:

    # Import tuner module.
    tuner   = importlib.import_module("tuner")
    hptuner = tuner.hptuner

    RFFSVC_tuner = functools.partial(hptuner.RFF_dim_std_tuner,     model_class=RFFSVC)
    ORFSVC_tuner = functools.partial(hptuner.RFF_dim_std_tuner,     model_class=ORFSVC)
    QRFSVC_tuner = functools.partial(hptuner.RFF_dim_std_tuner,     model_class=QRFSVC)
    RFFGPC_tuner = functools.partial(hptuner.RFF_dim_std_err_tuner, model_class=RFFGPC)
    ORFGPC_tuner = functools.partial(hptuner.RFF_dim_std_err_tuner, model_class=ORFGPC)
    QRFGPC_tuner = functools.partial(hptuner.RFF_dim_std_err_tuner, model_class=QRFGPC)
    RFFGPR_tuner = functools.partial(hptuner.RFF_dim_std_err_tuner, model_class=RFFGPR)
    ORFGPR_tuner = functools.partial(hptuner.RFF_dim_std_err_tuner, model_class=ORFGPR)
    QRFGPR_tuner = functools.partial(hptuner.RFF_dim_std_err_tuner, model_class=QRFGPR)

    __all__ += [
        "RFFRegression_tuner", "ORFRegression_tuner",
        "RFFSVC_tuner", "ORFSVC_tuner", "QRFSVC_tuner",
        "RFFGPC_tuner", "ORFGPC_tuner", "QRFGPC_tuner",
        "RFFGPR_tuner", "ORFGPR_tuner", "QRFGPR_tuner",
    ]

else:
    warnings.warn("rfflearn.gpu: package 'optuna' not found. SKip loading tuner submodule.",
                  ImportWarning)

# Import shap-related modules if `shap` is available.
if all(importlib.util.find_spec(name) is not None for name in ["matplotlib", "shap"]):

    # Import explainer module.
    explainer = importlib.import_module("explainer")

    # Load all available variables/functions in the explainer module.
    names = [name for name in explainer.__dict__ if not name.startswith("_")]

    # Update the global dictionary.
    globals().update({name: getattr(explainer, name) for name in names})

    # Update the published variable/function names.
    __all__ += names

else:
    warnings.warn("rfflearn.gpu: package 'shap' not found. SKip loading explainer submodule.",
                  ImportWarning)

# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
