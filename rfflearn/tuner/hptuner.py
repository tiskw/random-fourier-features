"""
Hyper parameter tuner for RFF models based on Optuna <https://optuna.org>.
"""

# Declare published functions and variables.
__all__ = ["RFF_dim_std_tuner", "RFF_dim_std_err_tuner"]

# Import 3rd-party packages.
import optuna


# List of possible arguments of optuna.study.Study.optimize.
SET_ARGS_OPT = {"callbacks", "catch", "gc_after_trial", "n_trials",
                "show_progress_bar", "timeout"}

# Define verbosity level.
VERBOSITY_LEVELS = {0: optuna.logging.CRITICAL,
                    1: optuna.logging.FATAL,
                    2: optuna.logging.ERROR,
                    3: optuna.logging.WARNING,
                    4: optuna.logging.INFO,
                    5: optuna.logging.DEBUG}


def get_suggest_fn(dtype, trial):
    """
    Returns appropriate suggest function.
    """
    suggest_functions = {
        "int"  : trial.suggest_int,
        "float": trial.suggest_float,
    }
    return suggest_functions.get(dtype, None)


def callback(study, trial):
    """
    Callback function to save only the best model.
    """
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_model", value=trial.user_attrs["model"])


def RFF_dim_std_tuner(model_class: type, train_set: tuple, valid_set: tuple,
                      verbose: int = 0, **kwargs: dict) -> optuna.study.Study:
    """
    Hyper parameter tuner for RFFRegression, ORFRegression, RFFSVC and ORFSVC.

    Args:
        model_class (type) : Target class of hyperparameter tuning.
        train_set   (tuple): A tuple of training data and label.
        valid_set   (tuple): A tuple of validation data and label.
        verbose     (int)  : Verbosity level (smaller is quieter).
        kwargs      (dict) : Keyword arguments for `model_class` or `optuna.study.Study.optimize`.

    Returns:
        (optuna.study.Study): Optimized study instance.
    """
    # Define default arguments.
    args_par = {"dtype_dim_kernel": "int",
                "range_dim_kernel": {"low": 128, "high": 1024},
                "dtype_std_kernel": "float",
                "range_std_kernel": {"low": 1e-3, "high": 1.0, "log": True}}

    # Update parameter arguments and delete from the `kwargs` variable.
    for key in args_par:
        if key in kwargs:
            args_par[key] = kwargs.pop(key)

    # Split arguments to arguments for hyper parameter tuning and arguments for model fit.
    args_opt = {key:val for key, val in kwargs.items() if key     in SET_ARGS_OPT}
    args_fit = {key:val for key, val in kwargs.items() if key not in SET_ARGS_OPT}

    def objective(trial: optuna.trial.Trial) -> float:
        """
        The objective function for hyper parameter tuning.

        Args:
            trial (optuna.trial.Trial): An object contains info of a trial.

        Returns:
            (float): Score of the trial.
        """
        # Define optuna variable for dim_kernel.
        suggest_fn = get_suggest_fn(args_par["dtype_dim_kernel"], trial)
        dim_kernel = suggest_fn("dim_kernel", **args_par["range_dim_kernel"])

        # Define optuna variable for std_kernel.
        suggest_fn = get_suggest_fn(args_par["dtype_std_kernel"], trial)
        std_kernel = suggest_fn("std_kernel", **args_par["range_std_kernel"])

        # Create classifier instance.
        model = model_class(dim_kernel=dim_kernel, std_kernel=std_kernel, **args_fit)

        # Train classifire, calculate score and return the score.
        score = model.fit(*train_set).score(*valid_set)

        # Set model instance as a user attribute.
        trial.set_user_attr(key="model", value=model)

        return score

    # Set verbosity level.
    optuna.logging.set_verbosity(VERBOSITY_LEVELS[verbose])

    # Run parameter search.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, callbacks=[callback], **args_opt)

    # Create new attribute.
    setattr(study, "best_model", study.user_attrs["best_model"])

    return study


def RFF_dim_std_err_tuner(model_class: type, train_set: tuple, valid_set: tuple,
                          verbose: int = 0, **kwargs: dict) -> optuna.study.Study:
    """
    Hyper parameter tuner for RFFGPR, ORFGPR, RFFGPC and ORFGPC.
    """
    # Define default arguments.
    args_par = {"dtype_dim_kernel": "int",
                "range_dim_kernel": {"low": 128, "high": 1024},
                "dtype_std_error" : "float",
                "range_std_error" : {"low": 1e-4, "high": 0.1, "log": True},
                "dtype_std_kernel": "float",
                "range_std_kernel": {"low": 1e-3, "high": 1.0, "log": True}}

    # Update parameter arguments and delete from the `kwargs` variable.
    for key in args_par:
        if key in kwargs:
            args_par[key] = kwargs.pop(key)

    # Split arguments to arguments for hyper parameter tuning and arguments for model fit.
    args_opt = {key:val for key, val in kwargs.items() if key     in SET_ARGS_OPT}
    args_fit = {key:val for key, val in kwargs.items() if key not in SET_ARGS_OPT}

    # The objective function for hyper parameter tuning.
    def objective(trial):

        # Define optuna variable for dim_kernel.
        suggest_fn = get_suggest_fn(args_par["dtype_dim_kernel"], trial)
        dim_kernel = suggest_fn("dim_kernel", **args_par["range_dim_kernel"])

        # Define optuna variable for std_kernel.
        suggest_fn = get_suggest_fn(args_par["dtype_std_kernel"], trial)
        std_kernel = suggest_fn("std_kernel", **args_par["range_std_kernel"])

        # Define optuna variable for std_error.
        suggest_fn = get_suggest_fn(args_par["dtype_std_error"], trial)
        std_error  = suggest_fn("std_error", **args_par["range_std_error"])

        # Create classifier instance.
        model = model_class(dim_kernel=dim_kernel, std_kernel=std_kernel,
                            std_error=std_error, **args_fit)

        # Train classifire, calculate score and return the score.
        score = model.fit(*train_set).score(*valid_set)

        # Set model instance as a user attribute.
        trial.set_user_attr("model", model)

        return score

    # Set verbosity level.
    optuna.logging.set_verbosity(VERBOSITY_LEVELS[verbose])

    # Run parameter search.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, callbacks=[callback], **args_opt)

    # Create new attribute.
    setattr(study, "best_model", study.user_attrs["best_model"])

    return study


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
