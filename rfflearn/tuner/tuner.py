#!/usr/bin/env python3
#
# Hyper parameter tuner for RFF models based on Optuna <https://optuna.org>.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : January 08, 2021
##################################################### SOURCE START #####################################################


import optuna


### List of possible arguments of optuna.study.Study.optimize.
LIST_ARGS_OPT = ["callbacks", "catch", "gc_after_trial", "n_trials", "show_progress_bar", "timeout"]


### Hyper parameter tuner for RFFRegression, ORFRegression, RFFSVC and ORFSVC.
def RFF_dim_std_tuner(train_set, valid_set, model_class, **kwargs):

    ### Define default arguments.
    args_par = {"dtype_dim_kernel": "int",
                "range_dim_kernel": (128, 1024),
                "dtype_std_kernel": "loguniform",
                "range_std_kernel": (1e-3, 1.0)}

    ### Update parameter arguments and delete from the `kwargs` variable.
    for key in args_par:
        if key in kwargs:
            args_par[key] = kwargs.pop(key)

    ### Split arguments to arguments for hyper parameter tuning and arguments for model fit.
    args_opt = {key:kwargs[key] for key in LIST_ARGS_OPT if key in kwargs}
    args_fit = {key:kwargs[key] for key in kwargs if key not in args_opt}

    ### The objective function for hyper parameter tuning.
    def objective(trial):

        ### Define optuna variable.
        dim_kernel = eval("trial.suggest_{dtype_dim_kernel}('dim_kernel', *{range_dim_kernel})".format(**args_par))
        std_kernel = eval("trial.suggest_{dtype_std_kernel}('std_kernel', *{range_std_kernel})".format(**args_par))

        ### Create classifier instance.
        model = model_class(dim_kernel = dim_kernel, std_kernel = std_kernel, **args_fit)

        ### Train classifire, calculate score and return the score.
        score = model.fit(*train_set).score(*valid_set)

        ### Set model instance as a user attribute.
        trial.set_user_attr("model", model)

        return score

    ### Callback function to save only the best model.
    def callback(study, trial):
        model = trial.user_attrs.pop("model")
        if study.best_trial.number == trial.number:
            study.set_user_attr("best_model", model)

    ### Run parameter search.
    study = optuna.create_study(direction = "maximize")
    study.optimize(objective, callbacks = [callback], **args_opt)

    return study


### Hyper parameter tuner for RFFGPR, ORFGPR, RFFGPC and ORFGPC.
def RFF_dim_std_err_tuner(train_set, valid_set, model_class, **kwargs):

    ### Define default arguments.
    args_par = {"dtype_dim_kernel": "int",
                "range_dim_kernel": (128, 1024),
                "dtype_std_error" : "loguniform",
                "range_std_error" : (1e-4, 0.1),
                "dtype_std_kernel": "loguniform",
                "range_std_kernel": (1e-3, 1.0)}

    ### Update parameter arguments and delete from the `kwargs` variable.
    for key in args_par:
        if key in kwargs:
            args_par[key] = kwargs.pop(key)

    ### Split arguments to arguments for hyper parameter tuning and arguments for model fit.
    args_opt = {key:kwargs[key] for key in LIST_ARGS_OPT if key in kwargs}
    args_fit = {key:kwargs[key] for key in kwargs if key not in args_opt}

    ### The objective function for hyper parameter tuning.
    def objective(trial):

        ### Define optuna variable.
        dim_kernel = eval("trial.suggest_{dtype_dim_kernel}('dim_kernel', *{range_dim_kernel})".format(**args_par))
        std_kernel = eval("trial.suggest_{dtype_std_kernel}('std_kernel', *{range_std_kernel})".format(**args_par))
        std_error  = eval("trial.suggest_{dtype_std_error} ('std_error',  *{range_std_error} )".format(**args_par))

        ### Create classifier instance.
        model = model_class(dim_kernel = dim_kernel, std_kernel = std_kernel, std_error = std_error, **args_fit)

        ### Train classifire, calculate score and return the score.
        score = model.fit(*train_set).score(*valid_set)

        ### Set model instance as a user attribute.
        trial.set_user_attr("model", model)

        return score

    ### Callback function to save only the best model.
    def callback(study, trial):
        model = trial.user_attrs.pop("model")
        if study.best_trial.number == trial.number:
            study.set_user_attr("best_model", model)

    ### Run parameter search.
    study = optuna.create_study(direction = "maximize")
    study.optimize(objective, callbacks = [callback], **args_opt)

    return study


##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
