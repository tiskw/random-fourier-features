"""
Overview:
    Sample code for the rfflearn.explainer submodule.
    This code computes feature importances and visualization them.

Usage:
    feature_importanves_for_california_housing.py [--n_trials <int>] [--seed <int>]
    feature_importanves_for_california_housing.py (-h | --help)

Options:
    --seed <int>  Random seed.       [default: 111]
    -h, --help    Show this message.
"""

# Import standard libraries.
import pathlib
import sys

# Import 3rd-party packages.
import docopt
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing


def generate_california_housing_dataset(args):
    """
    Create California housing dataset instance.
    """
    # Load California Housing data from sklearn.
    data = sklearn.datasets.fetch_california_housing()

    # Split data to train and test.
    Xs_train, Xs_test, ys_train, ys_test \
        = sklearn.model_selection.train_test_split(data["data"], data["target"],
                                                   test_size=0.2, random_state=args["--seed"])

    # Data standardization.
    scaler   = sklearn.preprocessing.StandardScaler().fit(Xs_train)
    Xs_train = scaler.transform(Xs_train)
    Xs_test  = scaler.transform(Xs_test)

    return (Xs_train, Xs_test, ys_train, ys_test, data.feature_names)


def main(args):
    """
    Main procedure.
    """
    # Fix seed for random fourier feature calclation
    rfflearn.seed(int(args["--seed"]))

    # Prepare training data
    Xs_train, Xs_test, ys_train, ys_test, feature_names = generate_california_housing_dataset(args)

    # Create model instance.
    model = rfflearn.RFFRegressor(dim_kernel=128, std_kernel=0.05)

    # Train the model.
    model.fit(Xs_train, ys_train)

    # Conduct prediction for the test data
    pred_test = model.predict(Xs_test)
    score_r2  = sklearn.metrics.r2_score(ys_test, pred_test)
    print("- R2 score of the model: ", score_r2)

    # Calculate feature importance (SHAP and permutation importance).
    shap_values = rfflearn.shap_feature_importance(model, Xs_test)
    perm_values = rfflearn.permutation_feature_importance(model, Xs_test, ys_test)

    # Print feature importances.
    print("shap_values.shape =", shap_values.shape)
    print("perm_values.shape =", perm_values.shape)

    # Try to plot.
    rfflearn.permutation_plot(perm_values, feature_names, show=False)
    plt.close()


if __name__ == "__main__":

    # Append path to rfflearn directory.
    sys.path.append(str(pathlib.Path(__file__).parent.parent))

    # Parse input arguments.
    args = docopt.docopt(__doc__)

    # Import rfflearn.
    import rfflearn.cpu as rfflearn

    # Convert all arguments to an appropriate type.
    for k, v in args.items():
        try   : args[k] = eval(str(v))
        except: args[k] = str(v)

    # Run main procedure.
    main(args)


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
