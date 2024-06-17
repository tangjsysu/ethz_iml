from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


# Add any additional imports here (however, the task is solvable without using
# any additional imports)
# import ...


def calculate_RMSE(w, X, y):
    """This function takes test data points (X and y), and computes the empirical RMSE of
    predicting y from X using a linear model with weights w.

    Parameters
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression
    X: matrix of floats, dim = (15,13), inputs with 13 features
    y: array of floats, dim = (15,), input labels

    Returns
    ----------
    RMSE: float: dim = 1, RMSE value
    """
    RMSE = 0
    sum = 0
    y_pred = w.predict(X)
    residuals = y - y_pred
    squared_errors = residuals ** 2
    mean_squared_error = np.mean(squared_errors)
    RMSE = np.sqrt(mean_squared_error)
    assert np.isscalar(RMSE)
    return RMSE


def average_LR_RMSE(X, y, lambdas, n_folds):
    """
    Main cross-validation loop, implementing 10-fold CV. In every iteration (for every train-test split), the RMSE for every lambda is calculated,
    and then averaged over iterations.

    Parameters
    ----------
    X: matrix of floats, dim = (150, 13), inputs with 13 features
    y: array of floats, dim = (150, ), input labels
    lambdas: list of floats, len = 5, values of lambda for which ridge regression is fitted and RMSE estimated
    n_folds: int, number of folds (pieces in which we split the dataset), parameter K in KFold CV

    Returns
    ----------
    avg_RMSE: array of floats: dim = (5,), average RMSE value for every lambda
    """
    RMSE_mat = np.zeros((n_folds, len(lambdas)))
    kf = KFold(n_splits=n_folds)
    for i, lam in enumerate(lambdas):
        for j, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            clf = Ridge(alpha=lam,solver="svd", fit_intercept=False)
            w = clf.fit(X_train, y_train)
            RMSE = calculate_RMSE(w, X_test, y_test)
            RMSE_mat[j, i] = RMSE

    # TODO: Enter your code here. Hint: Use functions 'fit' and 'calculate_RMSE' with training and test data
    # and fill all entries in the matrix 'RMSE_mat'

    avg_RMSE = np.mean(RMSE_mat, axis=0)
    assert avg_RMSE.shape == (5,)
    return avg_RMSE


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("../data/train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns="y")
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function calculating the average RMSE
    lambdas = [0.1, 1, 10, 100, 200]
    n_folds = 13
    avg_RMSE = average_LR_RMSE(X, y, lambdas, n_folds)
    # Save results in the required format
    np.savetxt("./results1a_2.csv", avg_RMSE, fmt="%.12f")
