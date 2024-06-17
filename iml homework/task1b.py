# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import numpy as np
import pandas as pd

# Add any additional imports here (however, the task is solvable without using 
# any additional imports)
# import ...

def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X) 
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant feature: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((700, 21))
    x_squre = np.square(X)
    x_exp = np.exp(X)
    x_cos = np.cos(X)
    x_1 = np.ones((700, 1))
    X_transformed = np.concatenate((X,x_squre,x_exp,x_cos,x_1), axis=1)

    # TODO: Enter your code here
    assert X_transformed.shape == (700, 21)
    return X_transformed


def fit(X, y):
    """
    This function receives training data points, transforms them, and then fits the linear regression on this 
    transformed data. Finally, it outputs the weights of the fitted linear regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels)

    Returns
    ----------
    w: array of floats: dim = (21,), optimal parameters of linear regression
    """
    w = np.zeros((21,))
    X_transformed = transform_data(X)
    # TODO: Enter your code here
    XTX = np.dot(X_transformed.T, X_transformed)

    regularization = 320 * np.eye(21)
    A = XTX + regularization
    b = np.dot(X_transformed.T, y)
    w = np.linalg.solve(A, b)
    assert w.shape == (21,)
    return w

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
    y_pred = np.dot(X, w)
    residuals = y - y_pred
    squared_errors = residuals ** 2
    mean_squared_error = np.mean(squared_errors)
    RMSE = np.sqrt(mean_squared_error)
    assert np.isscalar(RMSE)
    return RMSE
# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("../data/train_1b.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit(X, y)
    X_transformed = transform_data(X)
    rmse = calculate_RMSE(w,X_transformed,y)
    print(rmse)
    # Save results in the required format
    np.savetxt("./results_1b_2.csv", w, fmt="%.12f")
