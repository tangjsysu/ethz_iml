# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from fancyimpute import IterativeImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score
from sklearn.kernel_ridge import KernelRidge
import miceforest as mf
from sklearn.preprocessing import StandardScaler

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("../data/train_2.csv")

    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')

    # Load test data
    test_df = pd.read_csv("../data/test_2.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Perform data preprocessing, imputation and extract X_train, y_train and X_test
    train_df.replace({'spring': 1, 'summer': 2, 'autumn': 3, 'winter': 4}, inplace=True)
    test_df.replace({'spring': 1, 'summer': 2, 'autumn': 3, 'winter': 4}, inplace=True)
    train_names = train_df.columns
    transformer_train = IterativeImputer()
    transformer_test = IterativeImputer()
    ii_train = transformer_train.fit_transform(train_df.drop(['season'], axis=1))
    ii_test = transformer_test.fit_transform(test_df.drop(['season'], axis=1))
    ii_train = pd.DataFrame(ii_train, columns=train_names[1:])
    swiss_ele = ii_train['price_CHF']
    ii_train = ii_train.drop(["price_CHF"], axis=1)
    ii_test = pd.DataFrame(ii_test)
    X_train = np.column_stack((train_df['season'].values, ii_train.values))
    y_train = swiss_ele
    X_test = np.column_stack((test_df['season'].values, ii_test.values))

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (
                X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test
def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    # y_pred=np.zeros(X_test.shape[0])
    kf = KFold(n_splits=10)
    best_accuracy = 0
    for j, (train_idx, test_idx) in enumerate(kf.split(X_train)):
        X_subtrain, X_subtest = X_train[train_idx], X_train[test_idx]
        y_subtrain, y_subtest = y_train[train_idx], y_train[test_idx]
        gpr = GaussianProcessRegressor(kernel=RationalQuadratic(), alpha=0.15)
        gpr.fit(X_subtrain, y_subtrain)
        y_pred = gpr.predict(X_subtest)
        r2 = r2_score(y_subtest, y_pred)
        print("The accuracy of the Gaussian Process is: ", r2)
        if r2 > best_accuracy:
            best_accuracy = r2
            best_model = gpr
    # TODO: Define the model and fit it using training data. Then, use test data to make predictions
    y_pred = best_model.predict(X_test)
    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred = modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred)
    dt.columns = ['price_CHF']
    dt.to_csv('../data/results_task2a.csv', index=False)
    print("\nResults file successfully generated!")

