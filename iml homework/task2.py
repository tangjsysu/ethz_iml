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
    original_column_names = train_df.columns
    # Perform one-hot encoding on the 'season' column
    ordinal_encoder = OrdinalEncoder()
    train_df['season'] = ordinal_encoder.fit_transform(train_df[['season']])
    transformer = IterativeImputer()
    train_df = transformer.fit_transform(train_df)
    train_df = pd.DataFrame(train_df, columns=original_column_names)
    y_train = train_df['price_CHF']
    train_df = train_df.drop(['price_CHF'], axis=1)
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head())
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("../data/test_2.csv")
    test_df['season'] = ordinal_encoder.fit_transform(test_df[['season']])

    print("Test data:")
    print(test_df.shape)
    print(test_df.head())

    # Dummy initialization of the X_train, X_test and y_train
    # TODO: Depending on how you deal with the non-numeric data, you may want to

    X_train = train_df
    test_transformer = IterativeImputer()
    X_test = test_transformer.fit_transform(test_df)
    X_test = test_transformer.transform(X_test)
    print(X_test)



    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
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

    #y_pred=np.zeros(X_test.shape[0])
    X_train = X_train.values
    kf = KFold(n_splits=10)
    best_accuracy = 0
    for j, (train_idx, test_idx) in enumerate(kf.split(X_train)):
        X_subtrain, X_subtest = X_train[train_idx], X_train[test_idx]
        y_subtrain, y_subtest = y_train[train_idx], y_train[test_idx]
        gpr = GaussianProcessRegressor(kernel=DotProduct())
        gpr.fit(X_subtrain, y_subtrain)
        y_pred = gpr.predict(X_subtest)
        r2 = r2_score(y_subtest, y_pred)
        print("The accuracy of the Gaussian Process is: ", r2)
        if r2 > best_accuracy:
            best_accuracy = r2
            best_model = gpr
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions
    y_pred = best_model.predict(X_test)
    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred)
    dt.columns = ['price_CHF']
    dt.to_csv('../data/results_task2a.csv', index=False)
    print("\nResults file successfully generated!")

