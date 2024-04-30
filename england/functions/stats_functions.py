"""

Python script created by Tom Thorp 21/04/2024.

This script contains a selection of python functions to undertaken statistical analysis to help
better understand the data ahead of building the machine learning model.

"""

#Importing the required libraries
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

from helper_functions import *

def feature_selection_chi(matches: pd.DataFrame, predictor_loc: list, target_loc: list, k: int) -> pd.DataFrame:
    """
    Performs feature selection on a DataFrame using Chi-Squared test.
    
    Parameters:
    matches (pd.DataFrame): The input DataFrame.
    predictor_loc (list): The locations of the predictor variables in the DataFrame.
    target_loc (list): The location of the target variable in the DataFrame.
    k (int): The number of top features to select.
    
    Returns:
    pd.DataFrame: A DataFrame containing the Chi-Squared scores of the features, sorted by importance.
    """
    
    # Convert dataframe into np array
    matches_array = matches.values
    # Separate array for predictor variables and target variable
    predictor_variables = matches_array[:,predictor_loc]
    # Normalise predictor variables to between 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))
    predictor_variables = scaler.fit_transform(predictor_variables)
    # Convert target value to int
    target_variable = matches_array[:,target_loc].astype('int')
    # Feature extraction
    test = SelectKBest(score_func=chi2, k=k) # k variable determines number of top features to select
    fit = test.fit(predictor_variables, target_variable)
    # Set print arguments
    np.set_printoptions(precision=3, suppress=True)
    # Summarize scores using f strings
    for i in range(len(predictor_loc)):
        print(f"Variable {matches.columns[predictor_loc[i]]} has a score of {fit.scores_[i]}")
    # Put into simple df, so we can sort by importance     
    importance_by_chi_df = pd.DataFrame(fit.scores_,matches.columns[predictor_loc]).sort_values(by=[0],ascending=False)
    return importance_by_chi_df


def feature_selection_extra_trees(matches: pd.DataFrame, predictor_loc: list, target_variable: pd.Series) -> pd.DataFrame:
        """
        Perform feature selection using Extra Trees Classifier.

        Parameters:
        - matches (pandas.DataFrame): The dataset containing the predictor variables.
        - predictor_loc (list): The indices of the predictor variables in the dataset.
        - target_variable (pandas.Series): The target variable.

        Returns:
        - importance_by_extra_trees_df (pandas.DataFrame): A DataFrame containing the feature importance scores
            sorted in descending order.
        """
        # Separate array for predictor variables and target variable
        predictor_variables = matches.iloc[:, predictor_loc]
        # Convert target value to int
        target_variable = matches_array[:,target_loc].astype('int')
        # Create model variable and use to fit 
        model = ExtraTreesClassifier()
        model.fit(predictor_variables, target_variable)
        # Set print arguments
        np.set_printoptions(precision=3, suppress=True)
        # Print feature importance scores
        for i in range(len(predictor_loc)):
                print(f"Variable {matches.columns[predictor_loc[i]]} has a score of {model.feature_importances_[i]}")
        # Put into simple df, so we can sort by importance     
        importance_by_extra_trees_df = pd.DataFrame(model.feature_importances_, index=matches.columns[predictor_loc]).sort_values(by=0, ascending=False)
        return importance_by_extra_trees_df


def rolling_averages(group: pd.DataFrame, cols: list, new_cols: list) -> pd.DataFrame:
    """
    Calculate rolling averages for specified columns over a window of 3 observations.

    Parameters:
    - group (DataFrame): The input DataFrame containing the data to be processed.
    - cols (list): A list of column names for which rolling averages need to be calculated.
    - new_cols (list): A list of new column names to store the calculated rolling averages.

    Returns:
    - group (DataFrame): The updated DataFrame with the calculated rolling averages.
    """
    # Sort the DataFrame 'group' by the "Date" column in ascending order.
    group = group.sort_values("Date")
    # Calculate the rolling average of the columns specified in 'cols' over a window of 3 observations.
    # The 'closed' parameter is set to 'left', which means that the window includes the current observation and the two previous ones.
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    # Assign the calculated rolling averages to the new columns specified in 'new_cols' in the DataFrame 'group'.
    group[new_cols] = rolling_stats
    # Remove any rows in the DataFrame 'group' that have missing values in the new columns.
    group = group.dropna(subset=new_cols)
    # Return the updated DataFrame.
    return group


def perform_logistic_regression(matches_rolling, predictor_indices, target_index, test_size=0.2, random_state=7):
    """
    Performs logistic regression on a DataFrame of matches.
    
    Parameters:
    matches_rolling (pd.DataFrame): The DataFrame containing the match data.
    predictor_indices (list): The indices of the predictor variables.
    target_index (list): The index of the target variable.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): The seed used by the random number generator.
    
    Returns:
    float: The accuracy of the logistic regression model.
    """
    # Convert DataFrame to NumPy array
    matches_array = matches_rolling.values
    # Separate array for predictor variables and target variable
    predictor_variables = matches_array[:, predictor_indices]
    target_variable = matches_array[:, target_index].astype('int')
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(predictor_variables, target_variable, test_size=test_size, random_state=random_state)
    # Create and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    # Calculate the accuracy of the model
    accuracy = model.score(X_test, Y_test)
    result_percent = np.around(accuracy * 100,2)
    #Add a print statement to show the accuracy of the model and explain what it means
    print(f"Accuracy of {result_percent} %, this means that the model correctly predicted the outcome of the match {result_percent} % of the time using the rolling averages of the predictor variables")


def perform_k_cross_validation(matches_rolling, predictor_indices, target_index, num_folds=10, random_state=7):
    """
    Perform k-fold cross-validation using logistic regression.

    Parameters:
    - matches_rolling (DataFrame): The rolling matches data.
    - predictor_indices (list): The indices of the predictor variables in the matches data.
    - target_index (int): The index of the target variable in the matches data.
    - num_folds (int): The number of folds for cross-validation. Default is 10.
    - random_state (int): The random state for reproducibility. Default is 7.

    Returns:
    - results_mean (float): The mean accuracy of the cross-validation scores.
    - results_std (float): The standard deviation of the cross-validation scores.
    """
    # Convert DataFrame to NumPy array
    matches_array = matches_rolling.values
    # Separate array for predictor variables and target variable
    predictor_variables = matches_array[:, predictor_indices]
    target_variable = matches_array[:, target_index].astype('int')
    # Create a KFold object
    kfold = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)
    # Create a logistic regression model
    model = LogisticRegression()
    # Perform cross-validation
    results = cross_val_score(model, predictor_variables, target_variable, cv=kfold)
    # Calculate the mean and standard deviation of the cross-validation scores
    results_mean = np.around(results.mean() * 100,2)
    results_std = np.around(results.std() * 100,2)
    # Add a print statement to show the accuracy of the model and explain what it means
    print(f"Using k-fold cross-validation, the model correctly predicted the outcome of the match {results_mean} % of the time with a standard deviation of {results_std} %")

    
