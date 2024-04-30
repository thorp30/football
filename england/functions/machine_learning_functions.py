"""

Python script created by Tom Thorp 22/04/2024.

This script contains a selection of python functions to undertaken machine learning from the
scikit-learn library.

"""

# Importing the required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def compare_models(matches_rolling, predictor_indices, target_index, num_folds=10, scoring='accuracy'):
    """
    Compare different machine learning models using k-fold cross-validation.

    Parameters:
    - matches_rolling (DataFrame): The input DataFrame containing the rolling matches data.
    - predictor_indices (list): The indices of the predictor variables in the DataFrame.
    - target_index (int): The index of the target variable in the DataFrame.
    - num_folds (int): The number of folds for cross-validation. Default is 10.
    - scoring (str): The scoring metric to evaluate the models. Default is 'accuracy'.

    Returns:
    - model_results (list): A list of strings summarizing the results for each model.

    Raises:
    - Exception: If an error occurs while evaluating a model.

    """
    # Define the models to compare
    model_list = [
        ('Logistic Regression', LogisticRegression()),
        ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
        ('K-Nearest Neighbours', KNeighborsClassifier()),
        ('Classification & Regression Trees', DecisionTreeClassifier()),
        ('Random Forest Classifier', RandomForestClassifier()),
        ('Naive Bayes', GaussianNB()),
        ('Support Vector Machines', SVC()),
        ('Extra Tree Classifier', ExtraTreesClassifier()),
        ('AdaBoost Classifier', AdaBoostClassifier()),
        ('Gradient Boosting', GradientBoostingClassifier())
    ]
    # Convert DataFrame to NumPy array
    matches_array = matches_rolling.values
    # Separate array for predictor variables and target variable
    predictor_variables = matches_array[:, predictor_indices]
    target_variable = matches_array[:, target_index].astype('int')
    # Initialize lists to store the results
    cv_scores = []
    model_names = []
    model_results = []
    # Evaluate each model in turn
    for name, model in model_list:
        try:
            # Perform k-fold cross-validation
            kfold = KFold(n_splits=num_folds)
            model_cv_scores = cross_val_score(model, predictor_variables, target_variable, cv=kfold, scoring=scoring, n_jobs=-1)
            # Calculate the mean and standard deviation of the cross-validation scores
            mean_cv_score = round(model_cv_scores.mean()*100, 4)  # mean across each fold
            std_cv_score = round(model_cv_scores.std()*100, 4) 
            # Store the results
            cv_scores.append(model_cv_scores)
            model_names.append(name)
            # Create a string with the results for this model
            result_summary = f"Using {name} the model correctly predicted the outcome of the match {mean_cv_score} % of the time with a standard deviation of {std_cv_score}"
            model_results.append(result_summary)
            # Print the results for this model
            print(result_summary)
        except Exception as e:
            # If an error occurred while evaluating the model, print the error message
            print(f"Error occurred while evaluating model {name}: {e}")
    # Return the results for all models
    return model_results



def calculate_rolling_average(matches_data: pd.DataFrame, new_cols: list) -> pd.DataFrame:
    """
    Calculate the rolling average of the last 3 games for each predictor variable.

    Parameters:
    - matches_data (DataFrame): The dataframe containing the matches data.
    - predictor_columns (list): The list of predictor columns to calculate the rolling average for.

    Returns:
    - rolling_average_appended (DataFrame): The dataframe with team names and the rolling average values for each predictor variable.
    """
    # Create a dataframe of last 3 weeks of games
    recent_matches = matches_data.groupby(["Team"]).tail(3)
    # Create a rolling average of the last 3 games for each predictor variable as a new row
    rolling_average_initial = recent_matches[new_cols].rolling(3, closed='left').mean().iloc[::3, :]
    # Reset index and drop first row as this is just the mean of the first 2 games which is NaN
    rolling_average_reset = rolling_average_initial.reset_index(drop=True)[1:]
    # Append the mean values of the last 3 rows to the end of the dataframe
    rolling_average_appended = rolling_average_reset.append(recent_matches[new_cols].tail(3).mean(), ignore_index=True)
    # Prepend team names to dataframe. This now includes all team names, and the last 3 games rolling average value for each predictor variable
    rolling_average_appended.insert(0, "Team", matches_data["Team"].unique())
    return rolling_average_appended



def merge_past_future_fixtures(past_matches: pd.DataFrame, future_predictor_averages: pd.DataFrame, future_fixtures: pd.DataFrame) -> pd.DataFrame:
    """
    Merge past matches, future predictor averages, and future fixtures into a complete dataset.

    Args:
        past_matches (DataFrame): DataFrame containing past match data.
        future_predictor_averages (DataFrame): DataFrame containing future predictor averages.
        future_fixtures (DataFrame): DataFrame containing future fixture data.

    Returns:
        DataFrame: Complete dataset with merged data.

    """
    # Define the columns we will be using
    predictor_columns = ['Team', 'GF_rolling', 'GA_rolling', 'xG_x_rolling', 'Poss_rolling','SoT_rolling', 'xA_rolling', 'Err_rolling', 'KP_rolling', 'Cmp_rolling','PrgP_rolling', 'PPA_rolling', 'Att Pen_rolling', 'Clr_rolling','Recov_rolling', 'Fls_rolling', 'result_code_rolling', 'Date','Opponent', 'Venue', 'venue_code', 'opp_code', 'target']
    # Merge the future fixtures with the mean predictor variables from the rolling averages, drop the target column, append the new data to the old data, and reset the index
    complete_dataset = pd.merge(future_predictor_averages, future_fixtures, left_on='Team', right_on='Team') \
        .drop("Target", axis=1).append(past_matches[predictor_columns]).reset_index(drop=True)
    return complete_dataset



#Create predicting function 
def make_future_predictions(data, predictors):
    """
    Make future predictions using a machine learning model.

    Args:
        data (pandas.DataFrame): The input data containing the features and target variable.
        predictors (list): A list of column names representing the features used for prediction.

    Returns:
        tuple: A tuple containing the combined DataFrame of actual and predicted values, and the precision score.

    """
    train = data[data["Date"] < str(yesterday)]
    test = data[data["Date"] > str(yesterday)]
    etc.fit(train[predictors], train["target"])
    preds = etc.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    error = precision_score(test["target"], preds)
    return combined, error