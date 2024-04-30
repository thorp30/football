"""

Python script created by Tom Thorp 18/04/2024.

This script contains some general purpose helper functions that are used in the main script, and 
other useful variables that are used throughout the script.

"""

# Import the required libraries
import os
import glob
import re


def get_latest_file(path):
    """
    Get the path of the most recently created file in the specified directory.

    Args:
        path (str): The directory path to search for files.

    Returns:
        str: The path of the most recently created file.

    Raises:
        ValueError: If the specified directory is empty.

    """
    # Get a list of all files in the directory
    files = glob.glob(path + '/*')

    if not files:
        raise ValueError("The specified directory is empty.")

    # Find the most recently created file
    latest_file = max(files, key=os.path.getctime)
    return latest_file


"""

The below predictor and target related variables are used in the feature selection building step.

"""

# Define the predictor variable that will be used for feature selection
predictor_list = ['GF','GA','xG','xGA','Poss','Sh','SoT', 'Dist',
       'G-xG', 'xA', 'KP', '1/3', 'PPA', 'PrgP', 'Def Pen', 'Att Pen', 'Succ%',
       'CrdY', 'CrdR', 'Fls', 'Fld', 'Off', 'TklW', 'Recov', 'Int', 'Clr',
       'Err', 'venue_code', 'opp_code']

# Create a dictionary of the full names of the predictor variables
predictor_dict = {'GF':'Goals For','GA':'Goals Against','xG':'Expected Goals','xGA':'Expected Goals Against',
                    'Poss':'Possession','Sh':'Shots','SoT':'Shots on Target','Dist':'Average Shot Distance',
                    'G-xG':'Goals minus Expected Goals','xA':'Expected Assists','KP':'Key Passes','1/3':'Passes into Final Third',
                    'PPA':'Passes into Penalty Area','PrgP':'Progressive Passes','Def Pen':'Defensive Penalty Area Touches',
                    'Att Pen':'Attacking Penalty Area Touches','Succ%':'Success Take on %','CrdY':'Yellow Cards',
                    'CrdR':'Red Cards','Fls':'Fouls Commited','Fld':'Fouls Drawn','Off':'Offsides','TklW':'Tackles Won',
                    'Recov':'Ball Recoveries','Int':'Interceptions','Clr':'Clearances','Err':'Errors','venue_code':'Venue',
                    'opp_code':'Opponent'}

#Define the rolling predictor variables that will be used for the logistic regression model

rolling_predictor_list = ['GF_rolling','GA_rolling','Poss_rolling','SoT_rolling','xA_rolling',
                          'Err_rolling','KP_rolling', 'PrgP_rolling','PPA_rolling','Att Pen_rolling',
                          'Clr_rolling','Recov_rolling','Fls_rolling','result_code_rolling']



"""

The below pieces of information are used to extract the accuracy from the result string returned by the model evaluation - 
this is then used to determine which is the best performing ML model. 

"""

def extract_accuracy(result):
    """
    Extracts the accuracy value from the given result string.

    Args:
        result (str): The result string containing the accuracy value.

    Returns:
        float: The extracted accuracy value as a float. Returns 0 if no match is found.
    """
    match = re.search(r'(\d+\.\d+) %', result)
    return float(match.group(1)) if match else 0

# Dictionary of model names in string form and their function names
ml_model_dict = {
    'Logistic Regression': LogisticRegression(),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'K-Nearest Neighbours': KNeighborsClassifier(),
    'Classification & Regression Trees': DecisionTreeClassifier(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machines': SVC(),
    'Extra Tree Classifier': ExtraTreesClassifier(),
    'AdaBoost Classifier': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}


"""

The below variables, functions and dictionaries are used for the final prediction step.

"""

#Create a dictionary then use the pandas map function to match all teams names. Then apply to combined. 
#For example Brighton and Hove Albion -> Brighton. This allows for team and opponent to be called the same, and can be merged on. 
class MissingDict(dict):
    __missing__ = lambda self, key: key

team_map_values = {"Brighton and Hove Albion": "Brighton", "Manchester United": "Manchester Utd", "Newcastle United": "Newcastle Utd",
              "Nottingham Forest":"Nott'ham Forest" , "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham",
              "Wolverhampton Wanderers": "Wolves", "Sheffield United": "Sheffield Utd"} 
team_mapping = MissingDict(**team_map_values)