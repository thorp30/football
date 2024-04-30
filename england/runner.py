"""

Python script created by Tom Thorp 18/04/2024.

Provide a brief description of the script here.

"""

# Importing the required libraries
import sys

# setting the path to the functions folder for the runner.py file to find the functions
sys.path.insert(0, "/Users/tom/Documents/python/football/football_ML/england/functions/")

# Importing all functions required
from functions.scraping_football_data import *
from functions.scraping_future_fixtures import *
from functions.helper_functions import *
from functions.stats_functions import *
from functions.machine_learning_functions import *


"""

The first section of the runner.py script will scrape the data from the fbref website, and is stored in a pandas dataframe. 
The data will then be saved as a csv file in the assets folder, read back in (this step is put in to allow the user 
to choose which data to read in), and then formatted to make it more readable and to keep only the columns that are 
required for the model.

"""


# Get the URLs of all the teams in the Premier League
team_urls = get_team_urls()

# Get the squad team information from the team URLs
squad_team = get_squad_team(team_urls)

# Provide a blank list to store the match data for each of the teams
all_matches = []

# Loop through the team URLs to get the data for each team
for i in range(20): #20 teams in the premier league
        get_team_data(i, team_urls, squad_team, all_matches)
        if (i + 1) % 5 == 0:  # +1 because range starts from 0
            time.sleep(10)

#Export the data to a csv file
export_data(all_matches)

# Read the data from the csv  - because the data might not be in the environment we are working in, 
# we will read the data from the csv file that was saved in the assets/matches_csv folder in the previous step

# use the helper function to find the most recent file in matches - this is so that it can read in data that is made 
# on a different day

path = '/Users/tom/Documents/python/football/football_ML/england/assets/matches'
matches = pd.read_csv(get_latest_file(path))

# Format the data to make it more readable and keep only the columns that are required for the next steps
matches = format_data(matches)


"""

This next section will undertake a number of statistical analysis techniques on the data. This will enable us to
gain a better understanding of the data, and to identify any trends or patterns that may be present. The statistics are
undertaken as follows: 

- Feature selection (univariate selection using Chi-squared test, and feature importance using Extra Trees Classifier)

Based on the results, new variables will be created which provide rolling averages for the most important
variables determined by the feature selection process. This rolling variable process is designed to account for form, and can provide 
variables for future predictions where the result is unknown.

"""


# Setting the location within the DF of the predictor and target variables - this will be used for both the Chi-squared test and
# the feature importance using Extra Trees Classifier
predictor_loc = [np.where(matches.columns == i)[0][0] for i in predictor_list]
target_loc = np.where(matches.columns == 'target')[0][0]

# Perform feature selection using Chi-squared test
importance_by_chi_df = feature_selection_chi(matches, predictor_loc, target_loc, 10)
importance_by_chi_df = importance_by_chi_df.rename(index=predictor_dict)

# Perform feature importance using Extra Trees Classifier

#create model variable and use to fit the data 
model = ExtraTreesClassifier()
model.fit(predictor_variables, target_variable[:,0])

# Perform feature importance using Extra Trees Classifier
importance_by_extra_trees_df = feature_selection_extra_trees(matches, predictor_loc, target_loc)
importance_by_extra_trees_df = importance_by_extra_trees_df.rename(index=predictor_dict)

#create a seperate DF for each team using the groupby function
grouped_matches = matches.groupby("Team")

#Choose variables to compute rolling averages - these are taken from feature selection process rankings
cols = ["GF", "GA", "Poss", "SoT", "xA", "Err" , "KP", "PrgP", "PPA", "Att Pen", "Clr", "Recov","Fls" ,"result_code"]
#Create new columns for the rolling averages with the same name as the original columns but with "_rolling" appended
new_cols = [f"{c}_rolling" for c in cols]

#Create a new dataframe to store the rolling averages
matches_rolling = matches.groupby("Team").apply(lambda x: rolling_averages(x, cols, new_cols)).droplevel('Team').reset_index(drop=True)


"""

Following this resampling methods are used to estimate how well the algorithm will perform on unseen data (i.e. training on one portion of
the dataset, and testing on another). We can then retrain the model on the entire dataset, to prepare it for a "proper" prediction. The 
literature states that the best model evaluation tool is k-fold cross-validation.  

The literature states that the best model evaluation tool is k-fold cross-validation.  

- The first method used is a logistic regression model and the train_test_split library from sklearn.
- The second method used is a k-fold cross validation. 

"""


# Setting the location of the rolling predictor and target variables in the matches.rolling df - this will be used for both the 
# logistic regression model and the k-fold cross validation. Rolling predictor list is set in helper_functions.py file
rolling_predictor_loc = [np.where(matches_rolling.columns == i)[0][0] for i in rolling_predictor_list]
rolling_target_loc = np.where(matches_rolling.columns == 'target')[0][0]

#Undertake a logistic regression model using the train_test_split library from sklearn
perform_logistic_regression(matches_rolling, rolling_predictor_loc, rolling_target_loc)

#This method uses the K-Fold cross-validation method. This works by splitting the dataset into k-parts (e.g. k=5), and 
#each split is called a fold. The model is trained on k-1 folds with one held back to be tested upon. This is repeated so each 
#fold is held back to be the test dataset.

perform_k_cross_validation(matches_rolling,rolling_predictor_loc, rolling_target_loc, num_folds=10, random_state=7)


"""

This next section compares multiple machine learning algorithms on the dataset. These provide examples of a range of model
types available through the skikit-learn library (https://scikit-learn.org/stable/supervised_learning.html).

 The different algorithsm tested are: 
    
    Logistic regression (linear)
    Linear discriminant analysis (linear)
    k-nearest neighbours (nearest-neighbour)
    classification and regression trees (decision tree)
    random forest classification (ensemble - bagging)
    naive bayes (Naive bayes)
    support vector machines (support vector machines)
    extra trees (ensemble - bagging)
    adaBoost (ensemble - boosting)
    stochastic gradient boosting (ensemble - boosting)

At the end of this, we will have a variable called best_performing_model_name which will contain the model name for the next step.
    
"""

# Compare the different machine learning models using k-fold cross-validation
model_results = compare_models(matches_rolling, rolling_predictor_loc, rolling_target_loc, num_folds=10, scoring='accuracy')

# Extract the best result from the model results
best_performing_model = max(model_results, key=extract_accuracy)

# Create a var called best performing model name which we will fill in the loop below based on the model_results
best_performing_model_name = None

# Loop through the keys in ml_model_dict to find the best performing model, and store the details in a variable
for model_name in ml_model_dict.keys():
    # Check if the model name is in the best result
    if model_name in best_performing_model:
        best_performing_model_name = ml_model_dict[model_name]
        print(f"Best performing model is {model_name} with details: {ml_model_dict[model_name]}")
        break
else:
    print("No matching model found")

"""

This section scrapes the future fixtures for the next week from the fbref website. The fixtures are stored in a pandas dataframe,
and then exported as a csv file with today's date in the file name and stored in the assets folder.

"""

# Get the upcoming fixtures for in their raw form for the next "user-defined" days ahead
fixtures = get_upcoming_fixtures(3) # Default is 2 days ahead, but is user defined

# Create a dataframe of fixtures with selected columns that are required for the model
fixtures_df = create_fixtures_df(fixtures)

# Export the fixtures dataframe to a csv file
export_fixtures(fixtures_df)


"""

This final section will combine the past fixtures and the future fixtures and use the results from the ML model to predict the
outcomes of the future fixtures. The results will be stored in a pandas dataframe.

"""

# First create a list of predictor variables adding venue code and opposition code to the list of previously created rolling averages.
final_predictor_vars = ["venue_code", "opp_code"] + new_cols

# Create a df of just the last 3 matches for each team for each of the predictor variables - but not venue or opposition
recent_matches_rolling = calculate_rolling_average(matches_rolling, new_cols)

# Merge the recent matches with the future fixtures that will be predicted
full_match_dataset = merge_past_future_fixtures(recent_matches_rolling, final_predictor_vars, fixtures_df)

# Make predictions
make_future_predictions(full_match_dataset, final_predictor_vars)


"""
# Write all info into a auto report with stats, and future predictions 
"""



