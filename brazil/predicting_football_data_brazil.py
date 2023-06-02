# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:00:49 2023

@author: tom
"""


"""

This script:
    
    takes a csv file of information relating to football matches (output from scraping_football_data_v2.py)
    
    formats the data into a more usable design
    
    introduces feature selection
    
    estimates the performance of ML models using k-fold cross validation
    
    Evaluates a selection of ML approaches on their ability to predict the target variable
    
   Uses optimal ML appraoches to try to predict whether the team will win their next game. 

"""

# Importing libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

#read in previously made csv file - work 
# matches = pd.read_csv(r"C:\Users\tt13\football\matches_20230214.csv", index_col=0)


# Read in historic season data for 2021 and 2022
matches_2021 = pd.read_csv('/Users/tom/Documents/python/football/brazil/brazil_matches_2021.csv', index_col=0)
matches_2022 = pd.read_csv('/Users/tom/Documents/python/football/brazil/brazil_matches_2022.csv', index_col=0)

# Read in the matches for this season so far
matches_2023 = pd.read_csv('/Users/tom/Documents/python/football/brazil/brazil_matches_20230502.csv', index_col=0)

# Concat all 3 csv files together into 1
matches = matches_2021.append([matches_2022,matches_2023])

"""

This next section first drops the variables which will not be used and then creates predictor variables from string variables
which will be of use going forward. 

"""

#Drop columns that are not needed
matches = matches.drop(columns=['Captain','Formation','Referee','Match Report', 'Notes'])

#convert date column into datetime 
matches["Date"] = pd.to_datetime(matches["Date"])

#convert the string venue variable into integer categories and create a unique indicator for each outcome option. 0 = Away, 1 = Home
matches["venue_code"] = matches["Venue"].astype("category").cat.codes

#convert the string opponent variable into integer categories and create a unique indicator for each outcome option.
matches["opp_code"] = matches["Opponent"].astype("category").cat.codes

#Create a target column, which is when the result column shows W (0,1 for L, W) 
matches["target"] = (matches["Result"] == "W").astype("int")

#convert the string result variable into integer categories (0,1,2 for L,D,W)
matches["result_code"] = matches["Result"].astype("category").cat.codes


"""

This next section undertakes feature selection. This enables automatic selection of the features within the data that contribute most 
to the predictor variable to be selected. Feature selection can lead to a reduction in overfitting, imrpove model accuracy and reduce training
time if there is a large dataset. This will be done to pick out further variables to train the ML algorithm. 

First feature selection undertaken is univariate selection 
Second feature selection undertaken is feature importance

(Feature selection adapted from Jason Brownlee book "Machine Learning Mastery with Python") 

"""

#Undertaking univariate selection using chi-squared

#Convert dataframe into np array
array = matches.values

#Provide the location within the array of all potential predictor variables, and the location of target variable
predictor_loc = np.append(np.append([7,8,10,11,12,14,15,16,], np.arange(18,41)), [42,43])
target_loc = [44] 

#Seperate array for predictor variables and target variable
predictor_variables = array[:,predictor_loc]

#Normalise predictor variables to between 0-1
scaler = MinMaxScaler(feature_range=(0, 1))
predictor_variables = scaler.fit_transform(predictor_variables)

#Convert target value to int
target_variable = array[:,target_loc].astype('int')

# feature extraction
test = SelectKBest(score_func=chi2, k=10) #k variable determines number of top features to select
fit = test.fit(predictor_variables, target_variable)

# summarize scores usig f strings
for i in np.arange(np.size(predictor_loc)):
    np.set_printoptions(precision=3, suppress='True') #set print arguments
    string = f"Variable {matches.columns[predictor_loc[i]]}. has a score of {fit.scores_[i]}"
    print(string)
    
#put into simple df, so we can sort by importance     
importance_by_chi_df = pd.DataFrame(fit.scores_,matches.columns[predictor_loc]).sort_values(by=[0],ascending=False)





#Undertaking Feature importance with Extra Trees Classifier.  
#This uses the same predictor_variables and target_variable from previous example

#create model variable and use to fit 
model = ExtraTreesClassifier()
model.fit(predictor_variables, target_variable)

for i in np.arange(np.size(predictor_loc)):
    np.set_printoptions(precision=3, suppress='True') #set print arguments
    string = f"Variable {matches.columns[predictor_loc[i]]}. has a score of {model.feature_importances_[i]}"
    print(string)

#put into simple df, so we can sort by importance     
importance_by_extra_trees_df = pd.DataFrame(model.feature_importances_,matches.columns[predictor_loc]).sort_values(by=[0],ascending=False)


"""

Based on the results from the previous section, we will now create new variables which provide rolling averages for the most important
variables determined by the feature selection process. This process is designed to account for form, and can provide variables for 
future predictions where the result is unknown. 

"""

#The next section creates 3-match rolling variables to account for form

#create a seperate dataframe for each team, and group these 
grouped_matches = matches.groupby("Team")

#Create a function which takes each team in grouped_matches and creates additional 3 match rolling averages for each team for all
#variables defined in cols. 

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("Date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group


#Choose variables to compute rolling averages - these are taken from feature selection process rankings
cols = ["GF", "GA", "xGA", "xG_x", "Poss", "SoT", "xA", "Err" , "KP", "Cmp", "PrgP", "PPA", "Att Pen", "Clr", "Dist" ,"Recov","Fls" ,"result_code"]
new_cols = [f"{c}_rolling" for c in cols]

#Apply rolling averages to all teams grouped by Team variable
matches_rolling = matches.groupby("Team").apply(lambda x: rolling_averages(x, cols, new_cols))

#Drop team being index level
matches_rolling = matches_rolling.droplevel('Team')

#Set indexing column to go from 0 -> 
matches_rolling.index = range(matches_rolling.shape[0])


"""

This section uses resampling methods to estimate how well the algorithm will perform on unseen data (i.e. training on one portion of
the dataset, and testing on another). We can then retrain the model on the entire dataset, to prepare it for a "proper" prediction.

The literature states that the best model evaluation tool is k-fold cross-validation.  

The first method used is a logistic regression model and the train_test_split library from sklearn.
The second method used is a k-fold cross validation. 


"""

#This first method uses the train_test_split library from sklearn in conjunction with the logistic regression function. 

#Convert dataframe into np array
array = matches_rolling.values

#Provide the location within the array of all potential predictor variables, and the location of target variable
predictor_loc = np.append([42,43], np.arange(46,61))
target_loc = [44] 

#Seperate array for predictor variables and target variable
predictor_variables = array[:,predictor_loc]

#Convert target value to int
target_variable = array[:,target_loc].astype('int')

#Decide size of testing (80% training, 20% testing in this case
test_size = 0.2
random_state = 7 #this controls the shuffling process. If this is set to 0, the test/train sets are always the same. 

X_train, X_test, Y_train, Y_test = train_test_split(predictor_variables, target_variable, test_size=test_size,
random_state=random_state)

#Create regression model and create the result
model = LogisticRegression()
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
result_percent = result *100

print(f"Accuracy of {result_percent} %")



#This method uses the K-Fold cross-validation method. This works by splitting the dataset into k-parts (e.g. k=5), and 
#each split is called a fold. The model is trained on k-1 folds with one held back to be tested upon. This is repeated so each 
#fold is held back to be the test dataset. 

#This reuses the predictor_variable and target_variable from the previous section

num_folds = 10
random_state = 7
kfold = KFold(n_splits=num_folds)
model = LogisticRegression()
results = cross_val_score(model, predictor_variables, target_variable, cv=kfold)
results_mean = results.mean()*100
results_std = results.std()*100

print(f"Accuracy of {results_mean} %, with a std of {results_std}")



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
    


"""

#predictor_variables and target_variable taken from examples above

#To compare all models quickly, loop through 
#create empty model array
models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
models.append(('K-Nearest Neighbours', KNeighborsClassifier()))
models.append(('Classification & Regression Trees', DecisionTreeClassifier()))
models.append(('Random Forest Classifier', RandomForestClassifier()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Support Vector Machines', SVC()))
models.append(('Extra Tree Classifier', ExtraTreesClassifier()))
models.append(('AdaBoost Classifier', AdaBoostClassifier()))
models.append(('Gradient Boosting', GradientBoostingClassifier()))

# evaluate each model in turn
results = []
names = []
result_string = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(model, predictor_variables, target_variable, cv=kfold, scoring=scoring)
    cv_results_mean = round(cv_results.mean(),4) #mean across each fold
    cv_results_std = round(cv_results.std(),4)
    results.append(cv_results)
    names.append(name)
    string = f"{name}: mean accurary: {cv_results_mean}, std: {cv_results_std}"
    result_string.append(string)
    print(string)


"""

This next section trains the ML model to create a set of predictions using the previously created predictor variables. Based on the results
from the previous section, the highest scoring ML algorithm is the Extra Tree Classifier, therefore this is used going forwards. This is a dummy
section used as testing hence why sections commented out. Still needs to be run due to the dictionary section.   

This section is adapted from: 
https://github.com/dataquestio/project-walkthroughs/blob/master/football_matches/prediction.ipynb

"""

#Set this to the highest performing model as defined in the previous section! 
# etc = GaussianNB(n_estimators=50, min_samples_split=10, random_state=1)
etc = LogisticRegression()

#Create a list of predictor variables adding venue code and opposition code to the list of previously created rolling averages.
predictors = ["venue_code", "opp_code"] + new_cols


#Create predicting function 
# def make_predictions(data, predictors):
#     train = data[data["Date"] < '2022-11-25']
#     test = data[data["Date"] > '2022-11-25']
#     etc.fit(train[predictors], train["target"])
#     preds = etc.predict(test[predictors])
#     combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
#     error = precision_score(test["target"], preds)
#     return combined, error

#use the make predictions function to...... make predictions
#combined, error = make_predictions(matches_rolling, predictors)

#Add some more useful information to the predictions for better understanding
#combined = combined.merge(matches_rolling[["Date", "Team", "Opponent", "Result"]], left_index=True, right_index=True)

#Create a dictionary then use the pandas map function to match all teams names. Then apply to combined. 
#For example Brighton and Hove Albion -> Brighton. This allows for team and opponent to be called the same, and can be merged on. 
class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {"America MG": "América (MG)", "Atletico Goianiense": "Atl Goianiense", "Atletico Mineiro": "Atlético Mineiro",
              "Atletico Paranaense":"Atl Paranaense" , "Avai": "Avaí", "Botafogo RJ": "Botafogo (RJ)",
              "Ceara": "Ceará", "Cuiaba": "Cuiabá" , "Goias": "Goiás" , "Gremio": "Grêmio" , "Sao Paulo": "São Paulo"} 
mapping = MissingDict(**map_values)

#combined["new_team"] = combined["Team"].map(mapping)

#Merge df on itself to tidy up, and match up on duplicate predections (i.e. H v A and A v H)
#merged = combined.merge(combined, left_on=["Date", "new_team"], right_on=["Date", "Opponent"])

#outcome = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] ==0)]["actual_x"].value_counts()

#accuracy = (outcome[1]/np.sum((outcome[1],outcome[0])))*100


"""

This next section will require some manual input of the next round of fixtures (e.g. team, opposition, venue). A new dataframe will be created
containing the next weeks fixtures and all the predictor statistics will be created, then this will be appended to the bottom of the 
matches_rolling df to create predictions. 


"""

#create a dataframe of last 3 weeks of games
future_matches = matches_rolling.groupby(["Team"]).tail(3)
print(future_matches["Team"].value_counts()) #Sanity check it is bringing in all teams last 3 games

future_matches_predictors = future_matches[new_cols]

future_matches_predictors_mean = future_matches_predictors.rolling(3, closed='left').mean() 
future_matches_predictors_mean_1 = future_matches_predictors_mean.iloc[::3, :]

#Set indexing column to go from 0 -> 
future_matches_predictors_mean_1.index = range(future_matches_predictors_mean_1.shape[0])

future_matches_predictors_mean_1 = future_matches_predictors_mean_1[1:]

future_matches_predictors_mean_end = future_matches_predictors.tail(3).mean()

future_matches_predictors_mean_final = future_matches_predictors_mean_1.append(future_matches_predictors_mean_end,  ignore_index=True)

#Create List of Unique Team names to re append
team_names = matches_rolling["Team"].unique()

#Prepend team names to dataframe 
future_matches_predictors_mean_final.insert(0,"Team",team_names)

#Bring in this weeks games (created manually externally)
#future_fixtures = pd.read_csv(r"C:\Users\tt13\football\fixtures.csv")

future_fixtures = pd.read_csv('/Users/tom/Documents/python/football/brazil/fixtures.csv')

#convert date column into datetime 
future_fixtures["Date"] = pd.to_datetime(future_fixtures["Date"],yearfirst=True, dayfirst=True)

# future_fixtures["Date"] = pd.to_datetime(future_fixtures["Date"], format ="%Y/%m/%d")

#convert the string venue variable into integer categories and create a unique indicator for each outcome option. 0 = Away, 1 = Home
future_fixtures["venue_code"] = future_fixtures["Venue"].astype("category").cat.codes

#convert the string opponent variable into integer categories and create a unique indicator for each outcome option.
future_fixtures["opp_code"] = future_fixtures["Opponent"].astype("category").cat.codes

#Create a target column, which is when the result column shows W (0,1 for L, W) 
future_fixtures["target"] = (future_fixtures["Target"] == "W").astype("int")


future_matches_predictors_mean_final_2 = pd.merge(future_matches_predictors_mean_final,future_fixtures, left_on='Team',right_on='Team')
future_matches_predictors_mean_final_2 = future_matches_predictors_mean_final_2.drop("Target",axis=1)

#Now append "new future data" to previous data. 

#Take just the columns we will be using 
matches_rolling_short = matches_rolling[['Team', 'GF_rolling', 'GA_rolling', 'xG_x_rolling', 'xGA_rolling' , 'Dist_rolling' ,'Poss_rolling','SoT_rolling', 'xA_rolling', 'Err_rolling', 'KP_rolling', 'Cmp_rolling','PrgP_rolling', 'PPA_rolling', 'Att Pen_rolling', 'Clr_rolling','Recov_rolling', 'Fls_rolling', 'result_code_rolling', 'Date','Opponent', 'Venue', 'venue_code', 'opp_code', 'target']]

full_matches_dataset = pd.concat([matches_rolling_short,future_matches_predictors_mean_final_2])

full_matches_dataset = full_matches_dataset.reset_index()

# full_matches_dataset = full_matches_dataset.drop(columns=["index", "level_0"])

#Create predicting function 
def make_future_predictions(data, predictors):
    train = data[data["Date"] < '2023-05-01']
    test = data[data["Date"] > '2023-05-01']
    etc.fit(train[predictors], train["target"])
    preds = etc.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    error = precision_score(test["target"], preds)
    return combined, error

#use the make predictions function to...... make predictions
combined, error = make_future_predictions(full_matches_dataset, predictors)

#Add some more useful information to the predictions for better understanding
combined = combined.merge(full_matches_dataset[full_matches_dataset["Date"] > '2023-05-01'][["Date", "Team", "Opponent"]], left_index=True, right_index=True)

#Drop actual - as this has not happened
combined = combined.drop("actual",axis=1)

combined["new_team"] = combined["Team"].map(mapping)

#Merge df on itself to tidy up, and match up on duplicate predections (i.e. H v A and A v H)
merged = combined.merge(combined, left_on=["Date", "new_team"], right_on=["Date", "Opponent"])

outcome = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] ==0)]

outcome = outcome.reset_index()

outcome['Date'] = outcome['Date'].dt.strftime('%d/%m/%Y')

# Output results to text file 

final_predictions =[]

for i in outcome.index:
    winner = outcome["Team_x"][i]
    loser = outcome["Opponent_x"][i]
    date = outcome['Date'][i]
    string = f"{winner} are predicted to beat {loser} on the {date}"
    final_predictions.append("")
    final_predictions.append(string)
    final_predictions.append("")
    final_predictions.append('---------------------')
    
final_predictions = pd.DataFrame(final_predictions)
final_predictions.to_csv("/Users/tom/Documents/python/football/brazil/final_predictions_03052023.csv")
