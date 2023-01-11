# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:00:49 2023

@author: tom
"""


"""

This script takes a csv file of information relating to football matches, formats the data into a more usable format, then uses
different ML appraoches to try to predict whether the teams will win their next game. 


"""

# Importing libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

#read in previously made csv file
matches = pd.read_csv(r"C:\Users\tt13\football\matches_20230106.csv", index_col=0)

"""

This next section drops the variables which will not be used and creates predictor variables which will be of use. 

"""

#Drop columns that are not needed
matches = matches.drop(columns=['Captain','Formation','Referee','Match Report', 'Notes'])

#convert date column into datetime 
matches["Date"] = pd.to_datetime(matches["Date"])

#convert the string venue variable into integer categories
matches["venue_code"] = matches["Venue"].astype("category").cat.codes

#convert the string opponent variable into integer categories
matches["opp_code"] = matches["Opponent"].astype("category").cat.codes

#Create a target column, which is when the result column shows W (0,1 for L, W) 
matches["target"] = (matches["Result"] == "W").astype("int")

#convert the string result variable into integer categories (0,1,2 for L,D,W)
matches["result_code"] = matches["Result"].astype("category").cat.codes

#The next section creates 3-match rolling variables to account for form

#create a group of df for each team 
grouped_matches = matches.groupby("Team")

#Create a function which takes each team in grouped_matches and creates additional 3 match rolling averages for each team. 

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("Date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group


#Choose variables to compute rolling averages
cols = ["GF", "GA", "Poss", "Sh", "KP", "Cmp", "PPA", "Att Pen", "Recov", "Clr", "result_code"]
new_cols = [f"{c}_rolling" for c in cols]

#Apply rolling averages to all teams grouped by Team variable
matches_rolling = matches.groupby("Team").apply(lambda x: rolling_averages(x, cols, new_cols))

#Drop team being index level
matches_rolling = matches_rolling.droplevel('Team')

#Set indexing column to go from 0 -> 
matches_rolling.index = range(matches_rolling.shape[0])


"""

This next section trains the ML model to create a set of predictions using the previously created predictor variables. 

"""

#Create a list of predictor variables
predictors = ["venue_code", "opp_code"] + new_cols


#Create predicting function 
def make_predictions(data, predictors):
    train = data[data["Date"] < '2022-11-15']
    test = data[data["Date"] > '2022-11-15']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    error = precision_score(test["target"], preds)
    return combined, error

#make predictions
combined, error = make_predictions(matches_rolling, predictors)

#Add some more useful information to the predictions for better understanding
combined = combined.merge(matches_rolling[["Date", "Team", "Opponent", "Result"]], left_index=True, right_index=True)


#Create a dictionary then use the pandas map function to match all teams names. Then apply to combined. For example Brighton and Hove Albion -> Brighton
class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {"Brighton and Hove Albion": "Brighton", "Manchester United": "Manchester Utd", "Newcastle United": "Newcastle Utd","Nottingham Forest":"Nott'ham Forest" , "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves"} 
mapping = MissingDict(**map_values)
combined["new_team"] = combined["Team"].map(mapping)

#Merge df on itself to tidy up, and match up on duplicate predections (i.e. H v A and A v H)
merged = combined.merge(combined, left_on=["Date", "new_team"], right_on=["Date", "Opponent"])







