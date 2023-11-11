"""
Created on Fri Jun 02 18:55 2023

@author: tom
"""

# -*- coding: utf-8 -*-
"""
Python script created by Tom Thorp 02/06/2023
. 
This script reads in future fixtures from the fbref website and outputs them to a csv file ready for use in the
predicting script.
"""

# Importing libraries
import pandas as pd
import numpy as np
import requests 
from bs4 import BeautifulSoup
import time 
from datetime import date

#To get data we will use pandas read_html function to read in the table from the website.
#We will use the match argument to find the table we want.

# Defining the url of the website, directing to league specific stats tab.
url = "https://fbref.com/en/comps/24/schedule/Serie-A-Scores-and-Fixtures"

#Using pands to read in the table from the website, and convert to df
matches = pd.read_html(url, match = "Scores & Fixtures")
matches = pd.DataFrame(matches[0])

#We only want the future fixtures, so we will remove all games wehre the date is in the past.
matches = pd.DataFrame(matches[0])
today = date.today()
today = today.strftime("%Y-%m-%d")

#Removing all games before todays date
matches = matches[matches["Date"] >= today]

#We now want the games in the next week, so we will remove all games after 1 week from now.
#Getting the date 1 week from now
next_week = date.today() + pd.DateOffset(weeks=1)
next_week = next_week.strftime("%Y-%m-%d")

#Removing all games after 1 week from now
matches = matches[matches["Date"] <= next_week]

"""
We now have the fixtures for the next week, but we need to get the df into the correct format to export
for further use in the predicting script. We will need to create a df with the following columns:
    - Date (in the format DD/MM/YYYY)
    - Team
    - Opponent
    - Venue (home or away)
    - Target (always 'W') - this is the target variable for the model

"""
#Creating a new df with the columns we want
fixtures = pd.DataFrame(columns = ["Date", "Team", "Opponent", "Venue", "Target"])

#Assign W to target, as this will always be the target variable for the model
fixtures["Target"] = "W"

#Creating a list of all the teams in the league taken from the matches df columns Home and Away.
#Assign this to the Team column in the fixtures df.
teams_home = matches["Home"].unique()
teams_away = matches["Away"].unique()
teams = np.concatenate((teams_home, teams_away), axis = 0)
fixtures["Team"] = teams

#Create a list of all opponents. This is taken from the matches df columns Home and Away.
#Assign this to the Opponent column in the fixtures df.
opponents_home = matches["Away"].unique()
opponents_away = matches["Home"].unique()
opponents = np.concatenate((opponents_home, opponents_away), axis = 0)
fixtures["Opponent"] = opponents

#Work out if venue is home or away. This is done by checking if the team is in the home column of the matches df.
#If it is, then the venue is home, if not then it is away.
fixtures["Venue"] = np.where(fixtures["Team"].isin(matches["Home"]), "Home", "Away")

#We now need to get the date of the games from the matches df, and assign this to the Date column in the fixtures df.
#We will do this by looping through the fixtures df, and for each row, checking if the team is in the home column of the matches df.
#If it is, then the date is taken from the Date column of the matches df where the team is in the home column.
#If not, then the date is taken from the Date column of the matches df where the team is in the away column.
for i in range(len(fixtures)):
    if fixtures.iloc[i,1] in matches["Home"].values:
        fixtures.iloc[i,0] = matches[matches["Home"] == fixtures.iloc[i,1]]["Date"].values[0]
    else:
        fixtures.iloc[i,0] = matches[matches["Away"] == fixtures.iloc[i,1]]["Date"].values[0]

#Now we need to reformat the date column to be in the format DD/MM/YYYY
fixtures["Date"] = pd.to_datetime(fixtures["Date"])
fixtures["Date"] = fixtures["Date"].dt.strftime("%d/%m/%Y")

"""
The final thing we must do us use the mapping dictionary to map the team names to the names used in the model - this 
is for the teams in the opponent column only. This below code has been taken from the predicting script. 

"""
#Mapping dictionary to map team names to the names used in the model
class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {"America MG": "América (MG)", "Atletico Goianiense": "Atl Goianiense", "Atletico Mineiro": "Atlético Mineiro",
              "Atletico Paranaense":"Atl Paranaense" , "Avai": "Avaí", "Botafogo RJ": "Botafogo (RJ)",
              "Ceara": "Ceará", "Cuiaba": "Cuiabá" , "Goias": "Goiás" , "Gremio": "Grêmio" , "Sao Paulo": "São Paulo"} 
mapping = MissingDict(**map_values)

#Mapping the opponent column using the mapping dictionary
fixtures["Opponent"] = fixtures["Opponent"].map(mapping)

#Exporting the fixtures df to a csv file
fixtures.to_csv("football_ML/brazil/fixtures.csv", index = False)