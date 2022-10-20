#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 22:22:51 2022

@author: tom
"""

# -*- coding: utf-8 -*-
"""
Python script created by Tom Thorp 04/09/2022. 
This script reads in stats about a football league, and sorts them into panda dataframes,
saving them in csv output for further use in external scripts (e.g. plotting and ML). 
"""

# Importing libraries
import pandas as pd
import numpy as np
import requests 
from bs4 import BeautifulSoup
import os

""" 

Below scraping method adapted from (link continues over 3 lines): 
https://github.com/dataquestio/project-walkthroughs/blob/master/football_matches/scraping.ipynb
This provides the unique team url to scrape intial match data for each team in url link. 

"""
# Defining the url of the website, directing to league specific stats tab.
url = "https://fbref.com/en/comps/9/Premier-League-Stats"

#Using requests to make html request 
data = requests.get(url)

#Convert html text into string
soup = BeautifulSoup(data.text)
standing_table = soup.select("table.stats_table")[0]

#Picking out "a tag" which is the anchor item - picking out all atag items. 
links  = standing_table.find_all("a")

#Looping through links variable to get the href property from the atag.
links = [l.get("href") for l in links]
links = [l for l in links if '/squads/' in l]

#Prepending website domain to the link to provide full link. 
team_urls = [f"https://fbref.com{l}" for l in links]



"""
This following section creates a simple lookup array listing the team name and thier unique lookup code

"""
#Creating an array of squad codes
squad_code = []
for i in range(np.size(team_urls)):
    temp = team_urls[i].split("/")[-2].replace("-Stats", "")
    squad_code = np.append(squad_code, temp)
 
#Creating an array of team names
team_name = []
for i in range(np.size(team_urls)):
    temp_team_name = team_urls[i].split("/")[-1].replace("-Stats", "")
    team_name = np.append(team_name,temp_team_name)

#Joining team name and codes to lookup on to create unique team url for data
squad_team = np.column_stack((squad_code,team_name))



"""

This following section uses a direct pd.read_html lookup onto the html and takes data from the 
shooting, passing, pass types, goal and shot creation, defensive action, possession and misc stats. 
This loops through each team, appending the results to the variable all_matches. This creates a list of 
dataframes which are then concatenated at the end to create one df with all data. 

"""

all_matches = []

#Shooting Stats
for i in range(np.size(team_urls)):   
    
    #Using requests to make html request
    data = requests.get(team_urls[i])

    #creating df of general stats from scores and fixtures tab
    matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
    
    #Shooting stats
    temp_shooting = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2022-2023/matchlogs/all_comps/shooting/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_shooting.columns = temp_shooting.columns.droplevel()
    try:
        team_data = matches.merge(temp_shooting[["Date", "Sh", "SoT", "Dist", "G-xG", "PK", "xG"]], on="Date")
    except ValueError:
        continue


    #Passing Stats 
    temp_pass = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2022-2023/matchlogs/all_comps/passing/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_pass.columns = temp_pass.columns.droplevel()
    try:
        team_data = team_data.merge(temp_pass[["Date", "Cmp", "KP", "1/3", "PPA", "Prog"]], on="Date")
    except ValueError:
        continue


    #Possession Stats   
    temp_poss = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2022-2023/matchlogs/all_comps/possession/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_poss.columns = temp_poss.columns.droplevel()
    try:
        team_data = team_data.merge(temp_poss[["Date", "Def Pen", "Att Pen", "Succ%"]], on="Date")
    except ValueError:
        continue


    #Misc Stats  
    temp_misc = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2022-2023/matchlogs/all_comps/misc/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_misc.columns = temp_misc.columns.droplevel()
    try:
        team_data = team_data.merge(temp_misc[["Date", "CrdY", "CrdR", "Fls", "Fld", "Off", "TklW", "Recov"]], on="Date")
    except ValueError:
        continue


    #Def Stats  
    temp_defence = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2022-2023/matchlogs/all_comps/defense/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_defence.columns = temp_defence.columns.droplevel()
    try:
        team_data = team_data.merge(temp_defence[["Date", "TklW", "Press", "Succ", "Int", "Clr", "Err"]], on="Date")
    except ValueError:
        continue  
    
    
    #only select matches where they are in the premier leage
    team_data = team_data[team_data["Comp"] == "Premier League"]
    
    #Add a column to the df naming the team 
    team_data["Team"] = squad_team[i][1].replace("-", " ")
    all_matches.append(team_data)
    print(i)


#Concatenate all seperate dataframes relating to each individual team into 1 large df.
match_df = pd.concat(all_matches)

match_df.to_csv("matches.csv")


# Next thing is to change the titles of the df items, so that they are more readable
# Then look at creating a jupyter notebook looking into the stats and potenitally averaging over teams 
# Also remove columns not needed, and add in rolling stats and also other columns (venue code, opponent, day code etc). 
# Basically convert all non number columns to numbers.






 