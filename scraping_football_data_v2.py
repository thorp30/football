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

#Using requests to make html request
data = requests.get(team_urls[19])

#`creating df of general stats
matches = pd.read_html(data.text, match="Scores & Fixtures")[0]




"""

This following section uses a direct pd.read_html lookup onto the html and takes data from the 
shooting, passing, pass types, goal and shot creation, defensive action, possession and misc stats. 

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


#Shooting Stats
for i in range(np.size(team_urls)):   
    temp_shooting = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2022-2023/matchlogs/all_comps/shooting/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    
#Goalkeeper Stats
for i in range(np.size(team_urls)):   
    temp_keeper = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2022-2023/matchlogs/all_comps/keeper/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]

#Passing Stats
for i in range(np.size(team_urls)):   
    temp_pass = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2022-2023/matchlogs/all_comps/passing/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]

#Possession Stats
for i in range(np.size(team_urls)):   
    temp_poss = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2022-2023/matchlogs/all_comps/possession/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]

#Misc Stats
for i in range(np.size(team_urls)):   
    temp_misc = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2022-2023/matchlogs/all_comps/misc/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]

#Def Stats
for i in range(np.size(team_urls)):   
    temp_defence = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2022-2023/matchlogs/all_comps/defense/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]


# Next thing is to pick out key variables from each stat page



# temp.columns = temp.columns.droplevel()
# team_data = matches.merge(temp[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")











 