#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday 02/05/2023

@author: tom
"""

# -*- coding: utf-8 -*-
"""
Python script created by Tom Thorp 02/05/2023. 
This script reads in stats about a football league, and sorts them into panda dataframes,
saving them in csv output for further use in external scripts (e.g. plotting and ML). 

This is an updated version of the script originally written for the English Premier League, to
work for the Brazilian Serie A. 

"""

# Importing libraries
import pandas as pd
import numpy as np
import requests 
from bs4 import BeautifulSoup
import time 

""" 

Below scraping method adapted from (link continues over 3 lines): 
https://github.com/dataquestio/project-walkthroughs/blob/master/football_matches/scraping.ipynb
This provides the unique team url to scrape intial match data for each team in url link. 

"""
# Defining the url of the website, directing to league specific stats tab.
url = "https://fbref.com/en/comps/24/Serie-A-Stats"

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

06/01/2023
Due to a HTTP request error, I have split this up into 4 sepearte loops, as 1 loop of 20 teams was too much. All
loops still append the df to the all_matches empty array. Sleep set 10 seems to stop this error from happening. 

"""

all_matches = []

#Loop 1, teams in team_url 1-5. 
#Shooting Stats
for i in np.arange(0,5):   
    
    #Using requests to make html request
    data = requests.get(team_urls[i])

    #creating df of general stats from scores and fixtures tab
    matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
    
    #Shooting stats
    temp_shooting = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/shooting/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_shooting.columns = temp_shooting.columns.droplevel()
    try:
        team_data = matches.merge(temp_shooting[["Date", "Sh", "SoT", "Dist", "G-xG", "xG"]], on="Date")
    except ValueError:
        continue


    #Passing Stats 
    temp_pass = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/passing/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_pass.columns = temp_pass.columns.droplevel()
    try:
        team_data = team_data.merge(temp_pass[["Date", "Cmp", "xA", "KP", "1/3", "PPA", "PrgP"]], on="Date")
    except ValueError:
        continue


    #Possession Stats   
    temp_poss = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/possession/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_poss.columns = temp_poss.columns.droplevel()
    try:
        team_data = team_data.merge(temp_poss[["Date", "Def Pen", "Att Pen", "Succ%"]], on="Date")
    except ValueError:
        continue


    #Misc Stats  
    temp_misc = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/misc/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_misc.columns = temp_misc.columns.droplevel()
    try:
        team_data = team_data.merge(temp_misc[["Date", "CrdY", "CrdR", "Fls", "Fld", "Off", "TklW", "Recov"]], on="Date")
    except ValueError:
        continue


    #Def Stats  
    temp_defence = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/defense/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_defence.columns = temp_defence.columns.droplevel()
    try:
        team_data = team_data.merge(temp_defence[["Date", "Int", "Clr", "Err"]], on="Date")
    except ValueError:
        continue  
    
    
    #only select matches where they are in the premier leage
    team_data = team_data[team_data["Comp"] == "Série A"]
    
    #Add a column to the df naming the team 
    team_data["Team"] = squad_team[i][1].replace("-", " ")
    all_matches.append(team_data)
    print(i)
    time.sleep(10)

time.sleep(30)


#Loop 2, teams in team_url 6-10. 
#Shooting Stats
for i in np.arange(5,10): 
    
    #Using requests to make html request
    data = requests.get(team_urls[i])

    #creating df of general stats from scores and fixtures tab
    matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
    
    #Shooting stats
    temp_shooting = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/shooting/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_shooting.columns = temp_shooting.columns.droplevel()
    try:
        team_data = matches.merge(temp_shooting[["Date", "Sh", "SoT", "Dist", "G-xG", "xG"]], on="Date")
    except ValueError:
        continue


    #Passing Stats 
    temp_pass = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/passing/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_pass.columns = temp_pass.columns.droplevel()
    try:
        team_data = team_data.merge(temp_pass[["Date", "Cmp", "xA", "KP", "1/3", "PPA", "PrgP"]], on="Date")
    except ValueError:
        continue


    #Possession Stats   
    temp_poss = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/possession/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_poss.columns = temp_poss.columns.droplevel()
    try:
        team_data = team_data.merge(temp_poss[["Date", "Def Pen", "Att Pen", "Succ%"]], on="Date")
    except ValueError:
        continue


    #Misc Stats  
    temp_misc = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/misc/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_misc.columns = temp_misc.columns.droplevel()
    try:
        team_data = team_data.merge(temp_misc[["Date", "CrdY", "CrdR", "Fls", "Fld", "Off", "TklW", "Recov"]], on="Date")
    except ValueError:
        continue


    #Def Stats  
    temp_defence = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/defense/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_defence.columns = temp_defence.columns.droplevel()
    try:
        team_data = team_data.merge(temp_defence[["Date", "Int", "Clr", "Err"]], on="Date")
    except ValueError:
        continue  
    
    
    #only select matches where they are in the premier leage
    team_data = team_data[team_data["Comp"] == "Série A"]
    
    #Add a column to the df naming the team 
    team_data["Team"] = squad_team[i][1].replace("-", " ")
    all_matches.append(team_data)
    print(i)
    time.sleep(10)


time.sleep(30)

#Loop 3, teams in team_url 11-15. 
#Shooting Stats
for i in np.arange(10,15): 
    
    #Using requests to make html request
    data = requests.get(team_urls[i])

    #creating df of general stats from scores and fixtures tab
    matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
    
    #Shooting stats
    temp_shooting = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/shooting/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_shooting.columns = temp_shooting.columns.droplevel()
    try:
        team_data = matches.merge(temp_shooting[["Date", "Sh", "SoT", "Dist", "G-xG", "xG"]], on="Date")
    except ValueError:
        continue


    #Passing Stats 
    temp_pass = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/passing/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_pass.columns = temp_pass.columns.droplevel()
    try:
        team_data = team_data.merge(temp_pass[["Date", "Cmp", "xA", "KP", "1/3", "PPA", "PrgP"]], on="Date")
    except ValueError:
        continue


    #Possession Stats   
    temp_poss = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/possession/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_poss.columns = temp_poss.columns.droplevel()
    try:
        team_data = team_data.merge(temp_poss[["Date", "Def Pen", "Att Pen", "Succ%"]], on="Date")
    except ValueError:
        continue


    #Misc Stats  
    temp_misc = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/misc/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_misc.columns = temp_misc.columns.droplevel()
    try:
        team_data = team_data.merge(temp_misc[["Date", "CrdY", "CrdR", "Fls", "Fld", "Off", "TklW", "Recov"]], on="Date")
    except ValueError:
        continue


    #Def Stats  
    temp_defence = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/defense/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_defence.columns = temp_defence.columns.droplevel()
    try:
        team_data = team_data.merge(temp_defence[["Date", "Int", "Clr", "Err"]], on="Date")
    except ValueError:
        continue  
    
    
    #only select matches where they are in the premier leage
    team_data = team_data[team_data["Comp"] == "Série A"]
    
    #Add a column to the df naming the team 
    team_data["Team"] = squad_team[i][1].replace("-", " ")
    all_matches.append(team_data)
    print(i)
    time.sleep(10)

time.sleep(30)


#Loop 4, teams in team_url 16-20. 
#Shooting Stats
for i in np.arange(15,20):   
    
    #Using requests to make html request
    data = requests.get(team_urls[i])

    #creating df of general stats from scores and fixtures tab
    matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
    
    #Shooting stats
    temp_shooting = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/shooting/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_shooting.columns = temp_shooting.columns.droplevel()
    try:
        team_data = matches.merge(temp_shooting[["Date", "Sh", "SoT", "Dist", "G-xG", "xG"]], on="Date")
    except ValueError:
        continue


    #Passing Stats 
    temp_pass = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/passing/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_pass.columns = temp_pass.columns.droplevel()
    try:
        team_data = team_data.merge(temp_pass[["Date", "Cmp", "xA", "KP", "1/3", "PPA", "PrgP"]], on="Date")
    except ValueError:
        continue


    #Possession Stats   
    temp_poss = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/possession/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_poss.columns = temp_poss.columns.droplevel()
    try:
        team_data = team_data.merge(temp_poss[["Date", "Def Pen", "Att Pen", "Succ%"]], on="Date")
    except ValueError:
        continue


    #Misc Stats  
    temp_misc = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/misc/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_misc.columns = temp_misc.columns.droplevel()
    try:
        team_data = team_data.merge(temp_misc[["Date", "CrdY", "CrdR", "Fls", "Fld", "Off", "TklW", "Recov"]], on="Date")
    except ValueError:
        continue


    #Def Stats  
    temp_defence = pd.read_html("https://fbref.com/en/squads/"+squad_team[i][0]+"/2023/matchlogs/all_comps/defense/"+squad_team[i][1]+"-Match-Logs-All-Competitions")[0]
    temp_defence.columns = temp_defence.columns.droplevel()
    try:
        team_data = team_data.merge(temp_defence[["Date", "Int", "Clr", "Err"]], on="Date")
    except ValueError:
        continue  
    
    
    #only select matches where they are in the premier leage
    team_data = team_data[team_data["Comp"] == "Série A"]
    
    #Add a column to the df naming the team 
    team_data["Team"] = squad_team[i][1].replace("-", " ")
    all_matches.append(team_data)
    print(i)
    time.sleep(10)

#Concatenate all seperate dataframes relating to each individual team into 1 large df.
match_df = pd.concat(all_matches)

#Print how many games each team have played to check this is up to date
print(match_df["Team"].value_counts())

#Save initial output df as and post process in seperate analysis script. Can get HTTPError if make too many requests, so save as soon as file 
#created and dropped initial columns not needed. 

#Save final df as a csv
# match_df.to_csv(r"C:\Users\tt13\football\matches_20230331.csv") #append today's date

match_df.to_csv('/Users/tom/Documents/python/football/brazil/brazil_matches_20230502.csv') #append today's date



 