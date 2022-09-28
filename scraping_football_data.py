# -*- coding: utf-8 -*-
"""
Python script created by Tom Thorp 23/09/2022. 
This script reads in stats about a football league, and sorts them into panda dataframes,
saving them in csv output for further use in external scripts (e.g. plotting and ML). 
 
Below scraping method adapted from (link continues over 3 lines): https://github.com/oluwatosin17/
Web-Scraping-Football-Matches-From-The-EPL-With-Python-/blob/main/
Web%20Scraping%20Football%20Matches%20From%20The%20EPL%20.ipynb 
 
"""
# Importing libraries
import pandas as pd
import numpy as np
import requests 
from bs4 import BeautifulSoup
import os

# Defining the url of the website, directing to league specific stats tab.
url = "https://fbref.com/en/comps/29/Allsvenskan-Stats"

#Using requests to make html request 
data = requests.get(url)

#Convert html text into string
soup = BeautifulSoup(data.text)
standing_table = soup.select("table.stats_table")[0]

#Picking out "a tag" which is the anchor item - picking out all atag items. 
links  = standing_table.find_all("a")

#Looping through links variable to get the href property from the atag.
links = [l.get("href") for l in links]

#Pick out only squads in list 
links = [l for l in links if '/squads/' in l]

#Prepending website domain to the link to provide full link. 
team_urls = [f"https://fbref.com{l}" for l in links]

#Putting teams in alphabetical order - sorting by team name using key function
team_urls_sorted = sorted(team_urls, key=lambda x: x[37:43])

#Create list of teams and sort
team_names =[]
for i in range(np.size(team_urls)):
    team_names_temp = team_urls[i][37:]
    team_names = np.append(team_names_temp, team_names)    
team_names_sorted = sorted(team_names) 
                                                              
#Loop through all teams and output data to csv
for i in range(np.size(team_urls)):

    #Picking out 1 teams url
    team_url = team_urls_sorted[i]
    
    #Using requests to make html request
    data = requests.get(team_url)
    
    #Using pandas heat_html command to convert table into pandas df
    matches = pd.read_html(data.text, match = "Scores & Fixtures")
    
    #creating single column for teamname to append to full df
    matches[0]['Team'] = team_names_sorted[i]
    
    
# At this point start to add in more data to the df  
    
# testing git    
    
    
    #writing out df to csv
#    os.makedirs('/Users/tom/Documents/python/football/team_stats/', exist_ok=True)  
#    matches[0].to_csv('/Users/tom/Documents/python/football/team_stats/'+team_names_sorted[i]+'.csv')







